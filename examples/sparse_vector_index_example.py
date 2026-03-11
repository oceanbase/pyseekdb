################################################################################
# Sparse Vector Index Example
# install dependencies first:
# pip install "bm25s[full]" sentence-transformers
# Note: bm25s[full] is used for bm25 embedding function,
# sentence-transformers is used for hugging face embedding function and
# reranking with model
################################################################################

from __future__ import annotations

import contextlib
from typing import Any, Literal

from transformers.utils import logging as hf_logging

from pyseekdb import Client, HNSWConfiguration, K, Schema, SparseVectorIndexConfig
from pyseekdb.utils.embedding_functions import BM25SparseEmbeddingFunction

hf_logging.set_verbosity_error()

# 1. Initialize client (Embedded mode; creates seekdb.db in the current directory)
client = Client()

# Clean up stale collections from previous runs
for name in ["demo_sparse_collection", "hybrid_demo"]:
    with contextlib.suppress(Exception):
        client.delete_collection(name)

# 2. Define schema: BM25 sparse vector index using document content (K.DOCUMENT) as source
sparse_config = SparseVectorIndexConfig(
    embedding_function=BM25SparseEmbeddingFunction(k=1.2, b=0.75),
    source_key=K.DOCUMENT,
    drop_ratio_search=0.2,
)

schema = Schema(sparse_vector_index=sparse_config)

# 3. Create collection
collection_name = "demo_sparse_collection"
collection = client.create_collection(name=collection_name, schema=schema)

print(f"Collection '{collection_name}' created successfully.")

collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "Artificial Intelligence and machine learning are related.",
    ],
)

# 5. Run sparse vector search
results = collection.query(
    query_texts=["machine learning"],
    query_key=K.SPARSE_EMBEDDING,
    n_results=2,
)

print("Sparse Search Results:")
for i, doc_id in enumerate(results["ids"][0]):
    print(f"Rank {i + 1}: ID={doc_id}, Distance={results['distances'][0][i]}, Document={results['documents'][0][i]}")


# =====================================================================
# Hybrid search with pluggable reranking strategies
# =====================================================================

RerankMethod = Literal["rrf", "score", "model"]


def _normalize_scores(distances: list[float], higher_is_better: bool) -> list[float]:
    """Normalize a list of distance/score values to [0, 1]."""
    if not distances:
        return []
    lo, hi = min(distances), max(distances)
    if hi == lo:
        return [1.0] * len(distances)
    if higher_is_better:
        return [(d - lo) / (hi - lo) for d in distances]
    else:
        return [(hi - d) / (hi - lo) for d in distances]


def _rerank_rrf(
    dense_ids: list[str],
    sparse_ids: list[str],
    dense_results: dict[str, Any],
    sparse_results: dict[str, Any],
    rrf_k: int,
) -> dict[str, float]:
    """Reciprocal Rank Fusion: merge by rank position, ignore score magnitude."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def _rerank_score(
    dense_ids: list[str],
    sparse_ids: list[str],
    dense_results: dict[str, Any],
    sparse_results: dict[str, Any],
    alpha: float,
) -> dict[str, float]:
    """
    Score-based fusion: normalize both distance lists to [0, 1] and combine
    with ``alpha * dense + (1 - alpha) * sparse``.

    Dense distances are L2/cosine (lower = better); sparse distances are
    inner-product (higher = better).
    """
    dense_distances = dense_results["distances"][0] if dense_ids else []
    sparse_distances = sparse_results["distances"][0] if sparse_ids else []

    dense_norm = _normalize_scores(dense_distances, higher_is_better=False)
    sparse_norm = _normalize_scores(sparse_distances, higher_is_better=True)

    scores: dict[str, float] = {}
    for i, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha * dense_norm[i]
    for i, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 - alpha) * sparse_norm[i]
    return scores


def _rerank_model(
    query_text: str,
    candidate_ids: list[str],
    candidate_docs: dict[str, str],
    model_name: str,
) -> dict[str, float]:
    """
    Cross-encoder reranking: score every (query, document) pair with a
    cross-encoder model and return the relevance scores.

    Requires: ``pip install sentence-transformers``
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [(query_text, candidate_docs[doc_id]) for doc_id in candidate_ids]
    raw_scores: list[float] = model.predict(pairs).tolist()
    return dict(zip(candidate_ids, raw_scores, strict=True))


def perform_hybrid_search(
    collection,
    query_text: str,
    top_k: int = 5,
    rerank: RerankMethod = "rrf",
    *,
    rrf_k: int = 60,
    alpha: float = 0.5,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, Any]:
    """
    Hybrid search combining dense and sparse retrieval with pluggable reranking.

    Args:
        collection: pyseekdb collection (must have both dense and sparse indexes).
        query_text: The query string.
        top_k: Number of final results to return.
        rerank: Reranking strategy:
            - ``"rrf"``:   Reciprocal Rank Fusion (rank-based, no model).
            - ``"score"``: Weighted score fusion after min-max normalization.
            - ``"model"``: Cross-encoder reranker (best quality, needs
              ``sentence-transformers``).
        rrf_k: RRF constant (only used when ``rerank="rrf"``). Default 60.
        alpha: Weight for dense scores in score fusion (only used when
            ``rerank="score"``).  ``0.0`` = sparse only, ``1.0`` = dense only.
        rerank_model: HuggingFace cross-encoder model name (only used when
            ``rerank="model"``).

    Returns:
        ``{"ids": [...], "documents": [...], "scores": [...]}``
    """
    # 1. Retrieve candidates from both indexes
    dense_results = collection.query(query_texts=[query_text], n_results=top_k)
    sparse_results = collection.query(
        query_texts=[query_text],
        query_key=K.SPARSE_EMBEDDING,
        n_results=top_k,
    )

    dense_ids: list[str] = dense_results["ids"][0] if dense_results["ids"] else []
    sparse_ids: list[str] = sparse_results["ids"][0] if sparse_results["ids"] else []

    # 2. Rerank
    if rerank == "rrf":
        scores = _rerank_rrf(dense_ids, sparse_ids, dense_results, sparse_results, rrf_k)
    elif rerank == "score":
        scores = _rerank_score(dense_ids, sparse_ids, dense_results, sparse_results, alpha)
    elif rerank == "model":
        all_ids = list(dict.fromkeys(dense_ids + sparse_ids))
        if not all_ids:
            return {"ids": [], "documents": [], "scores": []}
        fetched = collection.get(ids=all_ids)
        id_to_doc = dict(zip(fetched["ids"], fetched["documents"], strict=True))
        scores = _rerank_model(query_text, all_ids, id_to_doc, rerank_model)
    else:
        raise ValueError(f"Unknown rerank method: {rerank!r}. Use 'rrf', 'score', or 'model'.")

    # 3. Sort by score descending, take top_k
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    if not sorted_ids:
        return {"ids": [], "documents": [], "scores": []}

    # 4. Fetch documents in final order
    fetched = collection.get(ids=sorted_ids)
    id_to_doc = dict(zip(fetched["ids"], fetched["documents"], strict=True))

    return {
        "ids": sorted_ids,
        "documents": [id_to_doc.get(sid, "") for sid in sorted_ids],
        "scores": [round(scores[sid], 6) for sid in sorted_ids],
    }


# =====================================================================
# Demo: hybrid search with all three reranking strategies
# =====================================================================

hybrid_collection_name = "hybrid_demo"

schema = Schema(
    vector_index=HNSWConfiguration(dimension=384),
    sparse_vector_index=SparseVectorIndexConfig(
        embedding_function=BM25SparseEmbeddingFunction(),
        source_key=K.DOCUMENT,
    ),
)
collection = client.create_collection(hybrid_collection_name, schema=schema)

docs = [
    "Apple uses advanced chips in iPhones.",
    "Apples are red and delicious fruits.",
    "Oranges are rich in Vitamin C.",
    "Apple pie is a popular dessert.",
    "The tech giant Apple released a new laptop.",
]
ids = [str(i) for i in range(len(docs))]
collection.add(documents=docs, ids=ids)

query = "apple fruit"

# --- RRF (default, no model needed) ---
print(f"\n=== Query: '{query}' | rerank='rrf' ===")
results = perform_hybrid_search(collection, query, rerank="rrf")
for i, doc_id in enumerate(results["ids"]):
    print(f"  {i + 1}. ID={doc_id} score={results['scores'][i]:.6f} | {results['documents'][i]}")

# --- Score-based fusion (alpha=0.5 gives equal weight to dense & sparse) ---
print(f"\n=== Query: '{query}' | rerank='score', alpha=0.5 ===")
results = perform_hybrid_search(collection, query, rerank="score", alpha=0.5)
for i, doc_id in enumerate(results["ids"]):
    print(f"  {i + 1}. ID={doc_id} score={results['scores'][i]:.6f} | {results['documents'][i]}")

# --- Cross-encoder reranker (requires sentence-transformers) ---
try:
    print(f"\n=== Query: '{query}' | rerank='model' ===")
    results = perform_hybrid_search(collection, query, rerank="model")
    for i, doc_id in enumerate(results["ids"]):
        print(f"  {i + 1}. ID={doc_id} score={results['scores'][i]:.6f} | {results['documents'][i]}")
except ImportError:
    print("\n(Skipping model rerank — install sentence-transformers to enable)")
