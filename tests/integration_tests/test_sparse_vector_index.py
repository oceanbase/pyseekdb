"""
Integration tests for sparse vector index support.

Tests create_collection (with Schema + SparseVectorIndexConfig),
collection.add, collection.update, collection.upsert, and collection.query
with sparse vector embeddings.

Run with embedded mode:
    pytest tests/integration_tests/test_sparse_vector_index.py -k embedded -v -s
"""

import contextlib
import importlib
import time
import uuid
from typing import Any

import pytest

from pyseekdb import (
    HNSWConfiguration,
    K,
    Schema,
    SparseVectorIndexConfig,
    VectorIndexConfig,
)
from pyseekdb.client.sparse_embedding_function import (
    Documents,
    SparseEmbeddingFunction,
    SparseVector,
    SparseVectors,
    register_sparse_embedding_function,
)
from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import BM25SparseEmbeddingFunction
from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import HuggingFaceSparseEmbeddingFunction


def _bm25_available() -> bool:
    return importlib.util.find_spec("bm25s") is not None


def _splade_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


# ── Fake sparse embedding function for deterministic testing ─────────


class FakeSparseEF(SparseEmbeddingFunction):
    """
    Deterministic sparse embedding function for integration testing.

    Produces sparse vectors based on word hashing: each unique word in the
    document gets a dimension (hash(word) % 1000) with weight 1/(position+1).
    """

    def __call__(self, documents: Documents) -> SparseVectors:
        if isinstance(documents, str):
            documents = [documents]
        result = []
        for doc in documents:
            words = doc.lower().split()
            sv_dict = {}
            for i, word in enumerate(words):
                dim = abs(hash(word)) % 1000
                sv_dict[dim] = round(1.0 / (i + 1), 4)
            result.append(SparseVector.from_dict(sv_dict) if sv_dict else SparseVector.from_dict({0: 0.01}))
        return result

    @staticmethod
    def name() -> str:
        return "fake_sparse_bm25"

    def get_config(self) -> dict[str, Any]:
        return {"type": "fake"}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "FakeSparseEF":
        return FakeSparseEF()


# ── Helpers ──────────────────────────────────────────────────────────

DIMENSION = 3


def _unique_name(prefix: str = "test_sparse") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def _cleanup_collection(client, name: str):
    with contextlib.suppress(Exception):
        client.delete_collection(name)


def _make_sparse_schema():
    """Create a Schema with HNSW(dim=3) + sparse index, no dense embedding function."""
    register_sparse_embedding_function(FakeSparseEF)
    schema = Schema(
        vector_index=HNSWConfiguration(dimension=DIMENSION, distance="l2"),
        sparse_vector_index=SparseVectorIndexConfig(
            embedding_function=FakeSparseEF(),
            source_key=K.DOCUMENT,
        ),
    )
    # VectorIndexConfig.__post_init__ forces DefaultEmbeddingFunction;
    # override to None so create_collection doesn't dimension-check against it.
    schema.vector_index.embedding_function = None
    return schema


def _create_sparse_collection(db_client, name):
    """Helper: create a collection with dense+sparse index and no dense EF."""
    return db_client.create_collection(name=name, schema=_make_sparse_schema())


# ── Tests ────────────────────────────────────────────────────────────


class TestCreateCollectionWithSparseIndex:
    """Test create_collection with Schema containing SparseVectorIndexConfig."""

    def test_create_collection_with_sparse_schema(self, db_client):
        """Create collection with dense + sparse indexes via Schema."""
        name = _unique_name("create_sparse")
        try:
            collection = _create_sparse_collection(db_client, name)
            assert collection is not None
            assert collection.name == name
            assert collection.sparse_vector_index_config is not None
            assert collection.sparse_embedding_function is not None
            print(f"   Created collection '{name}' with sparse index")
        finally:
            _cleanup_collection(db_client, name)

    def test_create_collection_with_sparse_only(self, db_client):
        name = _unique_name("create_sparse")
        try:
            schema = Schema(
                sparse_vector_index=SparseVectorIndexConfig(
                    embedding_function=FakeSparseEF(),
                    source_key=K.DOCUMENT,
                ),
            )

            collection = db_client.create_collection(name=name, schema=schema)
            assert collection is not None
            assert collection.name == name
            assert collection.sparse_vector_index_config is not None
            assert collection.sparse_embedding_function is not None
            print(f"   Created collection '{name}' with sparse index")
        finally:
            _cleanup_collection(db_client, name)

    def test_create_collection_without_sparse(self, db_client):
        """Create collection without sparse index (backward compatibility)."""
        name = _unique_name("create_no_sparse")
        try:
            collection = db_client.get_or_create_collection(
                name=name,
                configuration=HNSWConfiguration(dimension=DIMENSION, distance="l2"),
                embedding_function=None,
            )
            assert collection is not None
            assert collection.sparse_vector_index_config is None
            assert collection.sparse_embedding_function is None
            print(f"   Created collection '{name}' without sparse index")
        finally:
            _cleanup_collection(db_client, name)

    def test_get_or_create_with_sparse_schema(self, db_client):
        """get_or_create_collection correctly passes schema through."""
        name = _unique_name("get_or_create_sparse")
        try:
            collection = db_client.get_or_create_collection(name=name, schema=_make_sparse_schema())
            assert collection is not None
            assert collection.sparse_vector_index_config is not None
            print("   get_or_create_collection with sparse: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestCollectionAddWithSparse:
    """Test collection.add() with sparse vector auto-generation."""

    def test_add_single_item(self, db_client):
        """Add a single item; sparse embeddings auto-generated from document."""
        name = _unique_name("add_single")
        try:
            collection = _create_sparse_collection(db_client, name)
            test_id = str(uuid.uuid4())
            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="machine learning algorithms",
                metadatas={"category": "AI"},
            )
            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["ids"][0] == test_id
            assert results["documents"][0] == "machine learning algorithms"
            print("   Added single item with sparse vector: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_add_multiple_items(self, db_client):
        """Add multiple items; each gets its own sparse vector."""
        name = _unique_name("add_multi")
        try:
            collection = _create_sparse_collection(db_client, name)
            ids = [str(uuid.uuid4()) for _ in range(3)]
            collection.add(
                ids=ids,
                embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                documents=[
                    "machine learning tutorial",
                    "python programming guide",
                    "neural network introduction",
                ],
                metadatas=[
                    {"category": "AI"},
                    {"category": "Programming"},
                    {"category": "AI"},
                ],
            )
            results = collection.get(ids=ids)
            assert len(results["ids"]) == 3
            print("   Added 3 items with sparse vectors: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestCollectionUpdateWithSparse:
    """Test collection.update() with sparse vector regeneration."""

    def _setup(self, db_client, name):
        collection = _create_sparse_collection(db_client, name)
        test_id = str(uuid.uuid4())
        collection.add(
            ids=test_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="original document content",
            metadatas={"version": 1},
        )
        return collection, test_id

    def test_update_document_regenerates_sparse(self, db_client):
        """Updating document should regenerate sparse vector."""
        name = _unique_name("update_doc")
        try:
            collection, test_id = self._setup(db_client, name)
            collection.update(
                ids=test_id,
                embeddings=[1.1, 2.1, 3.1],
                documents="updated document with new content",
            )
            results = collection.get(ids=test_id)
            assert results["documents"][0] == "updated document with new content"
            print("   Updated document with sparse regeneration: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_update_metadata_only(self, db_client):
        """Updating only metadata should still work."""
        name = _unique_name("update_meta")
        try:
            collection, test_id = self._setup(db_client, name)
            collection.update(
                ids=test_id,
                metadatas={"version": 2, "updated": True},
            )
            results = collection.get(ids=test_id)
            assert results["metadatas"][0]["version"] == 2
            assert results["metadatas"][0]["updated"] is True
            print("   Updated metadata: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestCollectionUpsertWithSparse:
    """Test collection.upsert() with sparse vector support."""

    def test_upsert_insert_new_item(self, db_client):
        """Upsert a new item (insert path)."""
        name = _unique_name("upsert_new")
        try:
            collection = _create_sparse_collection(db_client, name)
            test_id = str(uuid.uuid4())
            collection.upsert(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="upserted new document",
                metadatas={"source": "upsert"},
            )
            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "upserted new document"
            print("   Upsert insert path with sparse: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_upsert_update_existing_item(self, db_client):
        """Upsert an existing item (update path)."""
        name = _unique_name("upsert_exist")
        try:
            collection = _create_sparse_collection(db_client, name)
            test_id = str(uuid.uuid4())

            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="original document",
                metadatas={"version": 1},
            )

            collection.upsert(
                ids=test_id,
                embeddings=[1.1, 2.1, 3.1],
                documents="upserted updated document",
                metadatas={"version": 2},
            )

            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "upserted updated document"
            assert results["metadatas"][0]["version"] == 2
            print("   Upsert update path with sparse: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_upsert_metadata_only_does_not_require_document(self, db_client):
        """Upsert metadata-only should not regenerate sparse vectors from documents."""
        name = _unique_name("upsert_meta_only")
        try:
            collection = _create_sparse_collection(db_client, name)
            test_id = str(uuid.uuid4())

            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="original document",
                metadatas={"version": 1},
            )

            # Source key is document; metadata-only upsert must still work.
            collection.upsert(
                ids=test_id,
                metadatas={"version": 2, "patched": True},
            )

            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["metadatas"][0]["version"] == 2
            assert results["metadatas"][0]["patched"] is True
            print("   Upsert metadata-only with sparse source=document: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestCollectionQueryWithSparse:
    """Test collection.query() with sparse vector index (query_key=K.SPARSE_EMBEDDING)."""

    def _setup_with_data(self, db_client, name, sparse_ef=None):
        if sparse_ef is None:
            sparse_ef = FakeSparseEF()
        schema = Schema(
            vector_index=VectorIndexConfig(
                hnsw=HNSWConfiguration(dimension=DIMENSION, distance="l2"), embedding_function=None
            ),
            sparse_vector_index=SparseVectorIndexConfig(
                embedding_function=sparse_ef,
                source_key=K.DOCUMENT,
            ),
        )
        collection = db_client.create_collection(name=name, schema=schema)

        ids = [str(uuid.uuid4()) for _ in range(5)]
        collection.add(
            ids=ids,
            embeddings=[
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [1.1, 2.1, 3.1],
                [2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2],
            ],
            documents=[
                "machine learning algorithms for classification",
                "python programming tutorial for beginners",
                "advanced deep learning neural networks",
                "data science with python and pandas",
                "introduction to natural language processing",
            ],
            metadatas=[
                {"category": "AI", "score": 95},
                {"category": "Programming", "score": 88},
                {"category": "AI", "score": 92},
                {"category": "Data Science", "score": 90},
                {"category": "AI", "score": 85},
            ],
        )
        return collection, ids, sparse_ef

    def test_sparse_query_with_texts(self, db_client):
        """Query using query_texts + query_key=K.SPARSE_EMBEDDING."""
        name = _unique_name("query_sparse_text")
        try:
            collection, _ids, _ = self._setup_with_data(db_client, name)
            results = collection.query(
                query_texts=["machine learning"],
                query_key=K.SPARSE_EMBEDDING,
                n_results=3,
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) == 1
            assert len(results["ids"][0]) > 0
            print(f"   Sparse text query returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_sparse_query_with_sparse_vector_directly(self, db_client):
        """Direct sparse vector query should be rejected."""
        name = _unique_name("query_sparse_vec")
        try:
            collection, _ids, sparse_ef = self._setup_with_data(db_client, name)
            sv = sparse_ef(["machine learning"])[0]
            with pytest.raises(ValueError, match="query_embeddings is not supported"):
                collection.query(
                    query_embeddings=sv,
                    query_key=K.SPARSE_EMBEDDING,
                    n_results=3,
                )
            print("   Direct sparse vector query rejected as expected: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_sparse_query_with_dict_directly(self, db_client):
        """Direct sparse dict query should be rejected."""
        name = _unique_name("query_sparse_dict")
        try:
            collection, _ids, sparse_ef = self._setup_with_data(db_client, name)
            sv = sparse_ef(["machine learning"])[0]
            with pytest.raises(ValueError, match="query_embeddings is not supported"):
                collection.query(
                    query_embeddings=sv.embeddings,
                    query_key=K.SPARSE_EMBEDDING,
                    n_results=3,
                )
            print("   Direct sparse dict query rejected as expected: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_sparse_query_with_string_query_key(self, db_client):
        """Query using string query_key '#sparse_embedding'."""
        name = _unique_name("query_sparse_string_key")
        try:
            collection, _ids, _ = self._setup_with_data(db_client, name)
            results = collection.query(
                query_texts=["machine learning"],
                query_key="#sparse_embedding",
                n_results=3,
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) == 1
            assert len(results["ids"][0]) > 0
            print("   Sparse query with string query_key returned results: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_dense_query_still_works(self, db_client):
        """Regular dense vector query still works on a collection with sparse index."""
        name = _unique_name("query_dense_with_sparse")
        try:
            collection, _ids, _ = self._setup_with_data(db_client, name)
            collection.refresh()
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3,
            )
            assert results is not None
            assert len(results["ids"][0]) > 0
            print(f"   Dense query on sparse-enabled collection returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)

    @pytest.mark.skipif(not _bm25_available(), reason="bm25s not installed")
    def test_sparse_query_with_bm25(self, db_client):
        """Query using BM25 sparse vector."""
        name = _unique_name("query_sparse_bm25")
        try:
            collection, _ids, _ = self._setup_with_data(db_client, name, sparse_ef=BM25SparseEmbeddingFunction())
            results = collection.query(
                query_texts=["machine learning"],
                query_key=K.SPARSE_EMBEDDING,
                n_results=3,
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) == 1
            assert len(results["ids"][0]) > 0
            print(f"   Sparse BM25 query returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)

    @pytest.mark.skipif(not _splade_available(), reason="sentence_transformers not installed")
    def test_sparse_query_with_splade(self, db_client):
        """Query using SPLADE sparse vector."""
        name = _unique_name("query_sparse_splade")
        try:
            collection, _ids, _ = self._setup_with_data(db_client, name, sparse_ef=HuggingFaceSparseEmbeddingFunction())
            results = collection.query(
                query_texts=["machine learning"],
                query_key=K.SPARSE_EMBEDDING,
                n_results=3,
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) == 1
            assert len(results["ids"][0]) > 0
            print(f"   Sparse SPLADE query returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseWithMetadataSource:
    """Test sparse vector generation from metadata field instead of document."""

    def test_sparse_from_metadata_field(self, db_client):
        """Sparse vectors generated from a metadata field (source_key='title')."""
        name = _unique_name("sparse_meta_src")
        sparse_ef = FakeSparseEF()
        schema = Schema(
            vector_index=VectorIndexConfig(
                hnsw=HNSWConfiguration(dimension=DIMENSION, distance="l2"), embedding_function=None
            ),
            sparse_vector_index=SparseVectorIndexConfig(
                embedding_function=sparse_ef,
                source_key="title",
            ),
        )

        try:
            collection = db_client.create_collection(name=name, schema=schema)
            test_id = str(uuid.uuid4())
            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="full document body here",
                metadatas={"title": "machine learning overview", "category": "AI"},
            )

            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "full document body here"
            print("   Sparse from metadata['title'] source: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseIndexSqlGeneration:
    """Test that the CREATE TABLE SQL is correctly generated with sparse index options."""

    def test_sparse_with_prune_and_refine(self, db_client):
        """Create collection with sparse index tuning parameters."""
        name = _unique_name("sparse_opts")
        sparse_ef = FakeSparseEF()
        schema = Schema(
            vector_index=VectorIndexConfig(
                hnsw=HNSWConfiguration(dimension=DIMENSION, distance="l2"), embedding_function=None
            ),
            sparse_vector_index=SparseVectorIndexConfig(
                embedding_function=sparse_ef,
                source_key=K.DOCUMENT,
                type="sindi",
                prune=True,
                refine=True,
                drop_ratio_build=0.1,
                drop_ratio_search=0.2,
                refine_k=4.0,
            ),
        )

        try:
            collection = db_client.create_collection(name=name, schema=schema)
            assert collection is not None
            assert collection.sparse_vector_index_config is not None

            test_id = str(uuid.uuid4())
            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="test with prune and refine options",
            )
            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            print("   Sparse index with prune/refine options: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseCollectionDeleteOperations:
    """Test delete operations on collections with sparse vector index."""

    def _setup_with_data(self, db_client, name):
        collection = _create_sparse_collection(db_client, name)
        ids = [str(uuid.uuid4()) for _ in range(5)]
        collection.add(
            ids=ids,
            embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]],
            documents=[
                "machine learning algorithms",
                "python programming guide",
                "deep learning neural networks",
                "data science with python",
                "natural language processing",
            ],
            metadatas=[
                {"category": "AI", "priority": 1},
                {"category": "Programming", "priority": 2},
                {"category": "AI", "priority": 3},
                {"category": "Data Science", "priority": 4},
                {"category": "AI", "priority": 5},
            ],
        )
        return collection, ids

    def test_delete_by_id(self, db_client):
        """Delete a single item by ID from a sparse collection."""
        name = _unique_name("del_id")
        try:
            collection, ids = self._setup_with_data(db_client, name)
            assert collection.count() == 5

            collection.delete(ids=ids[0])
            assert collection.count() == 4

            results = collection.get(ids=ids[0])
            assert len(results["ids"]) == 0
            print("   Delete by ID on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_delete_multiple_by_ids(self, db_client):
        """Delete multiple items by IDs from a sparse collection."""
        name = _unique_name("del_multi")
        try:
            collection, ids = self._setup_with_data(db_client, name)
            collection.delete(ids=ids[:3])
            assert collection.count() == 2

            remaining = collection.get(ids=ids[3:])
            assert len(remaining["ids"]) == 2
            print("   Delete multiple by IDs on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_delete_by_metadata_filter(self, db_client):
        """Delete items matching a metadata filter on a sparse collection."""
        name = _unique_name("del_meta")
        try:
            collection, _ids = self._setup_with_data(db_client, name)
            collection.delete(where={"category": {"$eq": "AI"}})

            remaining = collection.get(limit=100)
            for meta in remaining["metadatas"]:
                assert meta["category"] != "AI"
            print("   Delete by metadata filter on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_delete_by_document_filter(self, db_client):
        """Delete items matching a document filter on a sparse collection."""
        name = _unique_name("del_doc")
        try:
            collection, _ids = self._setup_with_data(db_client, name)
            original_count = collection.count()

            collection.delete(where_document={"$contains": "python"})

            new_count = collection.count()
            assert new_count < original_count
            results = collection.get(where_document={"$contains": "python"})
            assert len(results["ids"]) == 0
            print("   Delete by document filter on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseCollectionCountAndPeek:
    """Test count and peek operations on collections with sparse vector index."""

    def test_count_empty_collection(self, db_client):
        """Newly created sparse collection has count zero."""
        name = _unique_name("count_empty")
        try:
            collection = _create_sparse_collection(db_client, name)
            assert collection.count() == 0
            print("   Empty sparse collection count: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_count_after_add(self, db_client):
        """Count reflects the number of items added."""
        name = _unique_name("count_add")
        try:
            collection = _create_sparse_collection(db_client, name)
            ids = [str(uuid.uuid4()) for _ in range(4)]
            collection.add(
                ids=ids,
                embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                documents=[
                    "first document",
                    "second document",
                    "third document",
                    "fourth document",
                ],
            )
            assert collection.count() == 4
            print("   Sparse collection count after add: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_peek(self, db_client):
        """Peek returns a preview of items in a sparse collection."""
        name = _unique_name("peek")
        try:
            collection = _create_sparse_collection(db_client, name)
            ids = [str(uuid.uuid4()) for _ in range(3)]
            collection.add(
                ids=ids,
                embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                documents=["doc one", "doc two", "doc three"],
            )
            peeked = collection.peek(limit=2)
            assert "ids" in peeked
            assert len(peeked["ids"]) == 2
            print("   Sparse collection peek: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseCollectionGetWithFilters:
    """Test get operations with various filters on sparse collections."""

    def _setup(self, db_client, name):
        collection = _create_sparse_collection(db_client, name)
        ids = [str(uuid.uuid4()) for _ in range(4)]
        collection.add(
            ids=ids,
            embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            documents=[
                "machine learning tutorial",
                "python web development",
                "deep learning frameworks",
                "database optimization guide",
            ],
            metadatas=[
                {"topic": "AI", "level": 1},
                {"topic": "Web", "level": 2},
                {"topic": "AI", "level": 3},
                {"topic": "Database", "level": 2},
            ],
        )
        return collection, ids

    def test_get_by_metadata_where(self, db_client):
        """Get items by metadata filter on a sparse collection."""
        name = _unique_name("get_where")
        try:
            collection, _ids = self._setup(db_client, name)
            results = collection.get(where={"topic": {"$eq": "AI"}})
            assert len(results["ids"]) == 2
            for meta in results["metadatas"]:
                assert meta["topic"] == "AI"
            print("   Get by metadata where on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_get_by_document_filter(self, db_client):
        """Get items by document content filter on a sparse collection."""
        name = _unique_name("get_doc")
        try:
            collection, _ids = self._setup(db_client, name)
            results = collection.get(where_document={"$contains": "learning"})
            assert len(results["ids"]) == 2
            for doc in results["documents"]:
                assert "learning" in doc
            print("   Get by document filter on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_get_with_limit(self, db_client):
        """Get with limit on a sparse collection."""
        name = _unique_name("get_limit")
        try:
            collection, _ids = self._setup(db_client, name)
            results = collection.get(limit=2)
            assert len(results["ids"]) == 2
            print("   Get with limit on sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseSchemaChaining:
    """Test Schema.create_index chaining with sparse vector config."""

    def test_create_index_chaining(self, db_client):
        """Build schema via create_index method chaining."""
        name = _unique_name("chain")
        sparse_ef = FakeSparseEF()
        schema = (
            Schema()
            .create_index(HNSWConfiguration(dimension=DIMENSION, distance="l2"))
            .create_index(
                SparseVectorIndexConfig(
                    embedding_function=sparse_ef,
                    source_key=K.DOCUMENT,
                )
            )
        )
        schema.vector_index.embedding_function = None

        try:
            collection = db_client.create_collection(name=name, schema=schema)
            assert collection is not None
            assert collection.sparse_vector_index_config is not None
            assert collection.sparse_embedding_function is not None

            test_id = str(uuid.uuid4())
            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="chaining test document",
            )
            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "chaining test document"
            print("   Schema.create_index chaining with sparse: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseCollectionReopen:
    """Test that sparse config survives collection re-open via get_collection."""

    def test_get_collection_preserves_sparse(self, db_client):
        """After creating a sparse collection, get_collection should preserve sparse config."""
        name = _unique_name("reopen")
        try:
            collection = _create_sparse_collection(db_client, name)
            assert collection.sparse_vector_index_config is not None

            test_id = str(uuid.uuid4())
            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="document before reopen",
            )

            reopened = db_client.get_collection(name)
            assert reopened is not None
            assert reopened.name == name

            results = reopened.get(ids=test_id)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "document before reopen"
            assert isinstance(reopened.sparse_embedding_function, FakeSparseEF)
            assert reopened.sparse_vector_index_config is not None
            print("   get_collection preserves sparse data: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_get_or_create_existing_sparse_collection(self, db_client):
        """get_or_create_collection on an existing sparse collection returns it."""
        name = _unique_name("reopen_goc")
        try:
            _create_sparse_collection(db_client, name)

            reopened = db_client.get_or_create_collection(
                name=name,
                schema=_make_sparse_schema(),
            )
            assert reopened is not None
            assert reopened.sparse_vector_index_config is not None

            test_id = str(uuid.uuid4())
            reopened.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="after reopen via get_or_create",
            )
            results = reopened.get(ids=test_id)
            assert len(results["ids"]) == 1
            print("   get_or_create on existing sparse collection: OK")
        finally:
            _cleanup_collection(db_client, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
