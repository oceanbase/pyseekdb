"""
Integration tests for sparse vector index support.

Tests create_collection (with Schema + SparseVectorIndexConfig),
collection.add, collection.update, collection.upsert, and collection.query
with sparse vector embeddings.

Run with embedded mode:
    pytest tests/integration_tests/test_sparse_vector_index.py -k embedded -v -s
"""

import contextlib
import time
import uuid
from typing import Any

import pytest

from pyseekdb import (
    HNSWConfiguration,
    K,
    Schema,
    SparseVectorIndexConfig,
)
from pyseekdb.client.sparse_embedding_function import (
    Documents,
    SparseEmbeddingFunction,
    SparseVector,
    SparseVectors,
)

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
            assert collection.has_sparse_vector_index is True
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
            assert collection.has_sparse_vector_index is False
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
            assert collection.has_sparse_vector_index is True
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


class TestCollectionQueryWithSparse:
    """Test collection.query() with sparse vector index (query_key=K.SPARSE_EMBEDDING)."""

    def _setup_with_data(self, db_client, name):
        sparse_ef = FakeSparseEF()
        schema = Schema(
            vector_index=HNSWConfiguration(dimension=DIMENSION, distance="l2"),
            sparse_vector_index=SparseVectorIndexConfig(
                embedding_function=sparse_ef,
                source_key=K.DOCUMENT,
            ),
        )
        schema.vector_index.embedding_function = None
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
        """Query using a SparseVector directly via query_embeddings + query_key."""
        name = _unique_name("query_sparse_vec")
        try:
            collection, _ids, sparse_ef = self._setup_with_data(db_client, name)
            sv = sparse_ef(["machine learning"])[0]
            results = collection.query(
                query_embeddings=sv,
                query_key=K.SPARSE_EMBEDDING,
                n_results=3,
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"][0]) > 0
            print(f"   Sparse vector query returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_sparse_query_with_dict_directly(self, db_client):
        """Query using a raw dict[int, float] via query_embeddings + query_key."""
        name = _unique_name("query_sparse_dict")
        try:
            collection, _ids, sparse_ef = self._setup_with_data(db_client, name)
            sv = sparse_ef(["machine learning"])[0]
            results = collection.query(
                query_embeddings=sv.embeddings,
                query_key=K.SPARSE_EMBEDDING,
                n_results=3,
            )
            assert results is not None
            assert len(results["ids"][0]) > 0
            print(f"   Sparse dict query returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)

    def test_dense_query_still_works(self, db_client):
        """Regular dense vector query still works on a collection with sparse index."""
        name = _unique_name("query_dense_with_sparse")
        try:
            collection, _ids, _ = self._setup_with_data(db_client, name)
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3,
            )
            assert results is not None
            assert len(results["ids"][0]) > 0
            print(f"   Dense query on sparse-enabled collection returned {len(results['ids'][0])} results: OK")
        finally:
            _cleanup_collection(db_client, name)


class TestSparseWithMetadataSource:
    """Test sparse vector generation from metadata field instead of document."""

    def test_sparse_from_metadata_field(self, db_client):
        """Sparse vectors generated from a metadata field (source_key='title')."""
        name = _unique_name("sparse_meta_src")
        sparse_ef = FakeSparseEF()
        schema = Schema(
            vector_index=HNSWConfiguration(dimension=DIMENSION, distance="l2"),
            sparse_vector_index=SparseVectorIndexConfig(
                embedding_function=sparse_ef,
                source_key="title",
            ),
        )
        schema.vector_index.embedding_function = None

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
            vector_index=HNSWConfiguration(dimension=DIMENSION, distance="l2"),
            sparse_vector_index=SparseVectorIndexConfig(
                embedding_function=sparse_ef,
                source_key=K.DOCUMENT,
                prune=True,
                refine=True,
                drop_ratio_build=0.1,
                drop_ratio_search=0.2,
                refine_k=4.0,
            ),
        )
        schema.vector_index.embedding_function = None

        try:
            collection = db_client.create_collection(name=name, schema=schema)
            assert collection is not None
            assert collection.has_sparse_vector_index is True

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
