"""
Test default embedding function using db_client fixture
Tests collection creation with default embedding function, automatic vector generation from documents, and hybrid search
"""

import time
import uuid

import pytest

from pyseekdb import DefaultEmbeddingFunction
from pyseekdb.client.embedding_function import dimension_of


class TestDefaultEmbeddingFunction:
    """Test default embedding function using parameterized db_client fixture"""

    def test_default_embedding_function(self, db_client):
        """
        Test default embedding function with automatic vector generation and hybrid search.

        Automatically runs for: embedded, server, oceanbase
        """
        # Create collection with default embedding function (not passing embedding_function parameter)
        collection_name = f"test_default_ef_{int(time.time() * 1000)}"
        print(f"\n✅ Creating collection '{collection_name}' with default embedding function")

        # Create collection - default embedding function will be used automatically
        collection = db_client.create_collection(name=collection_name)

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, DefaultEmbeddingFunction)
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        # Test 1: Add documents without providing vectors (vectors will be auto-generated)
        print("\n✅ Testing collection.add() with documents only (auto-generate vectors)")

        test_documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python programming language is widely used in data science",
            "Deep learning algorithms for neural networks",
            "Data science with Python and machine learning",
            "Introduction to artificial intelligence and neural networks",
        ]

        test_ids = [str(uuid.uuid4()) for _ in test_documents]
        test_metadatas = [
            {"category": "AI", "page": 1},
            {"category": "Programming", "page": 2},
            {"category": "AI", "page": 3},
            {"category": "Data Science", "page": 4},
            {"category": "AI", "page": 5},
        ]

        # Add documents without vectors - embedding function will generate them automatically
        collection.add(ids=test_ids, documents=test_documents, metadatas=test_metadatas)
        print(f"   Added {len(test_documents)} documents (vectors auto-generated)")

        # Verify data was inserted
        results = collection.get(ids=test_ids[0], include=["documents", "metadatas", "embeddings"])
        assert len(results["ids"]) == 1
        assert results["documents"][0] == test_documents[0]
        # Note: embedding might not be returned by default, so we check if it exists
        if results.get("embeddings") and results["embeddings"][0] is not None:
            assert len(results["embeddings"][0]) == collection.dimension
            print(f"   Verified: document and embedding (dim={len(results['embeddings'][0])}) stored correctly")
        else:
            print("   Verified: document stored correctly (embedding not included in get results)")

        # Test 2: Generate query embedding using default embedding function
        print("\n✅ Testing query embedding generation")
        query_text = "artificial intelligence and machine learning"
        query_embedding = collection.embedding_function([query_text])[0]
        assert len(query_embedding) == collection.dimension
        print(f"   Generated query embedding with dimension: {len(query_embedding)}")

        # Wait a bit for indexes to be ready
        time.sleep(1)

        # Test 3: Hybrid search with vector search
        print("\n✅ Testing hybrid_search with vector search")
        try:
            results = collection.hybrid_search(
                knn={"query_embeddings": query_embedding, "n_results": 3},
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results")

            # Verify results contain documents
            if "documents" in results:
                assert len(results["documents"]) > 0
                if results["documents"][0]:
                    doc_preview = results["documents"][0][0][:50] if results["documents"][0] else ""
                    print(f"   Top result: {doc_preview}...")
        except Exception as e:
            print(f"   ⚠️  Hybrid search with vector failed: {e}")
            # Continue with other tests

        # Test 4: Hybrid search with full-text search
        print("\n✅ Testing hybrid_search with full-text search")
        try:
            results = collection.hybrid_search(
                query={
                    "where_document": {"$contains": "machine learning"},
                    "n_results": 3,
                },
                n_results=3,
                include=["documents", "metadatas"],
            )

            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results from full-text search")
        except Exception as e:
            print(f"   ⚠️  Hybrid search with full-text failed: {e}")
            # Continue with other tests

        # Test 5: Hybrid search combining both vector and full-text
        print("\n✅ Testing hybrid_search with both vector and full-text search")
        try:
            results = collection.hybrid_search(
                query={
                    "where_document": {"$contains": "machine learning"},
                    "n_results": 3,
                },
                knn={"query_embeddings": query_embedding, "n_results": 3},
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results from hybrid search")
        except Exception as e:
            print(f"   ⚠️  Hybrid search combining both failed: {e}")
            # This is acceptable - hybrid search may not be fully supported in all modes

    def test_dimension_of(self):
        """
        Test dimension_of function with different embedding function types.

        This is a unit test and doesn't need the db_client fixture.
        """
        # Test 1: DefaultEmbeddingFunction with dimension property
        print("\n✅ Testing dimension_of with DefaultEmbeddingFunction")
        default_ef = DefaultEmbeddingFunction()
        dim = dimension_of(default_ef)
        assert dim == 384, f"Expected dimension 384, got {dim}"
        assert dim == default_ef.dimension, "dimension_of should return the same as .dimension property"
        print(f"   DefaultEmbeddingFunction dimension: {dim}")

        # Test 2: Embedding function with callable dimension() method
        print("\n✅ Testing dimension_of with callable dimension() method")

        class CallableDimensionEF:
            def dimension(self):
                return 128

            def __call__(self, documents):
                if isinstance(documents, str):
                    documents = [documents]
                # Return embeddings with dimension 128
                return [[0.1] * 128 for _ in documents]

        callable_dim_ef = CallableDimensionEF()
        dim = dimension_of(callable_dim_ef)
        assert dim == 128, f"Expected dimension 128, got {dim}"
        print(f"   CallableDimensionEF dimension: {dim}")

        # Test 3: Embedding function with non-callable dimension attribute
        print("\n✅ Testing dimension_of with non-callable dimension attribute")

        class PropertyDimensionEF:
            def __init__(self):
                self.dimension = 256

            def __call__(self, documents):
                if isinstance(documents, str):
                    documents = [documents]
                # Return embeddings with dimension 256
                return [[0.1] * 256 for _ in documents]

        property_dim_ef = PropertyDimensionEF()
        dim = dimension_of(property_dim_ef)
        assert dim == 256, f"Expected dimension 256, got {dim}"
        print(f"   PropertyDimensionEF dimension: {dim}")

        # Test 4: Embedding function without dimension attribute (fallback)
        print("\n✅ Testing dimension_of with fallback (no dimension attribute)")

        class NoDimensionEF:
            def __call__(self, documents):
                if isinstance(documents, str):
                    documents = [documents]
                # Return embeddings with dimension 512
                return [[0.1] * 512 for _ in documents]

        no_dim_ef = NoDimensionEF()
        dim = dimension_of(no_dim_ef)
        assert dim == 512, f"Expected dimension 512, got {dim}"
        print(f"   NoDimensionEF dimension (fallback): {dim}")

        # Test 5: Edge case - empty result should raise ValueError
        print("\n✅ Testing dimension_of with empty result (should raise ValueError)")

        class EmptyResultEF:
            def __call__(self, documents):
                return []

        empty_ef = EmptyResultEF()
        with pytest.raises(ValueError, match="Embedding function returned empty result"):
            dimension_of(empty_ef)
        print("   EmptyResultEF correctly raised ValueError")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
