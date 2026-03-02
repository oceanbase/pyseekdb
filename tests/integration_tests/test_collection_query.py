"""
Collection query tests using db_client fixture
Demonstrates how to use the conftest.py fixtures to eliminate code duplication
"""

import time
import uuid

import pytest

from pyseekdb import HNSWConfiguration
from pyseekdb.client.query_types import QueryHint


class TestCollectionQuery:
    """Test collection.query() interface using parameterized db_client fixture"""

    def _generate_query_vector(self, dimension: int, base_vector: list[float] | None = None) -> list[float]:
        """Generate a query vector with the correct dimension"""
        if base_vector is None:
            base_vector = [1.0, 2.0, 3.0]

        if dimension <= len(base_vector):
            return base_vector[:dimension]
        else:
            extended = base_vector * ((dimension // len(base_vector)) + 1)
            return extended[:dimension]

    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        """Helper method to insert test data using direct SQL

        Args:
            client: Client instance
            collection_name: Collection name
            dimension: Actual dimension of the collection (used to generate vectors)
        """
        collection = client.get_collection(collection_name)

        # Base vectors (3D) - will be extended or truncated to match actual dimension
        base_vectors = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1],
            [2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2],
        ]

        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "document": "This is a test document about machine learning",
                "base_vector": base_vectors[0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"},
            },
            {
                "document": "Python programming tutorial for beginners",
                "base_vector": base_vectors[1],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"},
            },
            {
                "document": "Advanced machine learning algorithms",
                "base_vector": base_vectors[2],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"},
            },
            {
                "document": "Data science with Python",
                "base_vector": base_vectors[3],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"},
            },
            {
                "document": "Introduction to neural networks",
                "base_vector": base_vectors[4],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"},
            },
        ]

        insert_ids = [str(uuid.uuid4()) for _ in test_data]
        collection.add(
            ids=insert_ids,
            embeddings=[data["base_vector"] for data in test_data],
            documents=[data["document"] for data in test_data],
            metadatas=[data["metadata"] for data in test_data],
        )

        print(f"   Inserted {len(test_data)} test records (dimension={dimension})")
        return insert_ids

    def test_collection_query(self, db_client):
        """
        Test collection.query() with all three client modes.

        This single test function automatically runs 3 times:
        - test_collection_query[embedded]
        - test_collection_query[server]
        - test_collection_query[oceanbase]

        No manual client creation or cleanup needed!
        """
        # Create test collection
        collection_name = f"test_query_refactored_{int(time.time() * 1000)}"
        from pyseekdb import HNSWConfiguration

        config = HNSWConfiguration(dimension=3, distance="l2")
        collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )
        # Get actual dimension (may be different from requested due to default embedding function)
        actual_dimension = collection.dimension

        # Insert test data
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)

        # Test 1: Basic vector similarity query
        print("\n✅ Testing basic query")
        # Generate query vector with correct dimension
        query_vector = [1.0, 2.0, 3.0] * ((actual_dimension // 3) + 1)
        query_vector = query_vector[:actual_dimension]
        results = collection.query(query_embeddings=query_vector, n_results=3)
        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results")

        # Test 2: Query with metadata filter
        print("✅ Testing query with metadata filter")
        results = collection.query(query_embeddings=query_vector, where={"category": "AI"}, n_results=5)
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results with category='AI'")

        # Test 3: Query with document filter
        print("✅ Testing query with document filter")
        results = collection.query(
            query_embeddings=query_vector,
            where_document={"$contains": "machine learning"},
            n_results=5,
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results containing 'machine learning'")

        # Test 4: Query with document filter using regex
        print("✅ Testing query with document filter using regex")
        results = collection.query(
            query_embeddings=query_vector,
            where_document={"$regex": ".*machine.*"},
            n_results=5,
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results matching regex '.*machine.*'")

        # Test 5: Query with include parameter
        print("✅ Testing query with include parameter")
        results = collection.query(
            query_embeddings=query_vector,
            include=["documents", "metadatas"],
            n_results=3,
        )
        assert results is not None
        assert "ids" in results
        if len(results["ids"][0]) > 0:
            # Check that results have the expected fields
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"][0]) == len(results["documents"][0])
            assert len(results["ids"][0]) == len(results["metadatas"][0])

        # Test 6: Query with multiple vectors (should return dict with lists of lists)
        print("✅ Testing query with multiple vectors (returns dict with lists of lists)")
        query_vector2 = [2.0, 3.0, 4.0] * ((actual_dimension // 3) + 1)
        query_vector2 = query_vector2[:actual_dimension]
        results = collection.query(query_embeddings=[query_vector, query_vector2], n_results=2)
        assert results is not None
        assert isinstance(results, dict), "Multiple vectors should return dict"
        assert "ids" in results
        assert len(results["ids"]) == 2, f"Expected 2 ID lists, got {len(results['ids'])}"
        for i in range(len(results["ids"])):
            assert len(results["ids"][i]) > 0, f"ID list {i} should have at least one item"
            print(f"   Query {i}: {len(results['ids'][i])} items")

        # Test 7: Single vector returns dict with single list
        print("✅ Testing single vector returns dict format")
        results = collection.query(query_embeddings=query_vector, n_results=2)
        assert results is not None
        assert isinstance(results, dict), "Single vector should return dict"
        assert "ids" in results
        assert len(results["ids"]) == 1, "Single query should have one ID list"
        assert len(results["ids"][0]) > 0
        print(f"   Single query with {len(results['ids'][0])} items")

        # Test 8: Query with $in operator
        print("✅ Testing query with $in operator")
        results = collection.query(
            query_embeddings=query_vector,
            where={"tag": {"$in": ["ml", "python"]}},
            n_results=5,
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results with tag in ['ml', 'python']")

        # Test 9: Query with comparison operators
        print("✅ Testing query with comparison operators ($gte)")
        results = collection.query(query_embeddings=query_vector, where={"score": {"$gte": 90}}, n_results=5)
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results with score >= 90")

        # No cleanup needed - the fixture handles it automatically!
        print("   ✅ All tests passed (cleanup will be automatic)")

    def test_collection_query_with_query_hint(self, db_client):
        """
        Test collection.query() with QueryHint for database optimization.

        Tests:
        - Query with parallel hint
        - Query with query_timeout hint
        - Query with both hints and filters

        Automatically runs for: embedded, server, oceanbase
        """
        # Create test collection
        collection_name = f"test_query_hint_{int(time.time() * 1000)}"

        collection = db_client.create_collection(
            name=collection_name, configuration=HNSWConfiguration(dimension=3, distance="l2"), embedding_function=None
        )
        dimension = collection.dimension

        try:
            # Insert test data
            inserted_ids = self._insert_test_data(db_client, collection_name, dimension)
            assert len(inserted_ids) > 0

            # Generate query vector
            query_vector = self._generate_query_vector(dimension)

            # Test 1: Query with parallel hint
            print("\n✅ Testing query with parallel hint")
            query_hint = QueryHint(parallel=6)
            results = collection.query(query_embeddings=query_vector, n_results=3, query_hint=query_hint)
            assert results is not None
            assert "ids" in results
            assert len(results["ids"][0]) <= 3
            print(f"   Found {len(results['ids'][0])} results with parallel hint")

            # Test 2: Query with query_timeout hint
            print("✅ Testing query with query_timeout hint")
            query_hint = QueryHint(query_timeout=8.0)
            results = collection.query(query_embeddings=query_vector, n_results=3, query_hint=query_hint)
            assert results is not None
            assert "ids" in results
            assert len(results["ids"][0]) <= 3
            print(f"   Found {len(results['ids'][0])} results with query_timeout hint")

            # Test 3: Query with both hints and metadata filter
            print("✅ Testing query with combined hints and metadata filter")
            query_hint = QueryHint(parallel=4, query_timeout=12.0)
            results = collection.query(
                query_embeddings=query_vector, n_results=5, where={"category": "AI"}, query_hint=query_hint
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results with combined hints and filter")

            # Test 4: Query by text with hints
            print("✅ Testing query by text with hints")
            query_hint = QueryHint(parallel=2, query_timeout=10.0, vector_index=True)
            results = collection.query(query_embeddings=query_vector, n_results=3, query_hint=query_hint)
            assert results is not None
            assert "ids" in results
            assert len(results["ids"][0]) <= 3
            print(f"   Found {len(results['ids'][0])} results with text query and hints")

        finally:
            # Cleanup
            try:
                db_client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
