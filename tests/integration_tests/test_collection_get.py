"""
Collection get tests - testing collection.get() interface for all three modes using db_client fixture
"""

import contextlib
import time
import uuid

import pytest

import pyseekdb


# ==================== Simple 3D Embedding Function for Testing ====================
class Simple3DEmbeddingFunction:
    """Simple embedding function that returns 3-dimensional vectors for testing"""

    def __init__(self):
        self.dimension = 3

    def __call__(self, documents: str | list[str]) -> list[list[float]]:
        """Convert documents to 3D embeddings (simple hash-based)"""
        if isinstance(documents, str):
            documents = [documents]

        embeddings = []
        for doc in documents:
            # Simple hash-based 3D embedding for testing
            hash_val = hash(doc) % 1000
            embedding = [
                float((hash_val % 10) / 10.0),
                float(((hash_val // 10) % 10) / 10.0),
                float(((hash_val // 100) % 10) / 10.0),
            ]
            embeddings.append(embedding)

        return embeddings


class TestCollectionGet:
    """Test collection.get() interface for all three modes"""

    def _insert_test_data(self, client, collection_name: str):
        """Helper method to insert test data and return inserted IDs"""

        collection = client.get_collection(collection_name)

        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "_id": str(uuid.uuid4()),
                "document": "This is a test document about machine learning",
                "embedding": [1.0, 2.0, 3.0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"},
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Python programming tutorial for beginners",
                "embedding": [2.0, 3.0, 4.0],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"},
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Advanced machine learning algorithms",
                "embedding": [1.1, 2.1, 3.1],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"},
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Data science with Python",
                "embedding": [2.1, 3.1, 4.1],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"},
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Introduction to neural networks",
                "embedding": [1.2, 2.2, 3.2],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"},
            },
        ]

        # Store inserted IDs for return (using generated UUIDs)
        collection.add(
            ids=[data["_id"] for data in test_data],
            embeddings=[data["embedding"] for data in test_data],
            documents=[data["document"] for data in test_data],
            metadatas=[data["metadata"] for data in test_data],
        )
        return [data["_id"] for data in test_data]

    def test_metadata_array_in_nin_overlap(self, db_client):
        """
        Regression test for JSON array $in/$nin operators.

        Tests:
        - $in with overlap, disjoint, empty list
        - $nin with complementary expectations
        - Handles missing fields, null values, empty arrays

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_tags_in_{int(time.time() * 1000)}"
        collection = db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction(),
        )

        try:
            # Insert cases covering overlap, disjoint, missing, null, empty array
            collection.add(
                ids=["id_overlap", "id_disjoint", "id_missing", "id_null", "id_empty"],
                documents=["", "", "", "", ""],
                metadatas=[
                    {
                        "category": "AI",
                        "tags": ["ml", "ai"],
                    },  # overlaps with ["ml", "python"]
                    {"category": "Web", "tags": ["java", "cpp"]},  # disjoint
                    {"category": "Missing"},  # tags missing
                    {"category": "Null", "tags": None},  # tags explicit null
                    {"category": "Empty", "tags": []},  # tags empty array
                ],
            )

            # Test $in with overlap
            result = collection.get(
                where={"tags": {"$in": ["ml", "python"]}},
                include=["metadatas", "ids"],
            )
            assert result and "ids" in result
            assert set(result["ids"]) == {"id_overlap"}

            # Test $in with disjoint values -> no hits
            result = collection.get(
                where={"tags": {"$in": ["ruby"]}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert len(result["ids"]) == 0

            # Test $in with empty list -> should return 0 rows
            result = collection.get(
                where={"tags": {"$in": []}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert len(result["ids"]) == 0

            # Test $nin should keep disjoint/null/empty, exclude overlap
            result = collection.get(
                where={"tags": {"$nin": ["ml", "python"]}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert set(result["ids"]) == {"id_disjoint", "id_null", "id_empty"}

            # Additional $nin test with two records
            collection_name_2 = f"test_tags_nin_{int(time.time() * 1000)}"
            collection_2 = db_client.get_or_create_collection(
                name=collection_name_2,
                embedding_function=pyseekdb.DefaultEmbeddingFunction(),
            )

            try:
                # Insert two records: one with overlap, one without
                collection_2.add(
                    ids=["id1", "id2"],
                    documents=["", ""],
                    metadatas=[
                        {
                            "category": "AI",
                            "tags": ["ml", "ai"],
                        },  # has overlap with ["ml", "python"]
                        {
                            "category": "Web",
                            "tags": ["java", "cpp"],
                        },  # no overlap with ["ml", "python"]
                    ],
                )

                result = collection_2.get(
                    where={"tags": {"$nin": ["ml", "python"]}},
                    include=["metadatas", "ids"],
                )

                # Expect 1 row (id2) because its tags have no overlap with the exclusion list
                assert result is not None
                assert "ids" in result
                assert len(result["ids"]) == 1
                assert "id2" in result["ids"]
                assert "id1" not in result["ids"]
                assert result["metadatas"][0].get("tags") == ["java", "cpp"]
            finally:
                with contextlib.suppress(Exception):
                    db_client.delete_collection(name=collection_name_2)

        finally:
            try:
                db_client.delete_collection(name=collection_name)
            except Exception as cleanup_error:
                print(f"Warning: cleanup failed for {collection_name}: {cleanup_error}")

    def test_eq_ne_operators_with_array_fields(self, db_client):
        """
        Regression test for $eq and $ne operators with array fields.

        Tests:
        - $eq with array membership check (like $in)
        - $ne with array exclusion (like $nin)
        - Direct equality (no operator) consistency with $eq
        - Scalar fields still work correctly
        - Edge cases: empty array, null, missing field
        - Consistency: $eq behavior matches $in with single value

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_eq_ne_{int(time.time() * 1000)}"
        collection = db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction(),
        )

        try:
            # Insert comprehensive test data
            collection.add(
                ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
                documents=[
                    "doc1",
                    "doc2",
                    "doc3",
                    "doc4",
                    "doc5",
                    "doc6",
                    "doc7",
                    "doc8",
                ],
                metadatas=[
                    {"tags": ["ml", "ai"], "category": "AI"},  # id1: array with ml
                    {"tags": ["java"], "category": "Backend"},  # id2: array without ml
                    {"tags": "ml", "category": "ML"},  # id3: scalar ml
                    {
                        "tags": "python",
                        "category": "Scripting",
                    },  # id4: different scalar
                    {"category": "Other"},  # id5: missing tags field
                    {"tags": [], "category": "Empty"},  # id6: empty array
                    {"tags": None, "category": "Null"},  # id7: null value
                    {"tags": ["ml"], "category": "Single"},  # id8: single-element array
                ],
            )

            # Test 1: $eq with array field (membership check)
            print("\n✅ Test 1: $eq with array field")
            result = collection.get(
                where={"tags": {"$eq": "ml"}},
                include=["ids", "metadatas"],
            )
            assert result and "ids" in result
            assert set(result["ids"]) == {"id1", "id3", "id8"}, (
                f"Expected id1, id3, id8 (array+scalar+single), got {result['ids']}"
            )
            print(f"   Matched: {sorted(result['ids'])} (array with ml, scalar ml, single-element array)")

            # Test 2: $eq with non-matching value
            print("\n✅ Test 2: $eq with non-matching value")
            result = collection.get(
                where={"tags": {"$eq": "ruby"}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert len(result["ids"]) == 0, f"Expected no matches, got {result['ids']}"
            print("   Correctly returned 0 results")

            # Test 3: $ne with array field (exclusion)
            print("\n✅ Test 3: $ne with array field")
            result = collection.get(
                where={"tags": {"$ne": "ml"}},
                include=["ids"],
            )
            assert result and "ids" in result
            # Should match id2 (java), id4 (python) - excludes id1, id3, id8 which contain/equal ml
            # id5 (missing), id6 (empty), id7 (null) behavior depends on JSON_OVERLAPS NULL handling
            matched_ids = set(result["ids"])
            assert "id2" in matched_ids, "Expected id2 in results"
            assert "id4" in matched_ids, "Expected id4 in results"
            assert "id1" not in matched_ids, "id1 should be excluded (has ml)"
            assert "id3" not in matched_ids, "id3 should be excluded (is ml)"
            assert "id8" not in matched_ids, "id8 should be excluded (has ml)"
            print(f"   Correctly excluded ids with ml: {sorted(result['ids'])}")

            # Test 4: Direct equality (no operator) - should behave like $eq
            print("\n✅ Test 4: Direct equality consistency")
            result_direct = collection.get(
                where={"tags": "ml"},
                include=["ids"],
            )
            result_eq = collection.get(
                where={"tags": {"$eq": "ml"}},
                include=["ids"],
            )
            assert set(result_direct["ids"]) == set(result_eq["ids"]), (
                f"Direct equality should match $eq: {result_direct['ids']} vs {result_eq['ids']}"
            )
            print(f"   Direct equality matches $eq: {sorted(result_direct['ids'])}")

            # Test 5: $eq consistency with $in (single value)
            print("\n✅ Test 5: $eq consistency with $in")
            result_eq = collection.get(
                where={"tags": {"$eq": "ml"}},
                include=["ids"],
            )
            result_in = collection.get(
                where={"tags": {"$in": ["ml"]}},
                include=["ids"],
            )
            assert set(result_eq["ids"]) == set(result_in["ids"]), (
                f"$eq should match $in with single value: {result_eq['ids']} vs {result_in['ids']}"
            )
            print(f"   $eq matches $in: {sorted(result_eq['ids'])}")

            # Test 6: Scalar field still works correctly
            print("\n✅ Test 6: Scalar field not affected")
            result_scalar_eq = collection.get(
                where={"category": {"$eq": "AI"}},
                include=["ids"],
            )
            assert result_scalar_eq and "ids" in result_scalar_eq
            assert set(result_scalar_eq["ids"]) == {"id1"}, (
                f"Scalar $eq should still work, got {result_scalar_eq['ids']}"
            )

            result_scalar_direct = collection.get(
                where={"category": "AI"},
                include=["ids"],
            )
            assert set(result_scalar_direct["ids"]) == {"id1"}, (
                f"Scalar direct equality should still work, got {result_scalar_direct['ids']}"
            )
            print("   Scalar fields work correctly")

            # Test 7: Edge case - $eq with java (single-element array vs scalar)
            print("\n✅ Test 7: $eq with single-element array")
            result = collection.get(
                where={"tags": {"$eq": "java"}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert set(result["ids"]) == {"id2"}, f"Expected id2, got {result['ids']}"
            print(f"   Matched single-element array: {result['ids']}")

            # Test 8: Multiple conditions with $eq and $ne
            print("\n✅ Test 8: Combined $eq and $ne")
            result = collection.get(
                where={"$and": [{"tags": {"$ne": "ml"}}, {"category": {"$ne": "Empty"}}]},
                include=["ids"],
            )
            assert result and "ids" in result
            # Should exclude: id1/id3/id8 (have ml), id6 (category=Empty)
            matched_ids = set(result["ids"])
            assert "id2" in matched_ids or "id4" in matched_ids, f"Should match id2 or id4, got {result['ids']}"
            for excluded_id in ["id1", "id3", "id8", "id6"]:
                assert excluded_id not in matched_ids, f"{excluded_id} should be excluded, got {result['ids']}"
            print(f"   Combined filters work: {sorted(result['ids'])}")

            print("\n✅ All $eq/$ne operator tests passed!")

        finally:
            try:
                db_client.delete_collection(name=collection_name)
            except Exception as cleanup_error:
                print(f"Warning: cleanup failed for {collection_name}: {cleanup_error}")

    def test_collection_get(self, db_client):
        """
        Test collection.get() interface with various query patterns.

        Tests:
        - Get by single/multiple IDs
        - Get with metadata/document filters
        - Get with logical operators ($or)
        - Get with limit/offset
        - Get with include parameter
        - Get with scalar $in/$nin operators

        Automatically runs for: embedded, server, oceanbase
        """
        # Create test collection
        collection_name = f"test_get_{int(time.time() * 1000)}"
        config = pyseekdb.HNSWConfiguration(dimension=3, distance="l2")
        # Use a simple 3D embedding function to match the dimension
        embedding_function = Simple3DEmbeddingFunction()
        collection = db_client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=embedding_function,
        )

        try:
            inserted_ids = self._insert_test_data(db_client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")

            # Test 1: Get by single ID
            print("\n✅ Testing get by single ID")
            results = collection.get(ids=inserted_ids[0])
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) == 1
            print(f"   Found {len(results['ids'])} result for ID={inserted_ids[0]}")

            # Test 2: Get by multiple IDs
            print("✅ Testing get by multiple IDs")
            if len(inserted_ids) >= 3:
                results = collection.get(ids=inserted_ids[:3])
                assert results is not None
                assert "ids" in results
                assert len(results["ids"]) <= 3
                print(f"   Found {len(results['ids'])} results for IDs={inserted_ids[:3]}")

            # Test 3: Get by metadata filter
            print("✅ Testing get with metadata filter (category=AI)")
            results = collection.get(where={"category": {"$eq": "AI"}}, limit=10)
            assert results is not None
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results with category='AI'")

            # Test 4: Get with logical operators ($or)
            print("✅ Testing get with logical operators ($or)")
            results = collection.get(
                where={
                    "$or": [
                        {"category": "AI"},
                        {"tag": "python"},
                    ]
                },
                limit=10,
            )
            assert results is not None
            print(f"   Found {len(results['ids'])} results with $or condition")

            # Test 5: Get by document filter
            print("✅ Testing get with document filter")
            results = collection.get(where_document={"$contains": "machine learning"}, limit=10)
            assert results is not None
            print(f"   Found {len(results['ids'])} results containing 'machine learning'")

            # Test 6: Get with combined filters
            print("✅ Testing get with combined filters")
            results = collection.get(
                where={"category": {"$eq": "AI"}},
                where_document={"$contains": "machine"},
                limit=10,
            )
            assert results is not None
            print(f"   Found {len(results['ids'])} results matching all filters")

            # Test 7: Get with limit and offset
            print("✅ Testing get with limit and offset")
            results = collection.get(limit=3, offset=0)
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) <= 3
            print(f"   Found {len(results['ids'])} results (limit=3, offset=0)")

            # Test 8: Get all data without filters
            print("✅ Testing get all data without filters")
            results = collection.get(limit=100)
            assert results is not None
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} total results")

            # Test 9: Get with include parameter
            print("✅ Testing get with include parameter")
            results = collection.get(
                ids=inserted_ids[:2],
                include=["documents", "metadatas"],
            )
            assert results is not None
            assert isinstance(results, dict), "Should return dict"
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) == 2
            print(f"   Found {len(results['ids'])} results with documents and metadatas")

            # Test 10: Get by multiple IDs (should return dict)
            print("✅ Testing get by multiple IDs (returns dict)")
            if len(inserted_ids) >= 3:
                results = collection.get(ids=inserted_ids[:3])
                assert results is not None
                assert isinstance(results, dict), "Should return dict"
                assert "ids" in results
                assert len(results["ids"]) <= 3
                print(f"   Found {len(results['ids'])} results for {len(inserted_ids[:3])} IDs")

            # Test 11: Single ID returns dict format
            print("✅ Testing single ID returns dict format")
            results = collection.get(ids=inserted_ids[0])
            assert results is not None
            assert isinstance(results, dict), "Should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 1
            print(f"   Single result with {len(results['ids'])} item")

            # Test 12: Get with filters returns dict format
            print("✅ Testing get with filters returns dict format")
            results = collection.get(where={"category": {"$eq": "AI"}}, limit=10)
            assert results is not None
            assert isinstance(results, dict), "Should return dict"
            assert "ids" in results
            print(f"   Found {len(results['ids'])} items matching filter")

            # Test 13: Get with scalar $in operator
            print("✅ Testing get with scalar $in operator")
            results = collection.get(where={"tag": {"$in": ["ml", "python"]}}, limit=10)
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results with tag in ['ml', 'python']")

            # Test 14: Get with scalar $nin operator
            print("✅ Testing get with scalar $nin operator")
            results = collection.get(where={"tag": {"$nin": ["ml", "python"]}}, limit=10)
            assert results is not None
            assert "ids" in results
            # Should return rows with tag='neural' (excluded 'ml' and 'python')
            print(f"   Found {len(results['ids'])} results with tag not in ['ml', 'python']")

        finally:
            # Cleanup
            try:
                db_client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
