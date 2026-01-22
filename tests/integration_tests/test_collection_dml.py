"""
Collection DML tests using db_client fixture
Testing collection.add(), collection.delete(), collection.upsert(), collection.update() interfaces
"""

import time
import uuid

import pytest

import pyseekdb


class TestCollectionDML:
    """Test collection DML operations using parameterized db_client fixture"""

    def test_collection_dml(self, db_client):
        """
        Test collection DML operations (add, update, upsert, delete).

        Automatically runs for: embedded, server, oceanbase
        """
        # Create test collection using execute
        collection_name = f"test_dml_{int(time.time() * 1000)}"
        dimension = 3

        # Get collection object
        collection = db_client.get_or_create_collection(
            name=collection_name,
            configuration=pyseekdb.HNSWConfiguration(dimension=dimension),
            embedding_function=None,
        )

        # Test 1: collection.add - Add single item
        print("\n✅ Testing collection.add() - single item")
        test_id_1 = str(uuid.uuid4())
        collection.add(
            ids=test_id_1,
            embeddings=[1.0, 2.0, 3.0],
            documents="This is test document 1",
            metadatas={"category": "test", "score": 100},
        )

        # Verify using collection.get
        results = collection.get(ids=test_id_1)
        assert len(results["ids"]) == 1
        assert results["ids"][0] == test_id_1
        assert results["documents"][0] == "This is test document 1"
        assert results["metadatas"][0].get("category") == "test"
        print(f"   Successfully added and verified item with ID: {test_id_1}")

        # Test 2: collection.add - Add multiple items
        print("✅ Testing collection.add() - multiple items")
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        collection.add(
            ids=test_ids,
            embeddings=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            documents=["Document 2", "Document 3", "Document 4"],
            metadatas=[
                {"category": "test", "score": 90},
                {"category": "test", "score": 85},
                {"category": "demo", "score": 80},
            ],
        )

        # Verify using collection.get
        results = collection.get(ids=test_ids)
        assert len(results["ids"]) == 3
        print(f"   Successfully added and verified {len(results['ids'])} items")

        # Test 3: collection.update - Update existing item
        print("✅ Testing collection.update() - update existing item")
        collection.update(ids=test_id_1, metadatas={"category": "test", "score": 95, "updated": True})

        # Verify update using collection.get
        results = collection.get(ids=test_id_1)
        assert len(results["ids"]) == 1
        # Document should remain unchanged since we didn't update it
        assert results["documents"][0] == "This is test document 1"
        assert results["metadatas"][0].get("score") == 95
        assert results["metadatas"][0].get("updated") is True
        print(f"   Successfully updated and verified item with ID: {test_id_1}")

        # Test 4: collection.update - Update multiple items
        print("✅ Testing collection.update() - update multiple items")
        collection.update(
            ids=test_ids[:2],
            embeddings=[[2.1, 3.1, 4.1], [3.1, 4.1, 5.1]],
            metadatas=[
                {"category": "test", "score": 92},
                {"category": "test", "score": 87},
            ],
        )

        # Verify update using collection.get
        results = collection.get(ids=test_ids[:2])
        assert len(results["ids"]) == 2
        print(f"   Successfully updated and verified {len(results['ids'])} items")

        # Test 5: collection.upsert - Upsert existing item (should update)
        print("✅ Testing collection.upsert() - upsert existing item (update)")
        collection.upsert(
            ids=test_id_1,
            embeddings=[1.0, 2.0, 3.0],  # Use original vector
            documents="Upserted document 1",
            metadatas={"category": "test", "score": 98},
        )

        # Verify upsert using collection.get
        results = collection.get(ids=test_id_1)
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "Upserted document 1"
        assert results["metadatas"][0].get("score") == 98
        print(f"   Successfully upserted (update) and verified item with ID: {test_id_1}")

        # Test 6: collection.upsert - Upsert new item (should insert)
        print("✅ Testing collection.upsert() - upsert new item (insert)")
        test_id_new = str(uuid.uuid4())
        collection.upsert(
            ids=test_id_new,
            embeddings=[5.0, 6.0, 7.0],
            documents="New upserted document",
            metadatas={"category": "new", "score": 99},
        )

        # Verify upsert using collection.get
        results = collection.get(ids=test_id_new)
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "New upserted document"
        assert results["metadatas"][0].get("category") == "new"
        print(f"   Successfully upserted (insert) and verified item with ID: {test_id_new}")

        # Test 7: collection.delete - Delete by ID
        print("✅ Testing collection.delete() - delete by ID")
        # Delete one of the test items
        collection.delete(ids=test_ids[0])

        # Verify deletion using collection.get
        results = collection.get(ids=test_ids[0])
        assert len(results["ids"]) == 0
        print(f"   Successfully deleted item with ID: {test_ids[0]}")

        # Verify other items still exist
        results = collection.get(ids=test_ids[1:])
        assert len(results["ids"]) == 2
        print("   Verified other items still exist")

        # Test 8: collection.delete - Delete by metadata filter
        print("✅ Testing collection.delete() - delete by metadata filter")
        # Delete items with category="demo"
        collection.delete(where={"category": {"$eq": "demo"}})

        # Verify deletion using collection.get
        results = collection.get(where={"category": {"$eq": "demo"}})
        assert len(results["ids"]) == 0
        print("   Successfully deleted items with category='demo'")

        # Test 9: collection.delete - Delete by document filter
        print("✅ Testing collection.delete() - delete by document filter")
        # Add an item with specific document content
        test_id_doc = str(uuid.uuid4())
        collection.add(
            ids=test_id_doc,
            embeddings=[6.0, 7.0, 8.0],
            documents="Delete this document",
            metadatas={"category": "temp"},
        )

        # Delete by document filter
        collection.delete(where_document={"$contains": "Delete this"})

        # Verify deletion using collection.get
        results = collection.get(where_document={"$contains": "Delete this"})
        assert len(results["ids"]) == 0
        print("   Successfully deleted items by document filter")

        # Test 10: Verify final state using collection.get
        print("✅ Testing final state verification")
        all_results = collection.get(limit=100)
        print(f"   Final collection count: {len(all_results['ids'])} items")
        assert len(all_results["ids"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
