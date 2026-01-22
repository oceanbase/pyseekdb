"""
Client creation and connection tests using db_client fixture
Testing client creation, connection, and collection management for all three modes
"""

import contextlib
import random
import time
import uuid

import pytest

from pyseekdb import Configuration, FulltextAnalyzerConfig, HNSWConfiguration


class TestClientCreation:
    """Test client creation and collection management using parameterized db_client fixture"""

    def test_client_creation_and_collection_management(self, db_client):
        """
        Test client creation, connection, and all collection management interfaces.

        Tests include:
        - create_collection, get_collection, has_collection
        - get_or_create_collection, list_collections, delete_collection
        - count_collection, collection.count(), collection.peek()

        Automatically runs for: embedded, server, oceanbase
        """
        # Verify client is properly initialized
        assert db_client is not None
        assert hasattr(db_client, "_server")
        # Note: Client uses lazy loading, connection happens on first use

        # Test 1: create_collection - create a new collection
        test_collection_name = f"test_collection_{int(time.time() * 1000)}"
        test_dimension = 128

        # Create collection with HNSW configuration
        config = HNSWConfiguration(dimension=test_dimension, distance="cosine")
        collection = db_client.create_collection(
            name=test_collection_name, configuration=config, embedding_function=None
        )

        # Test: Verify Configuration class with fulltext parser works
        test_collection_name_config = f"test_collection_config_{int(time.time() * 1000)}"
        config_with_fulltext = Configuration(
            hnsw=HNSWConfiguration(dimension=test_dimension, distance="cosine"),
            fulltext_config=FulltextAnalyzerConfig(analyzer="ik"),
        )
        collection_config = db_client.create_collection(
            name=test_collection_name_config,
            configuration=config_with_fulltext,
            embedding_function=None,
        )
        assert collection_config is not None
        assert collection_config.name == test_collection_name_config
        # Clean up
        with contextlib.suppress(Exception):
            db_client.delete_collection(test_collection_name_config)

        # Verify collection object
        assert collection is not None
        assert collection.name == test_collection_name
        actual_dimension = collection.dimension
        assert actual_dimension > 0, f"Collection dimension should be positive, got {actual_dimension}"

        # Test 2: get_collection - get the collection we just created
        retrieved_collection = db_client.get_collection(name=test_collection_name)
        assert retrieved_collection is not None
        assert retrieved_collection.name == test_collection_name
        assert retrieved_collection.dimension == actual_dimension
        print(f"\n✅ Collection '{test_collection_name}' retrieved successfully")
        print(f"   Collection name: {retrieved_collection.name}")
        print(f"   Collection dimension: {retrieved_collection.dimension}")

        # Test 3: has_collection - should return False for non-existent collection
        non_existent_name = f"test_collection_nonexistent_{int(time.time() * 1000)}"
        assert not db_client.has_collection(non_existent_name)
        print("\n✅ has_collection correctly returns False for non-existent collection")

        # Test 4: has_collection - should return True for existing collection
        assert db_client.has_collection(test_collection_name)
        print("\n✅ has_collection correctly returns True for existing collection")

        # Test 5: get_or_create_collection - should get existing collection
        existing_collection = db_client.get_or_create_collection(
            name=test_collection_name, configuration=config, embedding_function=None
        )
        assert existing_collection is not None
        assert existing_collection.name == test_collection_name
        assert existing_collection.dimension == actual_dimension
        print("\n✅ get_or_create_collection successfully retrieved existing collection")

        # Test 6: get_or_create_collection - should create new collection
        test_collection_name_mgmt = f"test_collection_mgmt_{int(time.time() * 1000)}"
        new_collection = db_client.get_or_create_collection(
            name=test_collection_name_mgmt,
            configuration=config,
            embedding_function=None,
        )
        assert new_collection is not None
        assert new_collection.name == test_collection_name_mgmt
        assert new_collection.dimension == actual_dimension
        print(f"\n✅ get_or_create_collection successfully created collection '{test_collection_name_mgmt}'")

        # Test 7: list_collections - should include our collections
        collections = db_client.list_collections()
        assert isinstance(collections, list)
        collection_names = [c.name for c in collections]
        assert test_collection_name in collection_names
        assert test_collection_name_mgmt in collection_names
        print(f"\n✅ list_collections successfully listed collections: {len(collections)} found")
        print(f"   Collection names: {collection_names}")

        # Test 8: delete_collection - should delete the collection
        db_client.delete_collection(test_collection_name_mgmt)
        assert not db_client.has_collection(test_collection_name_mgmt)
        print(f"\n✅ delete_collection successfully deleted collection '{test_collection_name_mgmt}'")

        # Test 9: delete_collection - should raise error for non-existent collection
        try:
            db_client.delete_collection(test_collection_name_mgmt)
            pytest.fail("delete_collection should raise ValueError for non-existent collection")
        except ValueError as e:
            assert "does not exist" in str(e)
            print("\n✅ delete_collection correctly raises ValueError for non-existent collection")

        # Test 10: get_or_create_collection without configuration - should use default configuration
        test_collection_name_default = f"test_collection_default_{int(time.time() * 1000)}"
        default_collection = db_client.get_or_create_collection(name=test_collection_name_default)
        assert default_collection is not None
        assert default_collection.name == test_collection_name_default
        # Default dimension is 384 (matches default embedding function)
        assert default_collection.dimension == 384
        print("\n✅ get_or_create_collection successfully created collection with default configuration")

        # Test 11: count_collection - count the number of collections
        collection_count = db_client.count_collection()
        assert isinstance(collection_count, int)
        assert collection_count >= 1  # At least the test collection we created
        print(f"\n✅ count_collection successfully returned count: {collection_count}")

        # Test 12: collection.count() - count items in collection (should be 0 for empty collection)
        item_count = collection.count()
        assert isinstance(item_count, int)
        assert item_count == 0  # Collection is empty
        print(f"\n✅ collection.count() successfully returned count: {item_count}")

        # Test 13: collection.peek() - preview items in empty collection
        preview = collection.peek(limit=5)
        assert preview is not None
        assert "ids" in preview
        assert len(preview["ids"]) == 0  # Empty collection
        print(f"\n✅ collection.peek() successfully returned preview: {len(preview['ids'])} items")

        # Add some test data to test count and peek with data
        random.seed(42)  # For reproducibility
        test_ids = [str(uuid.uuid4()) for _ in range(3)]
        # Generate embeddings matching the collection's dimension
        embeddings = [[random.random() for _ in range(collection.dimension)] for _ in range(3)]  # noqa: S311
        collection.add(
            ids=test_ids,
            embeddings=embeddings,
            documents=[f"Test document {i}" for i in range(3)],
            metadatas=[{"index": i} for i in range(3)],
        )

        # Test 14: collection.count() - count items after adding data
        item_count_after = collection.count()
        assert item_count_after == 3
        print(f"\n✅ collection.count() after adding data: {item_count_after} items")

        # Test 15: collection.peek() - preview items with data
        preview_with_data = collection.peek(limit=2)
        assert preview_with_data is not None
        assert "ids" in preview_with_data
        assert "documents" in preview_with_data
        assert "metadatas" in preview_with_data
        assert "embeddings" in preview_with_data
        assert len(preview_with_data["ids"]) == 2  # Limited to 2 items
        # Verify preview items have expected fields
        assert len(preview_with_data["ids"]) == len(preview_with_data["documents"])
        assert len(preview_with_data["ids"]) == len(preview_with_data["metadatas"])
        assert len(preview_with_data["ids"]) == len(preview_with_data["embeddings"])
        print(f"\n✅ collection.peek() with data returned {len(preview_with_data['ids'])} items")

        # Test 16: collection.peek() with different limit
        preview_all = collection.peek(limit=10)
        assert len(preview_all["ids"]) == 3  # All 3 items
        print(f"\n✅ collection.peek(limit=10) returned {len(preview_all['ids'])} items")

        # Clean up: delete all test collections
        try:
            db_client.delete_collection(test_collection_name)
            print(f"   Cleaned up collection: {test_collection_name}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup {test_collection_name}: {cleanup_error}")

        try:
            db_client.delete_collection(test_collection_name_default)
            print(f"   Cleaned up collection: {test_collection_name_default}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup {test_collection_name_default}: {cleanup_error}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
