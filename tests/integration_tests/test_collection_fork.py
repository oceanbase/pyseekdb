"""
Integration tests for collection.fork method.

Tests the fork functionality of the Collection class against real databases, including:
- Successful fork operations
- Error handling for invalid collection names
- Error handling when fork is not enabled
- Verifying forked collection contains the same data as original
"""

import contextlib
import logging
import time
import uuid

import pytest

import pyseekdb

logger = logging.getLogger(__name__)


class TestCollectionFork:
    """Tests for collection.fork() method using real database connections."""

    def _is_fork_enabled(self, client) -> bool:
        """Check if fork is enabled for the given client."""
        try:
            return client._server._fork_enabled()
        except Exception:
            logger.exception("Failed to check if fork is enabled")
            return False

    def test_fork_success(self, db_client):
        """
        Test successful fork operation.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create test collection
        collection_name = f"test_fork_original_{int(time.time() * 1000)}"
        forked_name = f"test_fork_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        original_collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Add some test data to the original collection
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        original_collection.add(
            ids=test_ids,
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            documents=["Document 1", "Document 2", "Document 3"],
            metadatas=[{"tag": "A"}, {"tag": "B"}, {"tag": "C"}],
        )

        # Verify original collection has data
        original_count = original_collection.count()
        assert original_count == 3

        # Fork the collection
        print(f"\n✅ Testing fork from '{collection_name}' to '{forked_name}'")
        forked_collection = original_collection.fork(forked_name)

        # Verify forked collection exists
        assert forked_collection is not None
        assert forked_collection.name == forked_name
        assert forked_collection.dimension == original_collection.dimension

        # Verify forked collection has the same data
        forked_count = forked_collection.count()
        assert forked_count == original_count, (
            f"Forked collection should have {original_count} items, got {forked_count}"
        )

        # Verify data in forked collection
        forked_data = forked_collection.get(ids=test_ids)
        assert len(forked_data["ids"]) == 3
        assert set(forked_data["documents"]) == {"Document 1", "Document 2", "Document 3"}
        assert {metadata["tag"] for metadata in forked_data["metadatas"]} == {"A", "B", "C"}
        print(f"   Successfully forked collection with {forked_count} items")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(forked_name)
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_with_invalid_name(self, db_client):
        """
        Test fork with invalid collection name raises ValueError.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create test collection
        collection_name = f"test_fork_invalid_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Test fork with invalid name (contains dash)
        print("\n✅ Testing fork with invalid collection name")
        with pytest.raises(ValueError, match="Invalid collection name"):
            collection.fork("invalid-name")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_with_empty_name(self, db_client):
        """
        Test fork with empty name raises ValueError.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create test collection
        collection_name = f"test_fork_empty_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Test fork with empty name
        print("\n✅ Testing fork with empty collection name")
        with pytest.raises(ValueError, match="Invalid collection name"):
            collection.fork("")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_preserves_original_collection(self, db_client):
        """
        Test that fork preserves the original collection and its data.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create test collection
        collection_name = f"test_fork_preserve_{int(time.time() * 1000)}"
        forked_name = f"test_fork_preserve_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        original_collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Add test data
        test_id = str(uuid.uuid4())
        original_collection.add(
            ids=test_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="Test document",
            metadatas={"key": "value"},
        )

        # Fork the collection
        print("\n✅ Testing that fork preserves original collection")
        forked_collection = original_collection.fork(forked_name)

        # Verify original collection is unchanged
        assert original_collection.name == collection_name
        assert original_collection.count() == 1
        original_data = original_collection.get(ids=test_id)
        assert original_data["documents"][0] == "Test document"
        assert original_data["metadatas"][0]["key"] == "value"

        # Verify forked collection has the same data
        assert forked_collection.count() == 1
        forked_data = forked_collection.get(ids=test_id)
        assert forked_data["documents"][0] == "Test document"
        assert forked_data["metadatas"][0]["key"] == "value"

        print("   Original collection preserved correctly")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(forked_name)
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_independent_operations(self, db_client):
        """
        Test that operations on forked collection don't affect original and vice versa.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create test collection
        collection_name = f"test_fork_independent_{int(time.time() * 1000)}"
        forked_name = f"test_fork_independent_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        original_collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Add initial data
        original_id = str(uuid.uuid4())
        original_collection.add(
            ids=original_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="Original document",
            metadatas={"source": "original"},
        )

        # Fork the collection
        print("\n✅ Testing independent operations on forked collection")
        forked_collection = original_collection.fork(forked_name)

        # Add data to forked collection
        forked_id = str(uuid.uuid4())
        forked_collection.add(
            ids=forked_id,
            embeddings=[4.0, 5.0, 6.0],
            documents="Forked document",
            metadatas={"source": "forked"},
        )

        # Verify original collection only has original data
        assert original_collection.count() == 1
        original_data = original_collection.get()
        assert len(original_data["ids"]) == 1
        assert original_data["ids"][0] == original_id

        # Verify forked collection has both original and new data
        assert forked_collection.count() == 2
        forked_data = forked_collection.get()
        assert len(forked_data["ids"]) == 2
        assert original_id in forked_data["ids"]
        assert forked_id in forked_data["ids"]

        print("   Collections are independent")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(forked_name)
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_v1_collection_success(self, db_client):
        """
        Test successful fork operation for v1 collections.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create v1 test collection
        collection_name = f"test_fork_v1_original_{int(time.time() * 1000)}"
        forked_name = f"test_fork_v1_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        original_collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None, _collection_version=1
        )

        # Verify it's a v1 collection (no collection_id)
        assert original_collection.id is None
        assert original_collection.name == collection_name

        # Add some test data to the original collection
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        original_collection.add(
            ids=test_ids,
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            documents=["V1 Document 1", "V1 Document 2", "V1 Document 3"],
            metadatas=[{"tag": "V1-A"}, {"tag": "V1-B"}, {"tag": "V1-C"}],
        )

        # Verify original collection has data
        original_count = original_collection.count()
        assert original_count == 3

        # Fork the v1 collection
        print(f"\n✅ Testing fork from v1 collection '{collection_name}' to '{forked_name}'")
        forked_collection = original_collection.fork(forked_name)

        # Verify forked collection exists and is a v2 collection (has collection_id)
        assert forked_collection is not None
        assert forked_collection.name == forked_name
        assert forked_collection.id is not None, "Forked collection should be v2 (have collection_id)"
        assert forked_collection.dimension == original_collection.dimension

        # Verify forked collection has the same data
        forked_count = forked_collection.count()
        assert forked_count == original_count, (
            f"Forked collection should have {original_count} items, got {forked_count}"
        )

        # Verify data in forked collection
        forked_data = forked_collection.get(ids=test_ids)
        assert len(forked_data["ids"]) == 3
        assert set(forked_data["documents"]) == {"V1 Document 1", "V1 Document 2", "V1 Document 3"}
        assert {metadata["tag"] for metadata in forked_data["metadatas"]} == {"V1-A", "V1-B", "V1-C"}

        # Verify original v1 collection is unchanged
        assert original_collection.id is None, "Original v1 collection should remain v1"
        assert original_collection.count() == original_count

        print(f"   Successfully forked v1 collection with {forked_count} items to v2 collection")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(forked_name)
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_v1_collection_preserves_original(self, db_client):
        """
        Test that fork preserves the original v1 collection and its data.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create v1 test collection
        collection_name = f"test_fork_v1_preserve_{int(time.time() * 1000)}"
        forked_name = f"test_fork_v1_preserve_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        original_collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None, _collection_version=1
        )

        # Verify it's a v1 collection
        assert original_collection.id is None

        # Add test data
        test_id = str(uuid.uuid4())
        original_collection.add(
            ids=test_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="V1 Test document",
            metadatas={"key": "v1_value", "version": 1},
        )

        # Fork the v1 collection
        print("\n✅ Testing that fork preserves original v1 collection")
        forked_collection = original_collection.fork(forked_name)

        # Verify original v1 collection is unchanged
        assert original_collection.id is None, "Original should remain v1"
        assert original_collection.name == collection_name
        assert original_collection.count() == 1
        original_data = original_collection.get(ids=test_id)
        assert original_data["documents"][0] == "V1 Test document"
        assert original_data["metadatas"][0]["key"] == "v1_value"
        assert original_data["metadatas"][0]["version"] == 1

        # Verify forked collection is v2 and has the same data
        assert forked_collection.id is not None, "Forked collection should be v2"
        assert forked_collection.count() == 1
        forked_data = forked_collection.get(ids=test_id)
        assert forked_data["documents"][0] == "V1 Test document"
        assert forked_data["metadatas"][0]["key"] == "v1_value"
        assert forked_data["metadatas"][0]["version"] == 1

        print("   Original v1 collection preserved correctly, forked to v2")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(forked_name)
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_v1_collection_independent_operations(self, db_client):
        """
        Test that operations on forked v1 collection don't affect original and vice versa.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create v1 test collection
        collection_name = f"test_fork_v1_independent_{int(time.time() * 1000)}"
        forked_name = f"test_fork_v1_independent_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")
        original_collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None, _collection_version=1
        )

        # Verify it's a v1 collection
        assert original_collection.id is None

        # Add initial data
        original_id = str(uuid.uuid4())
        original_collection.add(
            ids=original_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="V1 Original document",
            metadatas={"source": "v1_original"},
        )

        # Fork the v1 collection
        print("\n✅ Testing independent operations on forked v1 collection")
        forked_collection = original_collection.fork(forked_name)

        # Verify forked collection is v2
        assert forked_collection.id is not None, "Forked collection should be v2"

        # Add data to forked collection
        forked_id = str(uuid.uuid4())
        forked_collection.add(
            ids=forked_id,
            embeddings=[4.0, 5.0, 6.0],
            documents="V2 Forked document",
            metadatas={"source": "v2_forked"},
        )

        # Add data to original v1 collection
        v1_new_id = str(uuid.uuid4())
        original_collection.add(
            ids=v1_new_id,
            embeddings=[7.0, 8.0, 9.0],
            documents="V1 New document",
            metadatas={"source": "v1_new"},
        )

        # Verify original v1 collection only has its own data
        assert original_collection.count() == 2
        original_data = original_collection.get()
        assert len(original_data["ids"]) == 2
        assert original_id in original_data["ids"]
        assert v1_new_id in original_data["ids"]
        assert forked_id not in original_data["ids"]

        # Verify forked v2 collection has original data plus new data
        assert forked_collection.count() == 2
        forked_data = forked_collection.get()
        assert len(forked_data["ids"]) == 2
        assert original_id in forked_data["ids"]
        assert forked_id in forked_data["ids"]
        assert v1_new_id not in forked_data["ids"]

        print("   V1 and V2 collections are independent")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(forked_name)
        with contextlib.suppress(Exception):
            db_client.delete_collection(collection_name)

    def test_fork_v1_and_v2_collections(self, db_client):
        """
        Test that both v1 and v2 collections can be forked successfully.

        Automatically runs for: embedded, server, oceanbase
        Skips if fork is not enabled for the database.
        """
        # Check if fork is enabled
        if not self._is_fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        # Create both v1 and v2 collections
        v1_name = f"test_fork_v1_{int(time.time() * 1000)}"
        v1_forked_name = f"test_fork_v1_forked_{int(time.time() * 1000)}"
        v2_name = f"test_fork_v2_{int(time.time() * 1000)}"
        v2_forked_name = f"test_fork_v2_forked_{int(time.time() * 1000)}"
        dimension = 3

        config = pyseekdb.HNSWConfiguration(dimension=dimension, distance="l2")

        # Create v1 collection
        v1_collection = db_client.create_collection(
            name=v1_name, configuration=config, embedding_function=None, _collection_version=1
        )
        assert v1_collection.id is None

        # Create v2 collection
        v2_collection = db_client.create_collection(name=v2_name, configuration=config, embedding_function=None)
        assert v2_collection.id is not None

        # Add data to both
        v1_id = str(uuid.uuid4())
        v1_collection.add(ids=v1_id, embeddings=[1.0, 2.0, 3.0], documents="V1 doc", metadatas={"type": "v1"})

        v2_id = str(uuid.uuid4())
        v2_collection.add(ids=v2_id, embeddings=[4.0, 5.0, 6.0], documents="V2 doc", metadatas={"type": "v2"})

        # Fork both collections
        print("\n✅ Testing fork for both v1 and v2 collections")
        v1_forked = v1_collection.fork(v1_forked_name)
        v2_forked = v2_collection.fork(v2_forked_name)

        # Verify v1 fork is v2
        assert v1_forked.id is not None, "Forked v1 collection should be v2"
        assert v1_forked.count() == 1
        v1_forked_data = v1_forked.get(ids=v1_id)
        assert v1_forked_data["documents"][0] == "V1 doc"
        assert v1_forked_data["metadatas"][0]["type"] == "v1"

        # Verify v2 fork is still v2
        assert v2_forked.id is not None, "Forked v2 collection should be v2"
        assert v2_forked.count() == 1
        v2_forked_data = v2_forked.get(ids=v2_id)
        assert v2_forked_data["documents"][0] == "V2 doc"
        assert v2_forked_data["metadatas"][0]["type"] == "v2"

        # Verify originals are unchanged
        assert v1_collection.id is None, "Original v1 should remain v1"
        assert v2_collection.id is not None, "Original v2 should remain v2"

        print("   Both v1 and v2 collections forked successfully")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(v1_forked_name)
            db_client.delete_collection(v2_forked_name)
            db_client.delete_collection(v1_name)
            db_client.delete_collection(v2_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
