"""
Integration tests for v1 collection compatibility.

Tests that all collection operations work correctly with v1 collections
created using _collection_version=1 parameter.
"""

import time
import uuid

import pytest

import pyseekdb
from pyseekdb.client.meta_info import CollectionNames


class TestCollectionV1Compatibility:
    """Test that collection operations are compatible with v1 collections"""

    def test_v1_collection_creation_and_retrieval(self, db_client):
        """Test creating and retrieving a v1 collection"""
        collection_name = f"test_v1_{int(time.time() * 1000)}"

        # Create v1 collection
        collection = db_client.create_collection(
            name=collection_name,
            _collection_version=1,
        )

        # Verify it's a v1 collection (no collection_id)
        assert collection.id is None
        assert collection.name == collection_name
        assert collection.dimension is not None

        # Verify it can be retrieved
        retrieved_collection = db_client.get_collection(collection_name)
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.id is None  # v1 collections don't have ID

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_has_collection(self, db_client):
        """Test has_collection works with v1 collections"""
        collection_name = f"test_v1_has_{int(time.time() * 1000)}"

        # Collection doesn't exist yet
        assert not db_client.has_collection(collection_name)

        # Create v1 collection
        db_client.create_collection(name=collection_name, _collection_version=1)

        # Should exist now
        assert db_client.has_collection(collection_name)

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_list_collections(self, db_client):
        """Test list_collections includes v1 collections"""
        collection_name = f"test_v1_list_{int(time.time() * 1000)}"

        # Get initial count
        initial_count = len(db_client.list_collections())

        # Create v1 collection
        db_client.create_collection(name=collection_name, _collection_version=1)

        # Should be in list
        collections = db_client.list_collections()
        assert len(collections) == initial_count + 1

        # Find our collection
        found = False
        for coll in collections:
            if coll.name == collection_name:
                found = True
                assert coll.id is None  # v1 collections don't have ID
                break

        assert found, f"Collection {collection_name} not found in list"

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_add_and_get(self, db_client):
        """Test add and get operations with v1 collections"""
        collection_name = f"test_v1_add_{int(time.time() * 1000)}"
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Add single item
        test_id = str(uuid.uuid4())
        collection.add(
            ids=test_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="Test document",
            metadatas={"category": "test", "score": 100},
        )

        # Get the item
        results = collection.get(ids=test_id)
        assert len(results["ids"]) == 1
        assert results["ids"][0] == test_id
        assert results["documents"][0] == "Test document"
        assert results["metadatas"][0]["category"] == "test"
        assert results["metadatas"][0]["score"] == 100

        # Add multiple items
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        collection.add(
            ids=test_ids,
            embeddings=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{"tag": "a"}, {"tag": "b"}],
        )

        # Get multiple items
        results = collection.get(ids=test_ids)
        assert len(results["ids"]) == 2
        assert set(results["ids"]) == set(test_ids)

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_update(self, db_client):
        """Test update operation with v1 collections"""
        collection_name = f"test_v1_update_{int(time.time() * 1000)}"
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Add item
        test_id = str(uuid.uuid4())
        collection.add(
            ids=test_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="Original document",
            metadatas={"category": "original", "score": 50},
        )

        # Update metadata
        collection.update(
            ids=test_id,
            metadatas={"category": "updated", "score": 75, "new_field": True},
        )

        # Verify update
        results = collection.get(ids=test_id)
        assert results["documents"][0] == "Original document"  # Document unchanged
        assert results["metadatas"][0]["category"] == "updated"
        assert results["metadatas"][0]["score"] == 75
        assert results["metadatas"][0]["new_field"] is True

        # Update document
        collection.update(
            ids=test_id,
            documents="Updated document",
            embeddings=[1.0, 2.0, 3.0],
        )

        # Verify document update
        results = collection.get(ids=test_id)
        assert results["documents"][0] == "Updated document"

        # Update embedding
        collection.update(
            ids=test_id,
            embeddings=[10.0, 20.0, 30.0],
        )

        # Verify embedding update
        results = collection.get(ids=test_id, include=["embeddings"])
        assert len(results["embeddings"][0]) == 3
        assert results["embeddings"][0] == [10.0, 20.0, 30.0]

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_upsert(self, db_client):
        """Test upsert operation with v1 collections"""
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection_name = f"test_v1_upsert_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Upsert new item
        test_id = str(uuid.uuid4())
        collection.upsert(
            ids=test_id,
            embeddings=[1.0, 2.0, 3.0],
            documents="New document",
            metadatas={"status": "new"},
        )

        # Verify it was inserted
        results = collection.get(ids=test_id)
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "New document"

        # Upsert same ID with different data (should update)
        collection.upsert(
            ids=test_id,
            embeddings=[2.0, 3.0, 4.0],
            documents="Updated document",
            metadatas={"status": "updated"},
        )

        # Verify it was updated
        results = collection.get(ids=test_id)
        assert results["documents"][0] == "Updated document"
        assert results["metadatas"][0]["status"] == "updated"

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_delete(self, db_client):
        """Test delete operation with v1 collections"""
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection_name = f"test_v1_delete_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Add items
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        collection.add(
            ids=test_ids,
            embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        # Delete by ID
        collection.delete(ids=test_ids[0])

        # Verify deletion
        results = collection.get(ids=test_ids)
        assert len(results["ids"]) == 2
        assert test_ids[0] not in results["ids"]
        assert test_ids[1] in results["ids"]
        assert test_ids[2] in results["ids"]

        # Delete multiple IDs
        collection.delete(ids=test_ids[1:])

        # Verify all deleted
        results = collection.get(ids=test_ids)
        assert len(results["ids"]) == 0

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_delete_with_filters(self, db_client):
        """Test delete with where filters on v1 collections"""
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection_name = f"test_v1_delete_filter_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Add items with metadata
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        collection.add(
            ids=test_ids,
            embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[
                {"category": "A", "score": 10},
                {"category": "B", "score": 20},
                {"category": "A", "score": 30},
            ],
        )

        # Delete with where filter
        collection.delete(where={"category": "A"})

        # Verify only category A items deleted
        results = collection.get(ids=test_ids)
        assert len(results["ids"]) == 1
        assert results["metadatas"][0]["category"] == "B"

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_query(self, db_client):
        """Test query operation with v1 collections"""
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection_name = f"test_v1_query_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Add test data
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        collection.add(
            ids=test_ids,
            embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.1, 2.1, 3.1]],
            documents=["Document 1", "Document 2", "Document 3"],
            metadatas=[
                {"category": "A"},
                {"category": "B"},
                {"category": "A"},
            ],
        )

        collection.refresh_index()

        # Query with vector similarity
        query_vector = [1.0, 2.0, 3.0]
        results = collection.query(
            query_embeddings=query_vector,
            n_results=2,
        )

        assert len(results["ids"]) == 1  # One query vector
        assert len(results["ids"][0]) == 2  # Two results
        assert "distances" in results
        assert len(results["distances"][0]) == 2

        # Query with where filter
        results = collection.query(
            query_embeddings=query_vector,
            n_results=10,
            where={"category": "A"},
        )

        # Should only return category A items
        assert len(results["ids"][0]) == 2
        for metadata in results["metadatas"][0]:
            assert metadata["category"] == "A"

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_count(self, db_client):
        """Test count operation with v1 collections"""
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        collection_name = f"test_v1_count_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=collection_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )

        # Initially empty
        assert collection.count() == 0

        # Add items
        test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        collection.add(
            ids=test_ids,
            embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        # Should have 3 items
        assert collection.count() == 3

        # Delete one
        collection.delete(ids=test_ids[0])
        assert collection.count() == 2

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_with_embedding_function(self, db_client):
        """Test v1 collection works with embedding function"""
        from pyseekdb import DefaultEmbeddingFunction

        collection_name = f"test_v1_ef_{int(time.time() * 1000)}"
        ef = DefaultEmbeddingFunction()

        # Create v1 collection with embedding function
        collection = db_client.create_collection(
            name=collection_name,
            embedding_function=ef,
            _collection_version=1,
        )

        # Add documents (should auto-generate embeddings)
        test_id = str(uuid.uuid4())
        collection.add(
            ids=test_id,
            documents="Test document for embedding",
        )

        # Verify document was added
        results = collection.get(ids=test_id)
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "Test document for embedding"

        # Query with text (should auto-generate embeddings)
        query_results = collection.query(
            query_texts="Test query",
            n_results=1,
        )

        assert len(query_results["ids"][0]) >= 0  # May or may not find results

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_table_name_format(self, db_client):
        """Test that v1 collections use correct table name format"""
        collection_name = f"test_v1_table_{int(time.time() * 1000)}"

        # Create v1 collection
        db_client.create_collection(
            name=collection_name,
            _collection_version=1,
        )

        # Verify table name uses v1 prefix
        # The table name should start with c$v1$ prefix
        table_name = CollectionNames.table_name(collection_name)
        assert table_name.startswith("c$v1$")
        assert collection_name in table_name

        # Cleanup
        db_client.delete_collection(collection_name)

    def test_v1_collection_delete_collection(self, db_client):
        """Test deleting a v1 collection"""
        collection_name = f"test_v1_del_coll_{int(time.time() * 1000)}"

        # Create v1 collection
        db_client.create_collection(name=collection_name, _collection_version=1)

        # Verify it exists
        assert db_client.has_collection(collection_name)

        # Delete collection
        db_client.delete_collection(collection_name)

        # Verify it's gone
        assert not db_client.has_collection(collection_name)

        # Should raise error when trying to get deleted collection
        with pytest.raises(ValueError, match="does not exist"):
            db_client.get_collection(collection_name)

    def test_v1_v2_collections_coexist(self, db_client):
        """Test that v1 and v2 collections can coexist"""
        v1_name = f"test_v1_coexist_{int(time.time() * 1000)}"
        v2_name = f"test_v2_coexist_{int(time.time() * 1000)}"

        # Create v1 collection
        hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        v1_collection = db_client.create_collection(
            name=v1_name,
            configuration=hnsw_config,
            embedding_function=None,
            _collection_version=1,
        )
        assert v1_collection.id is None

        # Create v2 collection (default)
        v2_collection = db_client.create_collection(name=v2_name, configuration=hnsw_config, embedding_function=None)
        assert v2_collection.id is not None

        # Both should be in list
        collections = db_client.list_collections()
        names = [c.name for c in collections]
        assert v1_name in names
        assert v2_name in names

        # Both should work independently
        v1_collection.add(ids="v1_id", embeddings=[1.0, 2.0, 3.0], documents="V1 doc")
        v2_collection.add(ids="v2_id", embeddings=[1.0, 2.0, 3.0], documents="V2 doc")

        v1_results = v1_collection.get(ids="v1_id")
        v2_results = v2_collection.get(ids="v2_id")

        assert v1_results["documents"][0] == "V1 doc"
        assert v2_results["documents"][0] == "V2 doc"

        # Cleanup
        db_client.delete_collection(v1_name)
        db_client.delete_collection(v2_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
