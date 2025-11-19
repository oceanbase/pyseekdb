"""
Collection DML tests - testing collection.add(), collection.delete(), collection.upsert(), collection.update() interfaces for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
import uuid
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb
from pyseekdb.client.meta_info import CollectionNames, CollectionFieldNames


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb_store"))
SEEKDB_DATABASE = os.environ.get('SEEKDB_DATABASE', 'test')

# Server mode
SERVER_HOST = os.environ.get('SERVER_HOST', '127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))
SERVER_DATABASE = os.environ.get('SERVER_DATABASE', 'test')
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SERVER_PASSWORD', '')

# OceanBase mode
OB_HOST = os.environ.get('OB_HOST', 'localhost')
OB_PORT = int(os.environ.get('OB_PORT', '11202'))
OB_TENANT = os.environ.get('OB_TENANT', 'mysql')
OB_DATABASE = os.environ.get('OB_DATABASE', 'test')
OB_USER = os.environ.get('OB_USER', 'root')
OB_PASSWORD = os.environ.get('OB_PASSWORD', '')


class TestCollectionDML:
    """Test collection DML operations (add, delete, upsert, update) for all three modes"""
    
    def test_embedded_collection_dml(self):
        """Test collection DML operations with embedded client"""
        # Check if seekdb package is available
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("SeekDB embedded package is not installed")
        
        # Create embedded client
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.SeekdbEmbeddedClient)
        
        # Create test collection using execute
        collection_name = f"test_dml_{int(time.time())}"
        table_name = CollectionNames.table_name(collection_name)
        dimension = 3
        
        # Create table using execute
        create_table_sql = f"""CREATE TABLE `{table_name}` (
            {CollectionFieldNames.ID} varbinary(512) PRIMARY KEY NOT NULL,
            {CollectionFieldNames.DOCUMENT} string,
            {CollectionFieldNames.EMBEDDING} vector({dimension}),
            {CollectionFieldNames.METADATA} json,
            FULLTEXT INDEX idx_fts({CollectionFieldNames.DOCUMENT}),
            VECTOR INDEX idx_vec ({CollectionFieldNames.EMBEDDING}) with(distance=cosine, type=hnsw, lib=vsag)
        ) ORGANIZATION = HEAP;"""
        client._server.execute(create_table_sql)
        
        # Get collection object
        collection = client.get_collection(name=collection_name, embedding_function=None)
        
        try:
            # Test 1: collection.add - Add single item
            print(f"\n✅ Testing collection.add() - single item for embedded client")
            test_id_1 = str(uuid.uuid4())
            collection.add(
                ids=test_id_1,
                embeddings=[1.0, 2.0, 3.0],
                documents="This is test document 1",
                metadatas={"category": "test", "score": 100}
            )
            
            # Verify using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            assert results["ids"][0] == test_id_1
            assert results["documents"][0] == "This is test document 1"
            assert results["metadatas"][0].get('category') == "test"
            print(f"   Successfully added and verified item with ID: {test_id_1}")
            
            # Test 2: collection.add - Add multiple items
            print(f"✅ Testing collection.add() - multiple items")
            test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
            collection.add(
                ids=test_ids,
                embeddings=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                documents=["Document 2", "Document 3", "Document 4"],
                metadatas=[
                    {"category": "test", "score": 90},
                    {"category": "test", "score": 85},
                    {"category": "demo", "score": 80}
                ]
            )
            
            # Verify using collection.get
            results = collection.get(ids=test_ids)
            assert len(results["ids"]) == 3
            print(f"   Successfully added and verified {len(results['ids'])} items")
            
            # Test 3: collection.update - Update existing item
            print(f"✅ Testing collection.update() - update existing item")
            collection.update(
                ids=test_id_1,
                metadatas={"category": "test", "score": 95, "updated": True}
            )
            
            # Verify update using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            # Document should remain unchanged since we didn't update it
            assert results["documents"][0] == "This is test document 1"
            assert results["metadatas"][0].get('score') == 95
            assert results["metadatas"][0].get('updated') is True
            print(f"   Successfully updated and verified item with ID: {test_id_1}")
            
            # Test 4: collection.upsert - Upsert existing item (should update)
            print(f"✅ Testing collection.upsert() - upsert existing item (update)")
            collection.upsert(
                ids=test_id_1,
                embeddings=[1.0, 2.0, 3.0],  # Use original vector
                documents="Upserted document 1",
                metadatas={"category": "test", "score": 98}
            )
            
            # Verify upsert using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "Upserted document 1"
            assert results["metadatas"][0].get('score') == 98
            print(f"   Successfully upserted (update) and verified item with ID: {test_id_1}")
            
            # Test 5: collection.upsert - Upsert new item (should insert)
            print(f"✅ Testing collection.upsert() - upsert new item (insert)")
            test_id_new = str(uuid.uuid4())
            collection.upsert(
                ids=test_id_new,
                embeddings=[5.0, 6.0, 7.0],
                documents="New upserted document",
                metadatas={"category": "new", "score": 99}
            )
            
            # Verify upsert using collection.get
            results = collection.get(ids=test_id_new)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "New upserted document"
            assert results["metadatas"][0].get('category') == "new"
            print(f"   Successfully upserted (insert) and verified item with ID: {test_id_new}")
            
            # Test 6: collection.delete - Delete by ID
            print(f"✅ Testing collection.delete() - delete by ID")
            # Delete one of the test items
            collection.delete(ids=test_ids[0])
            
            # Verify deletion using collection.get
            results = collection.get(ids=test_ids[0])
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted item with ID: {test_ids[0]}")
            
            # Verify other items still exist
            results = collection.get(ids=test_ids[1:])
            assert len(results["ids"]) == 2
            print(f"   Verified other items still exist")
            
            # Test 7: collection.delete - Delete by metadata filter
            print(f"✅ Testing collection.delete() - delete by metadata filter")
            # Delete items with category="demo"
            collection.delete(where={"category": {"$eq": "demo"}})
            
            # Verify deletion using collection.get
            results = collection.get(where={"category": {"$eq": "demo"}})
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted items with category='demo'")
            
            # Test 8: Verify final state using collection.get
            print(f"✅ Testing final state verification")
            all_results = collection.get(limit=100)
            print(f"   Final collection count: {len(all_results['ids'])} items")
            assert len(all_results["ids"]) > 0
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
    
    def test_server_collection_dml(self):
        """Test collection DML operations with server client"""
        # Create server client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",  # Default tenant for SeekDB Server
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection using execute
        collection_name = f"test_dml_{int(time.time())}"
        table_name = CollectionNames.table_name(collection_name)
        dimension = 3
        
        # Create table using execute
        create_table_sql = f"""CREATE TABLE `{table_name}` (
            {CollectionFieldNames.ID} varbinary(512) PRIMARY KEY NOT NULL,
            {CollectionFieldNames.DOCUMENT} string,
            {CollectionFieldNames.EMBEDDING} vector({dimension}),
            {CollectionFieldNames.METADATA} json,
            FULLTEXT INDEX idx_fts({CollectionFieldNames.DOCUMENT}),
            VECTOR INDEX idx_vec ({CollectionFieldNames.EMBEDDING}) with(distance=cosine, type=hnsw, lib=vsag)
        ) ORGANIZATION = HEAP;"""
        client._server.execute(create_table_sql)
        
        # Get collection object
        collection = client.get_collection(name=collection_name, embedding_function=None)
        
        try:
            # Test 1: collection.add - Add single item
            print(f"\n✅ Testing collection.add() - single item for server client")
            test_id_1 = str(uuid.uuid4())
            collection.add(
                ids=test_id_1,
                embeddings=[1.0, 2.0, 3.0],
                documents="This is test document 1",
                metadatas={"category": "test", "score": 100}
            )
            
            # Verify using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            assert results["ids"][0] == test_id_1
            assert results["documents"][0] == "This is test document 1"
            assert results["metadatas"][0].get('category') == "test"
            print(f"   Successfully added and verified item with ID: {test_id_1}")
            
            # Test 2: collection.add - Add multiple items
            print(f"✅ Testing collection.add() - multiple items")
            test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
            collection.add(
                ids=test_ids,
                embeddings=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                documents=["Document 2", "Document 3", "Document 4"],
                metadatas=[
                    {"category": "test", "score": 90},
                    {"category": "test", "score": 85},
                    {"category": "demo", "score": 80}
                ]
            )
            
            # Verify using collection.get
            results = collection.get(ids=test_ids)
            assert len(results["ids"]) == 3
            print(f"   Successfully added and verified {len(results['ids'])} items")
            
            # Test 3: collection.update - Update existing item
            print(f"✅ Testing collection.update() - update existing item")
            collection.update(
                ids=test_id_1,
                metadatas={"category": "test", "score": 95, "updated": True}
            )
            
            # Verify update using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            # Document should remain unchanged since we didn't update it
            assert results["documents"][0] == "This is test document 1"
            assert results["metadatas"][0].get('score') == 95
            assert results["metadatas"][0].get('updated') is True
            print(f"   Successfully updated and verified item with ID: {test_id_1}")
            
            # Test 4: collection.update - Update multiple items
            print(f"✅ Testing collection.update() - update multiple items")
            collection.update(
                ids=test_ids[:2],
                embeddings=[[2.1, 3.1, 4.1], [3.1, 4.1, 5.1]],
                metadatas=[
                    {"category": "test", "score": 92},
                    {"category": "test", "score": 87}
                ]
            )
            
            # Verify update using collection.get
            results = collection.get(ids=test_ids[:2])
            assert len(results["ids"]) == 2
            print(f"   Successfully updated and verified {len(results)} items")
            
            # Test 5: collection.upsert - Upsert existing item (should update)
            print(f"✅ Testing collection.upsert() - upsert existing item (update)")
            collection.upsert(
                ids=test_id_1,
                embeddings=[1.0, 2.0, 3.0],  # Use original vector
                documents="Upserted document 1",
                metadatas={"category": "test", "score": 98}
            )
            
            # Verify upsert using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "Upserted document 1"
            assert results["metadatas"][0].get('score') == 98
            print(f"   Successfully upserted (update) and verified item with ID: {test_id_1}")
            
            # Test 6: collection.upsert - Upsert new item (should insert)
            print(f"✅ Testing collection.upsert() - upsert new item (insert)")
            test_id_new = str(uuid.uuid4())
            collection.upsert(
                ids=test_id_new,
                embeddings=[5.0, 6.0, 7.0],
                documents="New upserted document",
                metadatas={"category": "new", "score": 99}
            )
            
            # Verify upsert using collection.get
            results = collection.get(ids=test_id_new)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "New upserted document"
            assert results["metadatas"][0].get('category') == "new"
            print(f"   Successfully upserted (insert) and verified item with ID: {test_id_new}")
            
            # Test 7: collection.delete - Delete by ID
            print(f"✅ Testing collection.delete() - delete by ID")
            # Delete one of the test items
            collection.delete(ids=test_ids[0])
            
            # Verify deletion using collection.get
            results = collection.get(ids=test_ids[0])
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted item with ID: {test_ids[0]}")
            
            # Verify other items still exist
            results = collection.get(ids=test_ids[1:])
            assert len(results["ids"]) == 2
            print(f"   Verified other items still exist")
            
            # Test 8: collection.delete - Delete by metadata filter
            print(f"✅ Testing collection.delete() - delete by metadata filter")
            # Delete items with category="demo"
            collection.delete(where={"category": {"$eq": "demo"}})
            
            # Verify deletion using collection.get
            results = collection.get(where={"category": {"$eq": "demo"}})
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted items with category='demo'")
            
            # Test 9: collection.delete - Delete by document filter
            print(f"✅ Testing collection.delete() - delete by document filter")
            # Add an item with specific document content
            test_id_doc = str(uuid.uuid4())
            collection.add(
                ids=test_id_doc,
                embeddings=[6.0, 7.0, 8.0],
                documents="Delete this document",
                metadatas={"category": "temp"}
            )
            
            # Delete by document filter
            collection.delete(where_document={"$contains": "Delete this"})
            
            # Verify deletion using collection.get
            results = collection.get(where_document={"$contains": "Delete this"})
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted items by document filter")
            
            # Test 10: Verify final state using collection.get
            print(f"✅ Testing final state verification")
            all_results = collection.get(limit=100)
            print(f"   Final collection count: {len(all_results)} items")
            assert len(all_results) > 0
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
    
    def test_oceanbase_collection_dml(self):
        """Test collection DML operations with OceanBase client"""
        # Create OceanBase client
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection using execute
        collection_name = f"test_dml_{int(time.time())}"
        table_name = CollectionNames.table_name(collection_name)
        dimension = 3
        
        # Create table using execute
        create_table_sql = f"""CREATE TABLE `{table_name}` (
            {CollectionFieldNames.ID} varbinary(512) PRIMARY KEY NOT NULL,
            {CollectionFieldNames.DOCUMENT} string,
            {CollectionFieldNames.EMBEDDING} vector({dimension}),
            {CollectionFieldNames.METADATA} json,
            FULLTEXT INDEX idx_fts({CollectionFieldNames.DOCUMENT}),
            VECTOR INDEX idx_vec ({CollectionFieldNames.EMBEDDING}) with(distance=cosine, type=hnsw, lib=vsag)
        ) ORGANIZATION = HEAP;"""
        client._server.execute(create_table_sql)
        
        # Get collection object
        collection = client.get_collection(name=collection_name, embedding_function=None)
        
        try:
            # Test 1: collection.add - Add single item
            print(f"\n✅ Testing collection.add() - single item for OceanBase client")
            test_id_1 = str(uuid.uuid4())
            collection.add(
                ids=test_id_1,
                embeddings=[1.0, 2.0, 3.0],
                documents="This is test document 1",
                metadatas={"category": "test", "score": 100}
            )
            
            # Verify using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            assert results["ids"][0] == test_id_1
            assert results["documents"][0] == "This is test document 1"
            assert results["metadatas"][0].get('category') == "test"
            print(f"   Successfully added and verified item with ID: {test_id_1}")
            
            # Test 2: collection.add - Add multiple items
            print(f"✅ Testing collection.add() - multiple items")
            test_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
            collection.add(
                ids=test_ids,
                embeddings=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                documents=["Document 2", "Document 3", "Document 4"],
                metadatas=[
                    {"category": "test", "score": 90},
                    {"category": "test", "score": 85},
                    {"category": "demo", "score": 80}
                ]
            )
            
            # Verify using collection.get
            results = collection.get(ids=test_ids)
            assert len(results["ids"]) == 3
            print(f"   Successfully added and verified {len(results['ids'])} items")
            
            # Test 3: collection.update - Update existing item
            print(f"✅ Testing collection.update() - update existing item")
            collection.update(
                ids=test_id_1,
                metadatas={"category": "test", "score": 95, "updated": True}
            )
            
            # Verify update using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            # Document should remain unchanged since we didn't update it
            assert results["documents"][0] == "This is test document 1"
            assert results["metadatas"][0].get('score') == 95
            assert results["metadatas"][0].get('updated') is True
            print(f"   Successfully updated and verified item with ID: {test_id_1}")
            
            # Test 4: collection.update - Update multiple items with embeddings
            print(f"✅ Testing collection.update() - update multiple items with embeddings")
            collection.update(
                ids=test_ids[:2],
                embeddings=[[2.1, 3.1, 4.1], [3.1, 4.1, 5.1]],
                metadatas=[
                    {"category": "test", "score": 92},
                    {"category": "test", "score": 87}
                ]
            )
            
            # Verify update using collection.get
            results = collection.get(ids=test_ids[:2], include=["embeddings"])
            assert len(results) == 2
            print(f"   Successfully updated and verified {len(results)} items with embeddings")
            
            # Test 5: collection.upsert - Upsert existing item (should update)
            print(f"✅ Testing collection.upsert() - upsert existing item (update)")
            collection.upsert(
                ids=test_id_1,
                embeddings=[1.0, 2.0, 3.0],  # Use original vector
                documents="Upserted document 1",
                metadatas={"category": "test", "score": 98}
            )
            
            # Verify upsert using collection.get
            results = collection.get(ids=test_id_1)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "Upserted document 1"
            assert results["metadatas"][0].get('score') == 98
            print(f"   Successfully upserted (update) and verified item with ID: {test_id_1}")
            
            # Test 6: collection.upsert - Upsert new item (should insert)
            print(f"✅ Testing collection.upsert() - upsert new item (insert)")
            test_id_new = str(uuid.uuid4())
            collection.upsert(
                ids=test_id_new,
                embeddings=[5.0, 6.0, 7.0],
                documents="New upserted document",
                metadatas={"category": "new", "score": 99}
            )
            
            # Verify upsert using collection.get
            results = collection.get(ids=test_id_new)
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "New upserted document"
            assert results["metadatas"][0].get('category') == "new"
            print(f"   Successfully upserted (insert) and verified item with ID: {test_id_new}")
            
            # Test 7: collection.delete - Delete by ID
            print(f"✅ Testing collection.delete() - delete by ID")
            # Delete one of the test items
            collection.delete(ids=test_ids[0])
            
            # Verify deletion using collection.get
            results = collection.get(ids=test_ids[0])
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted item with ID: {test_ids[0]}")
            
            # Verify other items still exist
            results = collection.get(ids=test_ids[1:])
            assert len(results["ids"]) == 2
            print(f"   Verified other items still exist")
            
            # Test 8: collection.delete - Delete by metadata filter with comparison operator
            print(f"✅ Testing collection.delete() - delete by metadata filter with $gte")
            # Add items with different scores
            test_id_score = str(uuid.uuid4())
            collection.add(
                ids=test_id_score,
                embeddings=[7.0, 8.0, 9.0],
                documents="High score document",
                metadatas={"category": "test", "score": 99}
            )
            
            # Delete items with score >= 99
            collection.delete(where={"score": {"$gte": 99}})
            
            # Verify deletion using collection.get
            results = collection.get(where={"score": {"$gte": 99}})
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted items with score >= 99")
            
            # Test 9: collection.delete - Delete by document filter
            print(f"✅ Testing collection.delete() - delete by document filter")
            # Add an item with specific document content
            test_id_doc = str(uuid.uuid4())
            collection.add(
                ids=test_id_doc,
                embeddings=[6.0, 7.0, 8.0],
                documents="Delete this document",
                metadatas={"category": "temp"}
            )
            
            # Delete by document filter
            collection.delete(where_document={"$contains": "Delete this"})
            
            # Verify deletion using collection.get
            results = collection.get(where_document={"$contains": "Delete this"})
            assert len(results["ids"]) == 0
            print(f"   Successfully deleted items by document filter")
            
            # Test 10: Verify final state using collection.get with filters
            print(f"✅ Testing final state verification")
            all_results = collection.get(limit=100)
            print(f"   Final collection count: {len(all_results['ids'])} items")
            assert len(all_results["ids"]) > 0
            
            # Verify by category filter
            test_results = collection.get(where={"category": {"$eq": "test"}})
            print(f"   Items with category='test': {len(test_results['ids'])}")
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SeekDBClient - Collection DML Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

