"""
Collection query tests - testing collection.query() interface for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
import json
import uuid
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb_store"))
SEEKDB_DATABASE = os.environ.get('SEEKDB_DATABASE', 'test')

# Server mode
SERVER_HOST = os.environ.get('SERVER_HOST', 'localhost')
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


class TestCollectionQuery:
    """Test collection.query() interface for all three modes"""
    
    def _insert_test_data(self, client, collection_name: str):
        """Helper method to insert test data using direct SQL"""
        table_name = f"c$v1${collection_name}"
        
        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "_id": str(uuid.uuid4()),
                "document": "This is a test document about machine learning",
                "embedding": [1.0, 2.0, 3.0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Python programming tutorial for beginners",
                "embedding": [2.0, 3.0, 4.0],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Advanced machine learning algorithms",
                "embedding": [1.1, 2.1, 3.1],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Data science with Python",
                "embedding": [2.1, 3.1, 4.1],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Introduction to neural networks",
                "embedding": [1.2, 2.2, 3.2],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"}
            }
        ]
        
        for data in test_data:
            # Use string ID directly (support any string format)
            id_str = data["_id"]
            # Escape single quotes in ID
            id_str_escaped = id_str.replace("'", "''")
            
            # Convert vector to string format: [1.0,2.0,3.0]
            vector_str = "[" + ",".join(map(str, data["embedding"])) + "]"
            # Convert metadata to JSON string
            metadata_str = json.dumps(data["metadata"]).replace("'", "\\'")
            # Escape single quotes in document
            document_str = data["document"].replace("'", "\\'")
            
            # Use CAST to convert string to binary for varbinary(512) field
            sql = f"""INSERT INTO `{table_name}` (_id, document, embedding, metadata) 
                     VALUES (CAST('{id_str_escaped}' AS BINARY), '{document_str}', '{vector_str}', '{metadata_str}')"""
            client._server.execute(sql)
    
    def test_embedded_collection_query(self):
        """Test collection.query() with embedded client"""
        if not os.path.exists(SEEKDB_PATH):
            pytest.skip(
                f"SeekDB data directory does not exist: {SEEKDB_PATH}\n"
                f"Set SEEKDB_PATH environment variable to run this test"
            )
        
        # Check if seekdb package is available
        try:
            import seekdb
        except ImportError:
            pytest.skip("SeekDB embedded package is not installed")
        
        # Create embedded client
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.SeekdbEmbeddedClient)
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        collection = client.create_collection(name=collection_name, dimension=3)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for embedded client")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results")
            
            # Test 2: Query with metadata filter
            print(f"✅ Testing query with metadata filter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"category": {"$eq": "AI"}},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with category='AI'")
            
            # Test 3: Query with document filter
            print(f"✅ Testing query with document filter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where_document={"$contains": "machine learning"},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results containing 'machine learning'")
            
            # Test 4: Query with include parameter
            print(f"✅ Testing query with include parameter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                include=["documents", "metadatas"],
                n_results=3
            )
            assert results is not None
            if len(results) > 0:
                # Check that results have the expected fields
                for item in results:
                    assert hasattr(item, '_id') or '_id' in item.to_dict()
            
            # Test 5: Query with multiple vectors (should return List[QueryResult])
            print(f"✅ Testing query with multiple vectors (returns List[QueryResult])")
            results = collection.query(
                query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, list), "Multiple vectors should return List[QueryResult]"
            assert len(results) == 2, f"Expected 2 QueryResult objects, got {len(results)}"
            for i, result in enumerate(results):
                assert isinstance(result, pyseekdb.QueryResult), f"Result {i} should be QueryResult"
                assert len(result) > 0, f"QueryResult {i} should have at least one item"
                print(f"   QueryResult {i}: {len(result)} items")
            
            # Test 6: Single vector still returns single QueryResult (backward compatibility)
            print(f"✅ Testing single vector returns single QueryResult (backward compatibility)")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, pyseekdb.QueryResult), "Single vector should return QueryResult, not list"
            assert len(results) > 0
            print(f"   Single QueryResult with {len(results)} items")
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
    
    def test_server_collection_query(self):
        """Test collection.query() with server client"""
        # Create server client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.SeekdbServerClient)
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.skip(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        collection = client.create_collection(name=collection_name, dimension=3)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for server client")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results")
            
            # Test 2: Query with metadata filter using comparison operators
            print(f"✅ Testing query with metadata filter ($gte)")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"score": {"$gte": 90}},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with score >= 90")
            
            # Test 3: Query with combined filters
            print(f"✅ Testing query with combined filters")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
                where_document={"$contains": "machine"},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results matching all filters")
            
            # Test 4: Query with $in operator
            print(f"✅ Testing query with $in operator")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"tag": {"$in": ["ml", "python"]}},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with tag in ['ml', 'python']")
            
            # Test 5: Query with multiple vectors (should return List[QueryResult])
            print(f"✅ Testing query with multiple vectors (returns List[QueryResult])")
            results = collection.query(
                query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.1, 2.1, 3.1]],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, list), "Multiple vectors should return List[QueryResult]"
            assert len(results) == 3, f"Expected 3 QueryResult objects, got {len(results)}"
            for i, result in enumerate(results):
                assert isinstance(result, pyseekdb.QueryResult), f"Result {i} should be QueryResult"
                assert len(result) > 0, f"QueryResult {i} should have at least one item"
                print(f"   QueryResult {i}: {len(result)} items")
            
            # Test 6: Single vector still returns single QueryResult (backward compatibility)
            print(f"✅ Testing single vector returns single QueryResult (backward compatibility)")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, pyseekdb.QueryResult), "Single vector should return QueryResult, not list"
            assert len(results) > 0
            print(f"   Single QueryResult with {len(results)} items")
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
    
    def test_oceanbase_collection_query(self):
        """Test collection.query() with OceanBase client"""
        # Create OceanBase client
        client = pyseekdb.OBClient(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.OceanBaseServerClient)
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.skip(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        collection = client.create_collection(name=collection_name, dimension=3)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for OceanBase client")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results")
            
            # Test 2: Query with multiple vectors (should return List[QueryResult])
            print(f"✅ Testing query with multiple vectors (returns List[QueryResult])")
            results = collection.query(
                query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, list), "Multiple vectors should return List[QueryResult]"
            assert len(results) == 2, f"Expected 2 QueryResult objects, got {len(results)}"
            for i, result in enumerate(results):
                assert isinstance(result, pyseekdb.QueryResult), f"Result {i} should be QueryResult"
                assert len(result) > 0, f"QueryResult {i} should have at least one item"
                print(f"   QueryResult {i}: {len(result)} items")
            
            # Test 3: Query with logical operators
            print(f"✅ Testing query with logical operators ($or)")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={
                    "$or": [
                        {"category": {"$eq": "AI"}},
                        {"tag": {"$eq": "python"}}
                    ]
                },
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with $or condition")
            
            # Test 4: Query with include parameter to get specific fields
            print(f"✅ Testing query with include parameter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                include=["documents", "metadatas", "embeddings"],
                n_results=3
            )
            assert results is not None
            if len(results) > 0:
                # Verify result structure
                for item in results:
                    result_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                    assert '_id' in result_dict
                    print(f"   Result keys: {list(result_dict.keys())}")
            
            # Test 5: Single vector still returns single QueryResult (backward compatibility)
            print(f"✅ Testing single vector returns single QueryResult (backward compatibility)")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, pyseekdb.QueryResult), "Single vector should return QueryResult, not list"
            assert len(results) > 0
            print(f"   Single QueryResult with {len(results)} items")
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SeekDBClient - Collection Query Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

