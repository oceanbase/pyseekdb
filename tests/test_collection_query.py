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


class TestCollectionQuery:
    """Test collection.query() interface for all three modes"""
    
    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        """Helper method to insert test data using direct SQL
        
        Args:
            client: Client instance
            collection_name: Collection name
            dimension: Actual dimension of the collection (used to generate vectors)
        """
        table_name = f"c$v1${collection_name}"
        
        # Base vectors (3D) - will be extended or truncated to match actual dimension
        base_vectors = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1],
            [2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2]
        ]
        
        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "document": "This is a test document about machine learning",
                "base_vector": base_vectors[0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"}
            },
            {
                "document": "Python programming tutorial for beginners",
                "base_vector": base_vectors[1],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"}
            },
            {
                "document": "Advanced machine learning algorithms",
                "base_vector": base_vectors[2],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"}
            },
            {
                "document": "Data science with Python",
                "base_vector": base_vectors[3],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"}
            },
            {
                "document": "Introduction to neural networks",
                "base_vector": base_vectors[4],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"}
            }
        ]
        
        for data in test_data:
            # Generate UUID for _id (use string format directly)
            id_str = str(uuid.uuid4())
            # Escape single quotes in ID
            id_str_escaped = id_str.replace("'", "''")
            
            # Generate vector with correct dimension
            base_vec = data["base_vector"]
            if dimension <= len(base_vec):
                # Truncate if dimension is smaller
                embedding = base_vec[:dimension]
            else:
                # Extend if dimension is larger (repeat pattern)
                embedding = base_vec * ((dimension // len(base_vec)) + 1)
                embedding = embedding[:dimension]
            
            # Convert vector to string format: [1.0,2.0,3.0]
            vector_str = "[" + ",".join(map(str, embedding)) + "]"
            # Convert metadata to JSON string
            metadata_str = json.dumps(data["metadata"], ensure_ascii=False).replace("'", "\\'")
            # Escape single quotes in document
            document_str = data["document"].replace("'", "\\'")
            
            # Use CAST to convert string to binary for varbinary(512) field
            sql = f"""INSERT INTO `{table_name}` (_id, document, embedding, metadata) 
                     VALUES (CAST('{id_str_escaped}' AS BINARY), '{document_str}', '{vector_str}', '{metadata_str}')"""
            client._server.execute(sql)
        
        print(f"   Inserted {len(test_data)} test records (dimension={dimension})")
    
    def test_embedded_collection_query(self):
        """Test collection.query() with embedded client"""
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
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        collection = client.create_collection(name=collection_name, configuration=config, embedding_function=None)
        # Get actual dimension (may be different from requested due to default embedding function)
        actual_dimension = collection.dimension
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for embedded client")
            # Generate query vector with correct dimension
            query_vector = [1.0, 2.0, 3.0] * ((actual_dimension // 3) + 1)
            query_vector = query_vector[:actual_dimension]
            results = collection.query(
                query_embeddings=query_vector,
                n_results=3
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Test 2: Query with metadata filter (simplified equality)
            print(f"✅ Testing query with metadata filter")
            results = collection.query(
                query_embeddings=query_vector,
                where={"category": "AI"},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results with category='AI'")
            
            # Test 3: Query with document filter
            print(f"✅ Testing query with document filter")
            results = collection.query(
                query_embeddings=query_vector,
                where_document={"$contains": "machine learning"},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results containing 'machine learning'")
            
            # Test 3.5: Query with document filter using regex
            print(f"✅ Testing query with document filter using regex")
            results = collection.query(
                query_embeddings=query_vector,
                where_document={"$regex": ".*machine.*"},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results matching regex '.*machine.*'")
            
            # Test 4: Query with include parameter
            print(f"✅ Testing query with include parameter")
            results = collection.query(
                query_embeddings=query_vector,
                include=["documents", "metadatas"],
                n_results=3
            )
            assert results is not None
            assert "ids" in results
            if len(results["ids"][0]) > 0:
                # Check that results have the expected fields
                assert "documents" in results
                assert "metadatas" in results
                assert len(results["ids"][0]) == len(results["documents"][0])
                assert len(results["ids"][0]) == len(results["metadatas"][0])
            
            # Test 5: Query with multiple vectors (should return dict with lists of lists)
            print(f"✅ Testing query with multiple vectors (returns dict with lists of lists)")
            query_vector2 = [2.0, 3.0, 4.0] * ((actual_dimension // 3) + 1)
            query_vector2 = query_vector2[:actual_dimension]
            results = collection.query(
                query_embeddings=[query_vector, query_vector2],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, dict), "Multiple vectors should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 2, f"Expected 2 ID lists, got {len(results['ids'])}"
            for i in range(len(results["ids"])):
                assert len(results["ids"][i]) > 0, f"ID list {i} should have at least one item"
                print(f"   Query {i}: {len(results['ids'][i])} items")
            
            # Test 6: Single vector returns dict with single list in lists
            print(f"✅ Testing single vector returns dict format")
            results = collection.query(
                query_embeddings=query_vector,
                n_results=2
            )
            assert results is not None
            assert isinstance(results, dict), "Single vector should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 1, "Single query should have one ID list"
            assert len(results["ids"][0]) > 0
            print(f"   Single query with {len(results['ids'][0])} items")
            
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
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        collection = client.create_collection(name=collection_name, configuration=config, embedding_function=None)
        # Get actual dimension (may be different from requested due to default embedding function)
        actual_dimension = collection.dimension
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for server client")
            # Generate query vector with correct dimension
            query_vector = [1.0, 2.0, 3.0] * ((actual_dimension // 3) + 1)
            query_vector = query_vector[:actual_dimension]
            results = collection.query(
                query_embeddings=query_vector,
                n_results=3
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Test 2: Query with metadata filter using comparison operators
            print(f"✅ Testing query with metadata filter ($gte)")
            results = collection.query(
                query_embeddings=query_vector,
                where={"score": {"$gte": 90}},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results with score >= 90")
            
            # Test 3: Query with combined filters
            print(f"✅ Testing query with combined filters")
            results = collection.query(
                query_embeddings=query_vector,
                where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
                where_document={"$contains": "machine"},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results matching all filters")
            
            # Test 3.5: Query with document filter using regex
            print(f"✅ Testing query with document filter using regex")
            results = collection.query(
                query_embeddings=query_vector,
                where_document={"$regex": ".*[Pp]ython.*"},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results matching regex '.*[Pp]ython.*'")
            
            # Test 4: Query with $in operator
            print(f"✅ Testing query with $in operator")
            results = collection.query(
                query_embeddings=query_vector,
                where={"tag": {"$in": ["ml", "python"]}},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results with tag in ['ml', 'python']")
            
            # Test 5: Query with multiple vectors (should return dict with lists of lists)
            print(f"✅ Testing query with multiple vectors (returns dict with lists of lists)")
            query_vector2 = [2.0, 3.0, 4.0] * ((actual_dimension // 3) + 1)
            query_vector2 = query_vector2[:actual_dimension]
            query_vector3 = [1.1, 2.1, 3.1] * ((actual_dimension // 3) + 1)
            query_vector3 = query_vector3[:actual_dimension]
            results = collection.query(
                query_embeddings=[query_vector, query_vector2, query_vector3],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, dict), "Multiple vectors should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 3, f"Expected 3 ID lists, got {len(results['ids'])}"
            for i in range(len(results["ids"])):
                assert len(results["ids"][i]) > 0, f"ID list {i} should have at least one item"
                print(f"   Query {i}: {len(results['ids'][i])} items")
            
            # Test 6: Single vector returns dict format
            print(f"✅ Testing single vector returns dict format")
            results = collection.query(
                query_embeddings=query_vector,
                n_results=2
            )
            assert results is not None
            assert isinstance(results, dict), "Single vector should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 1, "Single query should have one ID list"
            assert len(results["ids"][0]) > 0
            print(f"   Single query with {len(results['ids'][0])} items")
            
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
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        collection = client.create_collection(name=collection_name, configuration=config, embedding_function=None)
        # Get actual dimension (may be different from requested due to default embedding function)
        actual_dimension = collection.dimension
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for OceanBase client")
            # Generate query vector with correct dimension
            query_vector = [1.0, 2.0, 3.0] * ((actual_dimension // 3) + 1)
            query_vector = query_vector[:actual_dimension]
            results = collection.query(
                query_embeddings=query_vector,
                n_results=3
            )
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Test 2: Query with multiple vectors (should return dict with lists of lists)
            print(f"✅ Testing query with multiple vectors (returns dict with lists of lists)")
            query_vector2 = [2.0, 3.0, 4.0] * ((actual_dimension // 3) + 1)
            query_vector2 = query_vector2[:actual_dimension]
            results = collection.query(
                query_embeddings=[query_vector, query_vector2],
                n_results=2
            )
            assert results is not None
            assert isinstance(results, dict), "Multiple vectors should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 2, f"Expected 2 ID lists, got {len(results['ids'])}"
            for i in range(len(results["ids"])):
                assert len(results["ids"][i]) > 0, f"ID list {i} should have at least one item"
                print(f"   Query {i}: {len(results['ids'][i])} items")
            
            # Test 3: Query with logical operators
            print(f"✅ Testing query with logical operators ($or)")
            results = collection.query(
                query_embeddings=query_vector,
                where={
                    "$or": [
                        {"category": {"$eq": "AI"}},
                        {"tag": {"$eq": "python"}}
                    ]
                },
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results with $or condition")
            
            # Test 3.5: Query with document filter using regex
            print(f"✅ Testing query with document filter using regex")
            results = collection.query(
                query_embeddings=query_vector,
                where_document={"$regex": ".*neural.*"},
                n_results=5
            )
            assert results is not None
            assert "ids" in results
            print(f"   Found {len(results['ids'][0])} results matching regex '.*neural.*'")
            
            # Test 4: Query with include parameter to get specific fields
            print(f"✅ Testing query with include parameter")
            results = collection.query(
                query_embeddings=query_vector,
                include=["documents", "metadatas", "embeddings"],
                n_results=3
            )
            assert results is not None
            assert "ids" in results
            if len(results["ids"][0]) > 0:
                # Verify result structure
                assert "documents" in results
                assert "metadatas" in results
                assert "embeddings" in results
                assert len(results["ids"][0]) == len(results["documents"][0])
                print(f"   Result has all expected fields")
            
            # Test 5: Single vector returns dict format
            print(f"✅ Testing single vector returns dict format")
            results = collection.query(
                query_embeddings=query_vector,
                n_results=2
            )
            assert results is not None
            assert isinstance(results, dict), "Single vector should return dict"
            assert "ids" in results
            assert len(results["ids"]) == 1, "Single query should have one ID list"
            assert len(results["ids"][0]) > 0
            print(f"   Single query with {len(results['ids'][0])} items")
            
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

