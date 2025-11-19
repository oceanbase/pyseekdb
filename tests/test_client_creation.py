"""
Client creation and connection tests - testing connection and query execution for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
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


class TestClientCreation:
    """Test client creation, connection, and query execution for all three modes"""
    
    def _test_collection_management(self, client):
        """
        Common test function for all collection management interfaces
        
        Args:
            client: Client proxy object with _server attribute
        """
        # Test 1: create_collection - create a new collection
        test_collection_name = "test_collection_" + str(int(time.time()))
        test_dimension = 128
        
        # Create collection with HNSW configuration
        # Note: If embedding_function is not provided, default embedding function will be used
        # and dimension will be automatically updated to 384. We need to use the actual dimension.
        # To use a specific dimension, we need to set embedding_function=None
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=test_dimension, distance='cosine')
        collection = client.create_collection(
            name=test_collection_name,
            configuration=config,
            embedding_function=None
        )
        
        # Verify collection object
        assert collection is not None
        assert collection.name == test_collection_name
        # Use actual dimension (may be different from requested due to default embedding function)
        actual_dimension = collection.dimension
        assert actual_dimension > 0, f"Collection dimension should be positive, got {actual_dimension}"
        
        # Verify table was created by checking if it exists
        table_name = f"c$v1${test_collection_name}"
        try:
            # Try to describe table structure to verify it exists
            table_info = client._server.execute(f"DESCRIBE `{table_name}`")
            assert table_info is not None
            assert len(table_info) > 0
            
            # Verify table has expected columns
            # Handle both dict (server client) and tuple (embedded client) formats
            column_names = []
            column_types = {}  # Store column types for verification
            for row in table_info:
                if isinstance(row, dict):
                    # Server client returns dict with 'Field' or 'field' key
                    field_name = row.get('Field', row.get('field', ''))
                    field_type = row.get('Type', row.get('type', ''))
                    column_names.append(field_name)
                    if field_name:
                        column_types[field_name] = str(field_type).lower()
                elif isinstance(row, (tuple, list)):
                    # Embedded client returns tuple, first element is field name, second is type
                    field_name = row[0] if len(row) > 0 else ''
                    field_type = row[1] if len(row) > 1 else ''
                    column_names.append(field_name)
                    if field_name:
                        column_types[field_name] = str(field_type).lower()
                else:
                    # Fallback: try to convert to string
                    column_names.append(str(row))
            
            assert '_id' in column_names
            assert 'document' in column_names
            assert 'embedding' in column_names
            assert 'metadata' in column_names
            
            # Verify _id column type is varbinary (for varbinary(512) type)
            if '_id' in column_types:
                id_type = column_types['_id']
                # Check if it's varbinary type (could be 'varbinary(512)' or similar)
                assert 'varbinary' in id_type, f"Expected _id to be varbinary type, but got: {id_type}"
            
            print(f"\n✅ Collection '{test_collection_name}' created successfully")
            print(f"   Table name: {table_name}")
            print(f"   Dimension: {actual_dimension}")
            print(f"   Table columns: {', '.join(column_names)}")
            
        except Exception as e:
            # Clean up and fail
            try:
                client._server.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            except Exception:
                pass
            pytest.fail(f"Failed to verify collection table creation: {e}")
        
        # Test 2: get_collection - get the collection we just created
        retrieved_collection = client.get_collection(name=test_collection_name)
        assert retrieved_collection is not None
        assert retrieved_collection.name == test_collection_name
        assert retrieved_collection.dimension == actual_dimension
        print(f"\n✅ Collection '{test_collection_name}' retrieved successfully")
        print(f"   Collection name: {retrieved_collection.name}")
        print(f"   Collection dimension: {retrieved_collection.dimension}")
        
        # Test 3: has_collection - should return False for non-existent collection
        non_existent_name = "test_collection_nonexistent_" + str(int(time.time()))
        assert not client.has_collection(non_existent_name)
        print(f"\n✅ has_collection correctly returns False for non-existent collection")
        
        # Test 4: has_collection - should return True for existing collection
        assert client.has_collection(test_collection_name)
        print(f"\n✅ has_collection correctly returns True for existing collection")
        
        # Test 5: get_or_create_collection - should get existing collection
        existing_collection = client.get_or_create_collection(
            name=test_collection_name,
            configuration=config,
            embedding_function=None
        )
        assert existing_collection is not None
        assert existing_collection.name == test_collection_name
        assert existing_collection.dimension == actual_dimension
        print(f"\n✅ get_or_create_collection successfully retrieved existing collection")
        
        # Test 6: get_or_create_collection - should create new collection
        test_collection_name_mgmt = "test_collection_mgmt_" + str(int(time.time()))
        new_collection = client.get_or_create_collection(
            name=test_collection_name_mgmt,
            configuration=config,
            embedding_function=None
        )
        assert new_collection is not None
        assert new_collection.name == test_collection_name_mgmt
        # Use actual dimension for new collection too
        assert new_collection.dimension == actual_dimension
        print(f"\n✅ get_or_create_collection successfully created collection '{test_collection_name_mgmt}'")
        
        # Test 7: list_collections - should include our collections
        collections = client.list_collections()
        assert isinstance(collections, list)
        collection_names = [c.name for c in collections]
        assert test_collection_name in collection_names
        assert test_collection_name_mgmt in collection_names
        print(f"\n✅ list_collections successfully listed collections: {len(collections)} found")
        print(f"   Collection names: {collection_names}")
        
        # Test 8: delete_collection - should delete the collection
        client.delete_collection(test_collection_name_mgmt)
        assert not client.has_collection(test_collection_name_mgmt)
        print(f"\n✅ delete_collection successfully deleted collection '{test_collection_name_mgmt}'")
        
        # Test 9: delete_collection - should raise error for non-existent collection
        try:
            client.delete_collection(test_collection_name_mgmt)
            pytest.fail("delete_collection should raise ValueError for non-existent collection")
        except ValueError as e:
            assert "does not exist" in str(e)
            print(f"\n✅ delete_collection correctly raises ValueError for non-existent collection")
        
        # Test 10: get_or_create_collection without configuration - should use default configuration
        test_collection_name_default = "test_collection_default_" + str(int(time.time()))
        default_collection = client.get_or_create_collection(name=test_collection_name_default)
        assert default_collection is not None
        assert default_collection.name == test_collection_name_default
        # Default dimension is 384 (matches default embedding function)
        assert default_collection.dimension == 384
        print(f"\n✅ get_or_create_collection successfully created collection with default configuration")
        
        # Test 11: count_collection - count the number of collections
        collection_count = client.count_collection()
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
        import uuid
        import random
        random.seed(42)  # For reproducibility
        test_ids = [str(uuid.uuid4()) for _ in range(3)]
        # Generate embeddings matching the collection's dimension
        embeddings = [[random.random() for _ in range(collection.dimension)] for _ in range(3)]
        collection.add(
            ids=test_ids,
            embeddings=embeddings,
            documents=[f"Test document {i}" for i in range(3)],
            metadatas=[{"index": i} for i in range(3)]
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
        
        # Clean up: delete the test collection table
        try:
            client._server.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"   Cleaned up test table: {table_name}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup test table: {cleanup_error}")
    
    def test_create_embedded_client(self):
        """Test creating embedded client (lazy loading) and executing queries"""
        # Check if seekdb package is available and properly configured
        try:
            import sys
            project_root_str = str(project_root)
            if project_root_str in sys.path:
                sys.path.remove(project_root_str)
            import pylibseekdb as seekdb
            if not hasattr(seekdb, 'open') and not hasattr(seekdb, '_initialize_module'):
                pytest.fail(
                    "❌ seekdb embedded package is not properly installed!\n"
                    "The 'seekdb' module exists but lacks required methods.\n"
                    "Required: 'open' method or '_initialize_module' method\n\n"
                    "Solution: Please install the seekdb embedded package from correct source:\n"
                    "  pip install seekdb\n"
                    "Or contact the seekdb package maintainer for installation guide."
                )
        except ImportError:
            pytest.fail(
                "❌ seekdb embedded package is not installed!\n"
                "The 'seekdb' module cannot be imported.\n\n"
                "Solution: Please install the seekdb embedded package:\n"
                "  pip install seekdb\n"
                "Or contact the seekdb package maintainer for installation guide."
            )
        finally:
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
        
        # Create client (returns _ClientProxy)
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        # Verify client type and properties
        assert client is not None
        # Client now returns a proxy
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.SeekdbEmbeddedClient)
        assert client._server.mode == "SeekdbEmbeddedClient"
        assert client._server.database == SEEKDB_DATABASE
        
        # Should not be connected at this point (lazy loading)
        assert not client._server.is_connected()
        
        # Execute query through proxy (first use, triggers connection)
        result = client._server.execute("SELECT 1")
        assert result is not None
        assert len(result) > 0
        
        # Should be connected now
        assert client._server.is_connected()
        
        print(f"\n✅ Embedded client created and connected successfully: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
        print(f"   Query result: {result[0]}")
        
        # Test all collection management interfaces
        self._test_collection_management(client)
        
        # Automatic cleanup (via __del__)
    
    def test_create_server_client(self):
        """Test creating server client (lazy loading) and executing queries"""
        # Create client (returns _ClientProxy)
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",  # Default tenant for seekdb Server
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Verify client type and properties
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        assert client._server.mode == "RemoteServerClient"
        assert client._server.host == SERVER_HOST
        assert client._server.port == SERVER_PORT
        assert client._server.database == SERVER_DATABASE
        assert client._server.user == SERVER_USER
        
        # Should not be connected at this point (lazy loading)
        assert not client._server.is_connected()
        
        # Execute query through proxy (first use, triggers connection)
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
            assert len(result) > 0
            assert result[0]['test'] == 1
            
            # Should be connected now
            assert client._server.is_connected()
            
            print(f"\n✅ Server client created and connected successfully: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
            print(f"   Query result: {result[0]}")
            
            # Test all collection management interfaces
            self._test_collection_management(client)
            
        except Exception as e:
            pytest.fail(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}\n"
                       f"Hint: Please ensure seekdb Server is running on port {SERVER_PORT}")
        
        # Automatic cleanup (via __del__)
    
    def test_create_oceanbase_client(self):
        """Test creating OceanBase client (lazy loading) and executing queries"""
        # Create client (returns _ClientProxy)
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Verify client type and properties
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        assert client._server.mode == "RemoteServerClient"
        assert client._server.host == OB_HOST
        assert client._server.port == OB_PORT
        assert client._server.tenant == OB_TENANT
        assert client._server.database == OB_DATABASE
        assert client._server.user == OB_USER
        assert client._server.full_user == f"{OB_USER}@{OB_TENANT}"
        
        # Should not be connected at this point (lazy loading)
        assert not client._server.is_connected()
        
        # Execute query through proxy (first use, triggers connection)
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
            assert len(result) > 0
            assert result[0]['test'] == 1
            
            # Should be connected now
            assert client._server.is_connected()
            
            print(f"\n✅ OceanBase client created and connected successfully: {client._server.full_user}@{OB_HOST}:{OB_PORT}/{OB_DATABASE}")
            print(f"   Query result: {result[0]}")
            
            # Test all collection management interfaces
            self._test_collection_management(client)
            
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}\n"
                       f"Hint: Please ensure OceanBase is running and tenant '{OB_TENANT}' is created")
        
        # Automatic cleanup (via __del__)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("pyseekdb - Client Creation and Connection Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])
