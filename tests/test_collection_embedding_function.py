"""
Test collection creation with embedding function - testing create_collection, 
get_or_create_collection, and get_collection interfaces with embedding function handling
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
from pathlib import Path
from typing import List, Union

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb
from pyseekdb import DefaultEmbeddingFunction, HNSWConfiguration


# ==================== Simple 3D Embedding Function for Testing ====================
class Simple3DEmbeddingFunction:
    """Simple embedding function that returns 3-dimensional vectors for testing"""
    
    def __init__(self):
        self.dimension = 3
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """Convert documents to 3D embeddings (simple hash-based)"""
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for doc in input:
            # Simple hash-based 3D embedding for testing
            hash_val = hash(doc) % 1000
            embedding = [
                float((hash_val % 10) / 10.0),
                float(((hash_val // 10) % 10) / 10.0),
                float(((hash_val // 100) % 10) / 10.0)
            ]
            embeddings.append(embedding)
        
        return embeddings


# ==================== Simple 128D Embedding Function for Testing ====================
class Simple128DEmbeddingFunction:
    """Simple embedding function that returns 128-dimensional vectors for testing"""
    
    def __init__(self):
        self.dimension = 128
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """Convert documents to 128D embeddings"""
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for doc in input:
            # Simple hash-based 128D embedding for testing
            hash_val = hash(doc)
            embedding = [float((hash_val + i) % 100) / 100.0 for i in range(128)]
            embeddings.append(embedding)
        
        return embeddings


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


class TestCollectionEmbeddingFunction:
    """Test collection creation with embedding function handling"""
    
    def _test_create_collection_default_embedding_function(self, client):
        """Test create_collection with default embedding function (not provided)"""
        collection_name = f"test_default_ef_{int(time.time())}"
        print(f"\n✅ Testing create_collection with default embedding function")
        
        # Test: Not providing embedding_function should use DefaultEmbeddingFunction
        collection = client.create_collection(name=collection_name)
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, DefaultEmbeddingFunction)
        # Default embedding function produces 384-dim vectors
        assert collection.dimension == 384
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_create_collection_explicit_none(self, client):
        """Test create_collection with embedding_function=None"""
        collection_name = f"test_explicit_none_{int(time.time())}"
        print(f"\n✅ Testing create_collection with embedding_function=None")
        
        # Test: Explicitly set embedding_function=None, must provide configuration
        config = HNSWConfiguration(dimension=128, distance='cosine')
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is None
        assert collection.dimension == 128
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_create_collection_custom_embedding_function(self, client):
        """Test create_collection with custom embedding function"""
        collection_name = f"test_custom_ef_{int(time.time())}"
        print(f"\n✅ Testing create_collection with custom embedding function")
        
        # Test: Custom embedding function, dimension calculated via __call__("seekdb")
        custom_ef = Simple3DEmbeddingFunction()
        config = HNSWConfiguration(dimension=3, distance='l2')
        
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=custom_ef
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert collection.embedding_function == custom_ef
        assert collection.dimension == 3
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_create_collection_dimension_mismatch(self, client):
        """Test create_collection with dimension mismatch should raise error"""
        collection_name = f"test_dim_mismatch_{int(time.time())}"
        print(f"\n✅ Testing create_collection with dimension mismatch (should fail)")
        
        # Test: Configuration dimension doesn't match embedding function dimension
        custom_ef = Simple3DEmbeddingFunction()
        config = HNSWConfiguration(dimension=128, distance='cosine')  # Mismatch: 3 vs 128
        
        with pytest.raises(ValueError) as exc_info:
            client.create_collection(
                name=collection_name,
                configuration=config,
                embedding_function=custom_ef
            )
        
        assert "doesn't match" in str(exc_info.value).lower() or "dimension" in str(exc_info.value).lower()
        print(f"   Correctly raised ValueError: {exc_info.value}")
    
    def _test_create_collection_configuration_none_with_ef(self, client):
        """Test create_collection with configuration=None and embedding_function provided"""
        collection_name = f"test_config_none_with_ef_{int(time.time())}"
        print(f"\n✅ Testing create_collection with configuration=None and embedding_function provided")
        
        # Test: configuration=None, but embedding_function is provided, should calculate dimension
        custom_ef = Simple3DEmbeddingFunction()
        collection = client.create_collection(
            name=collection_name,
            configuration=None,
            embedding_function=custom_ef
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert collection.embedding_function == custom_ef
        assert collection.dimension == 3  # Should use calculated dimension
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_create_collection_both_none_error(self, client):
        """Test create_collection with embedding_function=None and configuration=None should raise error"""
        collection_name = f"test_both_none_{int(time.time())}"
        print(f"\n✅ Testing create_collection with both None (should fail)")
        
        # Test: Both embedding_function and configuration are None
        with pytest.raises(ValueError) as exc_info:
            client.create_collection(
                name=collection_name,
                configuration=None,
                embedding_function=None
            )
        
        assert "cannot determine dimension" in str(exc_info.value).lower() or "none" in str(exc_info.value).lower()
        print(f"   Correctly raised ValueError: {exc_info.value}")
    
    def _test_get_collection_default_embedding_function(self, client):
        """Test get_collection with default embedding function"""
        collection_name = f"test_get_default_ef_{int(time.time())}"
        print(f"\n✅ Testing get_collection with default embedding function")
        
        # First create a collection
        config = HNSWConfiguration(dimension=128, distance='cosine')
        created_collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None
        )
        
        # Then get it without providing embedding_function (should use default)
        retrieved_collection = client.get_collection(name=collection_name)
        
        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.dimension == 128
        # Should have default embedding function
        assert retrieved_collection.embedding_function is not None
        assert isinstance(retrieved_collection.embedding_function, DefaultEmbeddingFunction)
        print(f"   Collection dimension: {retrieved_collection.dimension}")
        print(f"   Embedding function: {retrieved_collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_get_collection_explicit_none(self, client):
        """Test get_collection with embedding_function=None"""
        collection_name = f"test_get_explicit_none_{int(time.time())}"
        print(f"\n✅ Testing get_collection with embedding_function=None")
        
        # First create a collection
        config = HNSWConfiguration(dimension=128, distance='cosine')
        created_collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None
        )
        
        # Then get it with embedding_function=None
        retrieved_collection = client.get_collection(
            name=collection_name,
            embedding_function=None
        )
        
        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.dimension == 128
        assert retrieved_collection.embedding_function is None
        print(f"   Collection dimension: {retrieved_collection.dimension}")
        print(f"   Embedding function: {retrieved_collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_get_or_create_collection_create_new(self, client):
        """Test get_or_create_collection creating new collection"""
        collection_name = f"test_get_or_create_new_{int(time.time())}"
        print(f"\n✅ Testing get_or_create_collection (create new)")
        
        # Test: Collection doesn't exist, should create with default embedding function
        collection = client.get_or_create_collection(name=collection_name)
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, DefaultEmbeddingFunction)
        assert collection.dimension == 384
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_get_or_create_collection_get_existing(self, client):
        """Test get_or_create_collection getting existing collection"""
        collection_name = f"test_get_or_create_existing_{int(time.time())}"
        print(f"\n✅ Testing get_or_create_collection (get existing)")
        
        # First create a collection
        config = HNSWConfiguration(dimension=128, distance='cosine')
        created_collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None
        )
        
        # Then get_or_create it
        retrieved_collection = client.get_or_create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None
        )
        
        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.dimension == 128
        print(f"   Collection dimension: {retrieved_collection.dimension}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_get_or_create_collection_custom_embedding_function(self, client):
        """Test get_or_create_collection with custom embedding function"""
        collection_name = f"test_get_or_create_custom_ef_{int(time.time())}"
        print(f"\n✅ Testing get_or_create_collection with custom embedding function")
        
        # Test: Create with custom embedding function
        custom_ef = Simple3DEmbeddingFunction()
        config = HNSWConfiguration(dimension=3, distance='l2')
        
        collection = client.get_or_create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=custom_ef
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert collection.embedding_function == custom_ef
        assert collection.dimension == 3
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        # Cleanup
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    def _test_get_or_create_collection_both_none_error(self, client):
        """Test get_or_create_collection with both None should raise error when creating"""
        collection_name = f"test_get_or_create_both_none_{int(time.time())}"
        print(f"\n✅ Testing get_or_create_collection with both None (should fail when creating)")
        
        # Test: Both None when creating new collection
        with pytest.raises(ValueError) as exc_info:
            client.get_or_create_collection(
                name=collection_name,
                configuration=None,
                embedding_function=None
            )
        
        assert "cannot determine dimension" in str(exc_info.value).lower() or "none" in str(exc_info.value).lower()
        print(f"   Correctly raised ValueError: {exc_info.value}")
    
    def test_embedded_client(self):
        """Test with embedded client"""
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("SeekDB embedded package is not installed")
        
        
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        
        # Run all test methods
        self._test_create_collection_default_embedding_function(client)
        self._test_create_collection_explicit_none(client)
        self._test_create_collection_custom_embedding_function(client)
        self._test_create_collection_dimension_mismatch(client)
        self._test_create_collection_configuration_none_with_ef(client)
        self._test_create_collection_both_none_error(client)
        self._test_get_collection_default_embedding_function(client)
        self._test_get_collection_explicit_none(client)
        self._test_get_or_create_collection_create_new(client)
        self._test_get_or_create_collection_get_existing(client)
        self._test_get_or_create_collection_custom_embedding_function(client)
        self._test_get_or_create_collection_both_none_error(client)
    
    def test_server_client(self):
        """Test with server client"""
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        assert client is not None
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Run all test methods
        self._test_create_collection_default_embedding_function(client)
        self._test_create_collection_explicit_none(client)
        self._test_create_collection_custom_embedding_function(client)
        self._test_create_collection_dimension_mismatch(client)
        self._test_create_collection_configuration_none_with_ef(client)
        self._test_create_collection_both_none_error(client)
        self._test_get_collection_default_embedding_function(client)
        self._test_get_collection_explicit_none(client)
        self._test_get_or_create_collection_create_new(client)
        self._test_get_or_create_collection_get_existing(client)
        self._test_get_or_create_collection_custom_embedding_function(client)
        self._test_get_or_create_collection_both_none_error(client)
    
    def test_oceanbase_client(self):
        """Test with OceanBase client"""
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        assert client is not None
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Run all test methods
        self._test_create_collection_default_embedding_function(client)
        self._test_create_collection_explicit_none(client)
        self._test_create_collection_custom_embedding_function(client)
        self._test_create_collection_dimension_mismatch(client)
        self._test_create_collection_configuration_none_with_ef(client)
        self._test_create_collection_both_none_error(client)
        self._test_get_collection_default_embedding_function(client)
        self._test_get_collection_explicit_none(client)
        self._test_get_or_create_collection_create_new(client)
        self._test_get_or_create_collection_get_existing(client)
        self._test_get_or_create_collection_custom_embedding_function(client)
        self._test_get_or_create_collection_both_none_error(client)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SeekDBClient - Collection Embedding Function Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

