"""
Collection hybrid search tests - testing collection.hybrid_search() interface for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
import json
import uuid
from pathlib import Path
from typing import List

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


class TestCollectionHybridSearch:
    """Test collection.hybrid_search() interface for all three modes"""
    
    def _create_test_collection(self, client, collection_name: str, dimension: int = None):
        """Helper method to create a test collection"""
        # Use client.create_collection to create the collection
        # If dimension is not provided, use default embedding function (384 dim)
        # If dimension is provided, set embedding_function=None to use the specified dimension
        from pyseekdb import HNSWConfiguration
        if dimension is not None:
            # Use specified dimension without embedding function
            config = HNSWConfiguration(dimension=dimension, distance='l2')
            collection = client.create_collection(
                name=collection_name,
                configuration=config,
                embedding_function=None
            )
        else:
            # Use default configuration and embedding function (auto-calculates dimension)
            collection = client.create_collection(
                name=collection_name
            )
        # Return both collection and actual dimension
        return collection, collection.dimension
    
    def _generate_query_vector(self, dimension: int, base_vector: List[float] = [1.0, 2.0, 3.0]) -> List[float]:
        """Generate a query vector with the correct dimension
        
        Args:
            dimension: Target dimension
            base_vector: Base vector pattern (default: [1.0, 2.0, 3.0])
            
        Returns:
            Vector with the specified dimension
        """
        if dimension <= len(base_vector):
            return base_vector[:dimension]
        else:
            # Extend if dimension is larger (repeat pattern)
            extended = base_vector * ((dimension // len(base_vector)) + 1)
            return extended[:dimension]
    
    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        """Helper method to insert test data via SQL
        
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
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [2.2, 3.2, 4.2],
            [1.4, 2.4, 3.4]
        ]
        
        # Insert test data with vectors, documents, and metadata
        # Data designed for hybrid search testing
        test_data = [
            {
                "document": "Machine learning is a subset of artificial intelligence",
                "base_vector": base_vectors[0],
                "metadata": {"category": "AI", "page": 1, "score": 95, "tag": "ml"}
            },
            {
                "document": "Python programming language is widely used in data science",
                "base_vector": base_vectors[1],
                "metadata": {"category": "Programming", "page": 2, "score": 88, "tag": "python"}
            },
            {
                "document": "Deep learning algorithms for neural networks",
                "base_vector": base_vectors[2],
                "metadata": {"category": "AI", "page": 3, "score": 92, "tag": "ml"}
            },
            {
                "document": "Data science with Python and machine learning",
                "base_vector": base_vectors[3],
                "metadata": {"category": "Data Science", "page": 4, "score": 90, "tag": "python"}
            },
            {
                "document": "Introduction to artificial intelligence and neural networks",
                "base_vector": base_vectors[4],
                "metadata": {"category": "AI", "page": 5, "score": 85, "tag": "neural"}
            },
            {
                "document": "Advanced machine learning techniques and algorithms",
                "base_vector": base_vectors[5],
                "metadata": {"category": "AI", "page": 6, "score": 93, "tag": "ml"}
            },
            {
                "document": "Python tutorial for beginners in programming",
                "base_vector": base_vectors[6],
                "metadata": {"category": "Programming", "page": 7, "score": 87, "tag": "python"}
            },
            {
                "document": "Natural language processing with machine learning",
                "base_vector": base_vectors[7],
                "metadata": {"category": "AI", "page": 8, "score": 91, "tag": "nlp"}
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
    
    def _cleanup_collection(self, client, collection_name: str):
        """Helper method to cleanup test collection"""
        table_name = f"c$v1${collection_name}"
        try:
            client._server.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"   Cleaned up test table: {table_name}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup test table: {cleanup_error}")
    
    def test_oceanbase_hybrid_search_full_text_only(self):
        """Test hybrid_search with only full-text search (query)"""
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
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 1: Full-text search only
            print(f"\n✅ Testing hybrid_search with full-text search only")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine learning"
                    }
                },
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Verify results contain "machine learning"
            for doc in results["documents"][0]:
                if doc:
                    assert "machine" in doc.lower() or "learning" in doc.lower()
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_hybrid_search_vector_only(self):
        """Test hybrid_search with only vector search (knn)"""
        # Create OceanBase client
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 2: Vector search only
            print(f"\n✅ Testing hybrid_search with vector search only")
            results = collection.hybrid_search(
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "n_results": 5
                },
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
            
            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Verify distances are reasonable
            # Note: APPROXIMATE ordering may not be perfectly sorted, so we only check
            # that distances are non-negative and reasonable
            distances = results["distances"][0]
            assert len(distances) > 0
            # All distances should be non-negative
            for dist in distances:
                assert dist >= 0, f"Distance should be non-negative, got {dist}"
            # At least one distance should be relatively small (close match)
            min_distance = min(distances)
            assert min_distance < 10.0, f"At least one distance should be reasonable, got min={min_distance}"
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_hybrid_search_combined(self):
        """Test hybrid_search with both full-text and vector search"""
        # Create OceanBase client
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 3: Combined full-text and vector search
            print(f"\n✅ Testing hybrid_search with both full-text and vector search")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine learning"
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "n_results": 10
                },
                rank={
                    "rrf": {
                        "rank_window_size": 60,
                        "rank_constant": 60
                    }
                },
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
            
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results after RRF ranking")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_hybrid_search_with_metadata_filter(self):
        """Test hybrid_search with metadata filters"""
        # Create OceanBase client
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 4: Hybrid search with metadata filter
            print(f"\n✅ Testing hybrid_search with metadata filter")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine"
                    },
                    "where": {
                        "$and": [
                            {"category": {"$eq": "AI"}},
                            {"page": {"$gte": 1}},
                            {"page": {"$lte": 5}}
                        ]
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "where": {
                        "$and": [
                            {"category": {"$eq": "AI"}},
                            {"score": {"$gte": 90}}
                        ]
                    },
                    "n_results": 10
                },
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with metadata filters")
            
            # Verify metadata filters are applied
            # Note: In hybrid search with RRF ranking, results may include records from both
            # full-text and vector search, so we check that all results meet at least one set of filters
            for metadata in results["metadatas"][0]:
                if metadata:
                    # Results should have category "AI" (common to both query and knn filters)
                    assert metadata.get("category") == "AI"
                    # Page filter may not be strictly applied in hybrid search results
                    # due to RRF ranking combining results from both queries
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_hybrid_search_with_logical_operators(self):
        """Test hybrid_search with logical operators in metadata filters"""
        # Create OceanBase client
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 5: Hybrid search with logical operators ($or, $in)
            print(f"\n✅ Testing hybrid_search with logical operators")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$and": [
                            {"$contains": "machine"},
                            {"$contains": "learning"}
                        ]
                    },
                    "where": {
                        "$or": [
                            {"tag": {"$eq": "ml"}},
                            {"tag": {"$eq": "python"}}
                        ]
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "where": {
                        "tag": {"$in": ["ml", "python"]}
                    },
                    "n_results": 10
                },
                rank={"rrf": {}},
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with logical operators")
            
            # Verify logical operators are applied
            for metadata in results["metadatas"][0]:
                if metadata and "tag" in metadata:
                    assert metadata["tag"] in ["ml", "python"]
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_seekdb_server_hybrid_search_full_text_only(self):
        """Test hybrid_search with only full-text search (query) using SeekdbServer"""
        # Create SeekdbServer client
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
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 1: Full-text search only
            print(f"\n✅ Testing hybrid_search with full-text search only (SeekdbServer)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine learning"
                    }
                },
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Verify results contain "machine learning"
            for doc in results["documents"][0]:
                if doc:
                    assert "machine" in doc.lower() or "learning" in doc.lower()
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_seekdb_server_hybrid_search_combined(self):
        """Test hybrid_search with both full-text and vector search using SeekdbServer"""
        # Create SeekdbServer client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",  # Default tenant for SeekDB Server
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Combined full-text and vector search
            print(f"\n✅ Testing hybrid_search with both full-text and vector search (SeekdbServer)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine learning"
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "n_results": 10
                },
                rank={
                    "rrf": {
                        "rank_window_size": 60,
                        "rank_constant": 60
                    }
                },
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
            
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results after RRF ranking")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_seekdb_server_hybrid_search_vector_only(self):
        """Test hybrid_search with only vector search (knn) using SeekdbServer"""
        # Create SeekdbServer client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Vector search only
            print(f"\n✅ Testing hybrid_search with vector search only (SeekdbServer)")
            results = collection.hybrid_search(
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "n_results": 5
                },
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
            
            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results")
            
            # Verify distances are reasonable
            distances = results["distances"][0]
            assert len(distances) > 0
            for dist in distances:
                assert dist >= 0, f"Distance should be non-negative, got {dist}"
            min_distance = min(distances)
            assert min_distance < 10.0, f"At least one distance should be reasonable, got min={min_distance}"
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_seekdb_server_hybrid_search_with_metadata_filter(self):
        """Test hybrid_search with metadata filters using SeekdbServer"""
        # Create SeekdbServer client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Hybrid search with metadata filter (simplified equality)
            print(f"\n✅ Testing hybrid_search with metadata filter (SeekdbServer)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine"
                    },
                    "where": {
                        "$and": [
                            {"category": "AI"},
                            {"page": {"$gte": 1}},
                            {"page": {"$lte": 5}}
                        ]
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "where": {
                        "$and": [
                            {"category": "AI"},
                            {"score": {"$gte": 90}}
                        ]
                    },
                    "n_results": 10
                },
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with metadata filters")
            
            # Verify metadata filters are applied
            for metadata in results["metadatas"][0]:
                if metadata:
                    assert metadata.get("category") == "AI"
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_seekdb_server_hybrid_search_with_logical_operators(self):
        """Test hybrid_search with logical operators in metadata filters using SeekdbServer"""
        # Create SeekdbServer client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Hybrid search with logical operators
            print(f"\n✅ Testing hybrid_search with logical operators (SeekdbServer)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$and": [
                            {"$contains": "machine"},
                            {"$contains": "learning"}
                        ]
                    },
                    "where": {
                        "$or": [
                            {"tag": {"$eq": "ml"}},
                            {"tag": {"$eq": "python"}}
                        ]
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "where": {
                        "tag": {"$in": ["ml", "python"]}
                    },
                    "n_results": 10
                },
                rank={"rrf": {}},
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with logical operators")
            
            # Verify logical operators are applied
            for metadata in results["metadatas"][0]:
                if metadata and "tag" in metadata:
                    assert metadata["tag"] in ["ml", "python"]
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)


    def test_embedded_hybrid_search_full_text_only(self):
        """Test hybrid_search with only full-text search (query) using SeekdbEmbedded"""
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
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 1: Full-text search only
            print(f"\n✅ Testing hybrid_search with full-text search only (SeekdbEmbedded)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine learning"
                    }
                },
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")
            
            # Verify results contain "machine learning"
            for doc in results["documents"][0]:
                if doc:
                    assert "machine" in doc.lower() or "learning" in doc.lower()
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_embedded_hybrid_search_vector_only(self):
        """Test hybrid_search with only vector search (knn) using SeekdbEmbedded"""
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
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Vector search only
            print(f"\n✅ Testing hybrid_search with vector search only (SeekdbEmbedded)")
            results = collection.hybrid_search(
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "n_results": 5
                },
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
            
            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            print(f"   Found {len(results['ids'])} results")
            
            # Verify distances are reasonable
            distances = results["distances"][0]
            assert len(distances) > 0
            for dist in distances:
                assert dist >= 0, f"Distance should be non-negative, got {dist}"
            min_distance = min(distances)
            assert min_distance < 10.0, f"At least one distance should be reasonable, got min={min_distance}"
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_embedded_hybrid_search_combined(self):
        """Test hybrid_search with both full-text and vector search using SeekdbEmbedded"""
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
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Combined full-text and vector search
            print(f"\n✅ Testing hybrid_search with both full-text and vector search (SeekdbEmbedded)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine learning"
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "n_results": 10
                },
                rank={
                    "rrf": {
                        "rank_window_size": 60,
                        "rank_constant": 60
                    }
                },
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
            
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results after RRF ranking")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_embedded_hybrid_search_with_metadata_filter(self):
        """Test hybrid_search with metadata filters using SeekdbEmbedded"""
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
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Hybrid search with metadata filter
            print(f"\n✅ Testing hybrid_search with metadata filter (SeekdbEmbedded)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$contains": "machine"
                    },
                    "where": {
                        "$and": [
                            {"category": {"$eq": "AI"}},
                            {"page": {"$gte": 1}},
                            {"page": {"$lte": 5}}
                        ]
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "where": {
                        "$and": [
                            {"category": {"$eq": "AI"}},
                            {"score": {"$gte": 90}}
                        ]
                    },
                    "n_results": 10
                },
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with metadata filters")
            
            # Verify metadata filters are applied
            for metadata in results["metadatas"][0]:
                if metadata:
                    assert metadata.get("category") == "AI"
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_embedded_hybrid_search_with_logical_operators(self):
        """Test hybrid_search with logical operators in metadata filters using SeekdbEmbedded"""
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
        
        # Create test collection
        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test: Hybrid search with logical operators
            print(f"\n✅ Testing hybrid_search with logical operators (SeekdbEmbedded)")
            results = collection.hybrid_search(
                query={
                    "where_document": {
                        "$and": [
                            {"$contains": "machine"},
                            {"$contains": "learning"}
                        ]
                    },
                    "where": {
                        "$or": [
                            {"tag": {"$eq": "ml"}},
                            {"tag": {"$eq": "python"}}
                        ]
                    },
                    "n_results": 10
                },
                knn={
                    "query_embeddings": self._generate_query_vector(actual_dimension),
                    "where": {
                        "tag": {"$in": ["ml", "python"]}
                    },
                    "n_results": 10
                },
                rank={"rrf": {}},
                n_results=5,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with logical operators")
            
            # Verify logical operators are applied
            for metadata in results["metadatas"][0]:
                if metadata and "tag" in metadata:
                    assert metadata["tag"] in ["ml", "python"]
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
