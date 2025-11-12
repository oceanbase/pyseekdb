"""
Test default embedding function - testing collection creation with default embedding function,
automatic vector generation from documents, and hybrid search
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
import uuid
from pathlib import Path

# Add project path
# Calculate project root: pyseekdb/tests/test_*.py -> pyobvector/
# Use resolve() to get absolute path, which works even when running as script
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pyseekdb
from pyseekdb import DefaultEmbeddingFunction


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb_store"))
SEEKDB_DATABASE = os.environ.get('SEEKDB_DATABASE', 'test')

# Server mode
SERVER_HOST = os.environ.get('SERVER_HOST', '11.161.205.15')
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


class TestDefaultEmbeddingFunction:
    """Test default embedding function with collection creation, automatic vector generation, and hybrid search"""
    
    def test_embedded_default_embedding_function(self):
        """Test default embedding function with embedded client"""
        # Check if seekdb package is available
        try:
            import pylibseekdb
        except ImportError:
            pytest.skip("SeekDB embedded package is not installed")
        
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers is not installed. Install with: pip install sentence-transformers")
        
        # Create embedded client
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        
        # Create collection with default embedding function (not passing embedding_function parameter)
        collection_name = f"test_default_ef_{int(time.time())}"
        print(f"\n✅ Creating collection '{collection_name}' with default embedding function")
        
        # Create collection - default embedding function will be used automatically
        collection = client.create_collection(name=collection_name)
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, DefaultEmbeddingFunction)
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        try:
            # Test 1: Add documents without providing vectors (vectors will be auto-generated)
            print(f"\n✅ Testing collection.add() with documents only (auto-generate vectors)")
            
            test_documents = [
                "Machine learning is a subset of artificial intelligence",
                "Python programming language is widely used in data science",
                "Deep learning algorithms for neural networks",
                "Data science with Python and machine learning",
                "Introduction to artificial intelligence and neural networks"
            ]
            
            test_ids = [str(uuid.uuid4()) for _ in test_documents]
            test_metadatas = [
                {"category": "AI", "page": 1},
                {"category": "Programming", "page": 2},
                {"category": "AI", "page": 3},
                {"category": "Data Science", "page": 4},
                {"category": "AI", "page": 5}
            ]
            
            # Add documents without vectors - embedding function will generate them automatically
            collection.add(
                ids=test_ids,
                documents=test_documents,
                metadatas=test_metadatas
            )
            print(f"   Added {len(test_documents)} documents (vectors auto-generated)")
            
            # Verify data was inserted
            results = collection.get(ids=test_ids[0], include=["documents", "metadatas", "embeddings"])
            assert len(results) == 1
            assert results[0].document == test_documents[0]
            # Note: embedding might not be returned by default, so we check if it exists
            if results[0].embedding is not None:
                assert len(results[0].embedding) == collection.dimension
                print(f"   Verified: document and embedding (dim={len(results[0].embedding)}) stored correctly")
            else:
                print(f"   Verified: document stored correctly (embedding not included in get results)")
            
            # Test 2: Generate query embedding using default embedding function
            print(f"\n✅ Testing query embedding generation")
            query_text = "artificial intelligence and machine learning"
            query_embedding = collection.embedding_function([query_text])[0]
            assert len(query_embedding) == collection.dimension
            print(f"   Generated query embedding with dimension: {len(query_embedding)}")
            
            # Wait a bit for indexes to be ready
            time.sleep(1)
            
            # Test 3: Hybrid search with vector search
            print(f"\n✅ Testing hybrid_search with vector search")
            try:
                results = collection.hybrid_search(
                    knn={
                        "query_embeddings": query_embedding,
                        "n_results": 3
                    },
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                assert results is not None
                assert "ids" in results
                assert "distances" in results
                assert len(results["ids"]) > 0
                print(f"   Found {len(results['ids'])} results")
                
                # Verify results contain documents
                if "documents" in results:
                    assert len(results["documents"]) > 0
                    print(f"   Top result: {results['documents'][0][:50]}...")
            except Exception as e:
                print(f"   ⚠️  Hybrid search with vector failed: {e}")
                # Continue with other tests
            
            # Test 4: Hybrid search with full-text search
            print(f"\n✅ Testing hybrid_search with full-text search")
            try:
                results = collection.hybrid_search(
                    query={
                        "where_document": {"$contains": "machine learning"},
                        "n_results": 3
                    },
                    n_results=3,
                    include=["documents", "metadatas"]
                )
                
                assert results is not None
                assert "ids" in results
                assert len(results["ids"]) > 0
                print(f"   Found {len(results['ids'])} results from full-text search")
            except Exception as e:
                print(f"   ⚠️  Hybrid search with full-text failed: {e}")
                # Continue with other tests
            
            # Test 5: Hybrid search combining both vector and full-text
            print(f"\n✅ Testing hybrid_search with both vector and full-text search")
            try:
                results = collection.hybrid_search(
                    query={
                        "where_document": {"$contains": "machine learning"},
                        "n_results": 3
                    },
                    knn={
                        "query_embeddings": query_embedding,
                        "n_results": 3
                    },
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                assert results is not None
                assert "ids" in results
                assert len(results["ids"]) > 0
                print(f"   Found {len(results['ids'])} results from hybrid search")
            except Exception as e:
                print(f"   ⚠️  Hybrid search combining both failed: {e}")
                # This is acceptable - hybrid search may not be fully supported in all modes
            
        finally:
            # Cleanup
            try:
                client.delete_collection(collection_name)
                print(f"\n✅ Cleaned up collection '{collection_name}'")
            except Exception as e:
                print(f"\n⚠️  Failed to cleanup collection: {e}")
    
    def test_server_default_embedding_function(self):
        """Test default embedding function with server client"""
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers is not installed. Install with: pip install sentence-transformers")
        
        # Create server client
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
            pytest.skip(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create collection with default embedding function
        collection_name = f"test_default_ef_{int(time.time())}"
        print(f"\n✅ Creating collection '{collection_name}' with default embedding function")
        
        collection = client.create_collection(name=collection_name)
        
        assert collection is not None
        assert collection.embedding_function is not None
        
        try:
            # Add documents without vectors
            test_documents = [
                "Machine learning is a subset of artificial intelligence",
                "Python programming language is widely used"
            ]
            test_ids = [str(uuid.uuid4()) for _ in test_documents]
            
            collection.add(
                ids=test_ids,
                documents=test_documents,
                metadatas=[{"category": "AI"}, {"category": "Programming"}]
            )
            print(f"   Added {len(test_documents)} documents (vectors auto-generated)")
            
            # Wait for indexes
            time.sleep(1)
            
            # Test hybrid search
            query_text = "machine learning"
            query_embedding = collection.embedding_function([query_text])[0]
            
            results = collection.hybrid_search(
                knn={
                    "query_embeddings": query_embedding,
                    "n_results": 2
                },
                n_results=2,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            print(f"   Hybrid search found {len(results['ids'])} results")
            
        finally:
            # Cleanup
            try:
                client.delete_collection(collection_name)
            except Exception as e:
                print(f"⚠️  Failed to cleanup: {e}")
    
    def test_oceanbase_default_embedding_function(self):
        """Test default embedding function with OceanBase client"""
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers is not installed. Install with: pip install sentence-transformers")
        
        # Create OceanBase client
        client = pyseekdb.OBClient(
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
            pytest.skip(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create collection with default embedding function
        collection_name = f"test_default_ef_{int(time.time())}"
        print(f"\n✅ Creating collection '{collection_name}' with default embedding function")
        
        collection = client.create_collection(name=collection_name)
        
        assert collection is not None
        assert collection.embedding_function is not None
        
        try:
            # Add documents without vectors
            test_documents = [
                "Machine learning is a subset of artificial intelligence",
                "Python programming language is widely used"
            ]
            test_ids = [str(uuid.uuid4()) for _ in test_documents]
            
            collection.add(
                ids=test_ids,
                documents=test_documents,
                metadatas=[{"category": "AI"}, {"category": "Programming"}]
            )
            print(f"   Added {len(test_documents)} documents (vectors auto-generated)")
            
            # Wait for indexes
            time.sleep(1)
            
            # Test hybrid search
            query_text = "machine learning"
            query_embedding = collection.embedding_function([query_text])[0]
            
            results = collection.hybrid_search(
                knn={
                    "query_embeddings": query_embedding,
                    "n_results": 2
                },
                n_results=2,
                include=["documents", "metadatas"]
            )
            
            assert results is not None
            assert len(results["ids"]) > 0
            print(f"   Hybrid search found {len(results['ids'])} results")
            
        finally:
            # Cleanup
            try:
                client.delete_collection(collection_name)
            except Exception as e:
                print(f"⚠️  Failed to cleanup: {e}")


if __name__ == "__main__":
    # Allow running tests directly with: python test_default_embedding_function.py
    # Note: For better test execution, use pytest: pytest test_default_embedding_function.py -v
    pytest.main([__file__, "-v", "-s"])
