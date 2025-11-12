"""
Test user-defined embedding function - testing collection creation with custom embedding functions,
automatic vector generation from documents, and querying

This test demonstrates how to:
1. Create custom embedding functions
2. Use them to create collections
3. Add documents (vectors will be auto-generated using the custom embedding function)
4. Query collections (query vectors will be auto-generated using the same embedding function)

Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
from pathlib import Path
from typing import List, Union

# Add project path
# Calculate project root: pyseekdb/tests/test_*.py -> pyobvector/
# Use resolve() to get absolute path, which works even when running as script
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pyseekdb
from pyseekdb import EmbeddingFunction, HNSWConfiguration

# Type aliases for clarity
Documents = Union[str, List[str]]
Embeddings = List[List[float]]


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', "./seekdb_store")
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


# ==================== Custom Embedding Function Definitions ====================

class SimpleHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A simple custom embedding function that uses hash-based vectorization.
    
    This is a demonstration embedding function. In production, you would use
    a proper embedding model (e.g., sentence-transformers, OpenAI, etc.).
    
    This function creates 128-dimensional vectors by hashing the text.
    """
    
    def __init__(self, dimension: int = 128):
        """
        Initialize the hash-based embedding function.
        
        Args:
            dimension: The dimension of the embedding vectors (default: 128)
        """
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents using hash-based vectorization.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors (List[List[float]])
        """
        print(f"[SimpleHashEmbeddingFunction.__call__] Called with input type: {type(input)}")
        import hashlib
        
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        embeddings = []
        for doc in input:
            # Create hash-based embedding
            # Use MD5 hash and convert to vector
            hash_obj = hashlib.md5(doc.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()
            
            # Convert hash to vector
            # Each pair of hex digits becomes a float value
            vector = []
            for i in range(0, min(len(hash_hex), self._dimension * 2), 2):
                # Convert hex pair to float in range [0, 1]
                hex_pair = hash_hex[i:i+2]
                value = int(hex_pair, 16) / 255.0
                vector.append(value)
            
            # Pad or truncate to exact dimension
            while len(vector) < self._dimension:
                # Use a simple pattern to pad
                vector.append(vector[len(vector) % len(vector)] if vector else 0.0)
            
            vector = vector[:self._dimension]
            embeddings.append(vector)
        
        return embeddings
    
    def __repr__(self) -> str:
        return f"SimpleHashEmbeddingFunction(dimension={self._dimension})"


class SentenceTransformerCustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using sentence-transformers with a specific model.
    
    This is a more realistic example that wraps sentence-transformers.
    You can customize the model, device, and other parameters.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the sentence-transformer embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None
    
    def _ensure_model_loaded(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Get dimension from model by encoding a test string
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                self._dimension = len(test_embedding[0])
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it with: pip install sentence-transformers"
                )
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        self._ensure_model_loaded()
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents using sentence-transformers.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors (List[List[float]])
        """
        print(f"[SentenceTransformerCustomEmbeddingFunction.__call__] Called with input type: {type(input)}")
        self._ensure_model_loaded()
        
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Generate embeddings
        embeddings = self._model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]
    
    def __repr__(self) -> str:
        return f"SentenceTransformerCustomEmbeddingFunction(model_name='{self.model_name}', device='{self.device}')"


# ==================== Test Classes ====================

class TestUserDefinedEmbeddingFunction:
    """Test user-defined embedding functions with collection creation, automatic vector generation, and querying"""
    
    def test_embedded_simple_hash_embedding_function(self):
        """Test simple hash-based embedding function with embedded client"""
        # Check if seekdb package is available
        try:
            import pylibseekdb
        except ImportError:
            pytest.skip("SeekDB embedded package is not installed")
        
        # Create embedded client
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        
        # Create custom embedding function
        custom_ef = SimpleHashEmbeddingFunction(dimension=128)
        assert custom_ef.dimension == 128
        
        # Create collection with custom embedding function
        collection_name = f"test_user_ef_hash_{int(time.time())}"
        print(f"\n✅ Creating collection '{collection_name}' with custom hash embedding function")
        
        config = HNSWConfiguration(dimension=128, distance='cosine')
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=custom_ef
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, SimpleHashEmbeddingFunction)
        assert collection.dimension == 128
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        try:
            # Test 1: Add documents without providing vectors (vectors will be auto-generated)
            print(f"\n✅ Testing collection.add() with documents only (auto-generate vectors)")
            
            test_documents = [
                "Machine learning is a subset of artificial intelligence",
                "Python programming language is widely used in data science",
                "Vector databases enable efficient semantic search",
                "Neural networks are inspired by the structure of the human brain",
                "Natural language processing helps computers understand human language"
            ]
            
            ids = [f"doc_{i+1}" for i in range(len(test_documents))]
            
            collection.add(
                ids=ids,
                documents=test_documents,
                metadatas=[
                    {"category": "AI", "topic": "machine learning"},
                    {"category": "Programming", "topic": "python"},
                    {"category": "Database", "topic": "vector search"},
                    {"category": "AI", "topic": "neural networks"},
                    {"category": "NLP", "topic": "language processing"}
                ]
            )
            
            print(f"   ✅ Added {len(test_documents)} documents with auto-generated vectors")
            
            # Test 2: Query with query_texts (query vector will be auto-generated)
            print(f"\n✅ Testing collection.query() with query_texts (auto-generate query vector)")
            
            query_text = "artificial intelligence and machine learning"
            results = collection.query(
                query_texts=[query_text],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            assert results is not None
            assert len(results) > 0
            print(f"   ✅ Query completed: '{query_text}'")
            print(f"   Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                assert result._id is not None
                assert result.document is not None
                assert result.distance is not None
                print(f"   Result {i}: ID={result._id}, Distance={result.distance:.6f}")
            
            # Test 3: Verify collection count
            count = collection.count()
            assert count == len(test_documents)
            print(f"   ✅ Collection count: {count}")
            
        finally:
            # Clean up
            print(f"\n🧹 Cleaning up collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print("✅ Collection deleted")
    
    def test_embedded_sentence_transformer_embedding_function(self):
        """Test sentence-transformer custom embedding function with embedded client"""
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
        
        # Create custom sentence-transformer embedding function
        st_ef = SentenceTransformerCustomEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",  # 384 dimensions
            device="cpu"
        )
        assert st_ef.dimension == 384
        
        # Create collection with custom embedding function
        collection_name = f"test_user_ef_st_{int(time.time())}"
        print(f"\n✅ Creating collection '{collection_name}' with custom sentence-transformer embedding function")
        
        config = HNSWConfiguration(dimension=st_ef.dimension, distance='cosine')
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=st_ef
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, SentenceTransformerCustomEmbeddingFunction)
        assert collection.dimension == 384
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")
        
        try:
            # Test 1: Add documents without providing vectors (vectors will be auto-generated)
            print(f"\n✅ Testing collection.add() with documents only (auto-generate vectors)")
            
            test_documents = [
                "Machine learning is a subset of artificial intelligence",
                "Python programming language is widely used in data science",
                "Vector databases enable efficient semantic search",
                "Neural networks are inspired by the structure of the human brain",
                "Natural language processing helps computers understand human language"
            ]
            
            ids = [f"doc_{i+1}" for i in range(len(test_documents))]
            
            collection.add(
                ids=ids,
                documents=test_documents,
                metadatas=[
                    {"category": "AI", "topic": "machine learning"},
                    {"category": "Programming", "topic": "python"},
                    {"category": "Database", "topic": "vector search"},
                    {"category": "AI", "topic": "neural networks"},
                    {"category": "NLP", "topic": "language processing"}
                ]
            )
            
            print(f"   ✅ Added {len(test_documents)} documents with auto-generated vectors")
            
            # Test 2: Query with query_texts (query vector will be auto-generated)
            print(f"\n✅ Testing collection.query() with query_texts (auto-generate query vector)")
            
            query_text = "artificial intelligence and machine learning"
            results = collection.query(
                query_texts=[query_text],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            assert results is not None
            assert len(results) > 0
            print(f"   ✅ Query completed: '{query_text}'")
            print(f"   Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                assert result._id is not None
                assert result.document is not None
                assert result.distance is not None
                print(f"   Result {i}: ID={result._id}, Distance={result.distance:.6f}")
            
            # Test 3: Verify collection count
            count = collection.count()
            assert count == len(test_documents)
            print(f"   ✅ Collection count: {count}")
            
        finally:
            # Clean up
            print(f"\n🧹 Cleaning up collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print("✅ Collection deleted")
    
    def test_server_simple_hash_embedding_function(self):
        """Test simple hash-based embedding function with server client"""
        # Create server client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        assert client is not None
        
        # Create custom embedding function
        custom_ef = SimpleHashEmbeddingFunction(dimension=128)
        assert custom_ef.dimension == 128
        
        # Create collection with custom embedding function
        collection_name = f"test_user_ef_hash_{int(time.time())}"
        print(f"\n✅ Creating collection '{collection_name}' with custom hash embedding function")
        
        config = HNSWConfiguration(dimension=128, distance='cosine')
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=custom_ef
        )
        
        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, SimpleHashEmbeddingFunction)
        assert collection.dimension == 128
        
        try:
            # Test: Add documents and query
            test_documents = [
                "Machine learning is a subset of artificial intelligence",
                "Python programming language is widely used in data science"
            ]
            
            ids = [f"doc_{i+1}" for i in range(len(test_documents))]
            
            collection.add(
                ids=ids,
                documents=test_documents
            )
            
            # Query
            results = collection.query(
                query_texts=["artificial intelligence"],
                n_results=2
            )
            
            assert results is not None
            assert len(results) > 0
            
        finally:
            # Clean up
            client.delete_collection(collection_name)

