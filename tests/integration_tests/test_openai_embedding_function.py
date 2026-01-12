"""
Test OpenAIEmbeddingFunction using LiteLLM.

This test is marked to skip in normal test scenarios as it requires:
- litellm package to be installed
- OPENAI_API_KEY environment variable to be set
- External API access to OpenAI

To run this test manually:
    pytest tests/integration_tests/test_openai_embedding_function.py -v -s
    # Or with environment variable:
    OPENAI_API_KEY=your-key pytest tests/integration_tests/test_openai_embedding_function.py -v -s
"""
import pytest
import os
import time

import pyseekdb
from pyseekdb.utils.embedding_functions import OpenAIEmbeddingFunction
from pyseekdb.client.embedding_function import dimension_of


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY environment variable is not set")
class TestOpenAIEmbeddingFunction:
    """Test OpenAIEmbeddingFunction - skipped by default, requires manual execution"""

    def test_litellm_env(self):
        """Test if litellm is installed."""
        try:
            import litellm
        except ImportError:
            print("litellm package is not installed")
            assert False

        """Test if OPENAI_API_KEY is set."""
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY environment variable is not set")
            assert False

    def test_openai_embedding_function_initialization(self):
        """Test OpenAIEmbeddingFunction initialization."""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization")

        # Check if litellm is available
        self.test_litellm_env()

        # Test initialization with default api_key_env
        ef = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        assert ef is not None
        assert ef.model_name == "text-embedding-ada-002"
        assert ef.api_key_env == "OPENAI_API_KEY"
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")

        # Test initialization with custom api_key_env
        custom_key_env = "CUSTOM_OPENAI_KEY"
        ef_custom = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key_env=custom_key_env
        )
        assert ef_custom.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef_custom.api_key_env}")

    def test_openai_embedding_function_generate_embeddings(self):
        """Test OpenAIEmbeddingFunction embedding generation."""
        print("\n✅ Testing OpenAIEmbeddingFunction embedding generation")

        # Check if litellm is available
        self.test_litellm_env()

        # Initialize embedding function
        ef = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")

        # Test single document
        print("   Testing single document embedding")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

        # Test multiple documents
        print("   Testing multiple documents embedding")
        multiple_docs = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks"
        ]
        embeddings = ef(multiple_docs)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(multiple_docs)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == len(embeddings[0])  # All should have same dimension
        print(f"   Multiple documents embedding dimension: {len(embeddings[0])}")
        print(f"   Multiple documents embeddings: {embeddings}")

        # Test empty input
        print("   Testing empty input")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_openai_embedding_function_dimension(self):
        """Test OpenAIEmbeddingFunction dimension detection."""
        print("\n✅ Testing OpenAIEmbeddingFunction dimension detection")

        # Check if litellm is available
        self.test_litellm_env()

        # Test dimension detection
        ef = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        dim = dimension_of(ef)
        assert dim > 0
        # text-embedding-ada-002 has 1536 dimensions
        assert dim == 1536, f"Expected dimension 1536 for text-embedding-ada-002, got {dim}"
        print(f"   Detected dimension: {dim}")

    def test_openai_embedding_function_with_collection(self, db_client):
        """Test OpenAIEmbeddingFunction with a collection."""
        print("\n✅ Testing OpenAIEmbeddingFunction with collection")

        # Check if litellm is available
        self.test_litellm_env()

        collection_name = f"test_openai_ef_{int(time.time() * 1000)}"
        print(f"   Creating collection: {collection_name}")

        # Create embedding function
        ef = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")

        # Create collection with OpenAI embedding function
        collection = db_client.create_collection(
            name=collection_name,
            embedding_function=ef
        )

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, OpenAIEmbeddingFunction)
        assert collection.dimension == 1536  # text-embedding-ada-002 dimension
        print(f"   Collection dimension: {collection.dimension}")

        # Add documents
        test_documents = [
            "OpenAI provides powerful language models",
            "Embeddings are vector representations of text",
            "Vector databases store and search embeddings efficiently"
        ]
        test_ids = ["1", "2", "3"]
        test_metadatas = [
            {"source": "openai"},
            {"source": "embeddings"},
            {"source": "vector_db"}
        ]

        collection.add(
            ids=test_ids,
            documents=test_documents,
            metadatas=test_metadatas
        )
        print(f"   Added {len(test_documents)} documents")

        # Query
        query_text = "What are embeddings?"
        results = collection.query(
            query_texts=[query_text],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )

        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) > 0
        print(f"   Query returned {len(results['ids'])} results")

        # Cleanup
        try:
            db_client.delete_collection(name=collection_name)
            print(f"   Cleaned up collection: {collection_name}")
        except Exception as e:
            print(f"   Warning: Failed to cleanup collection: {e}")

    def test_openai_embedding_function_different_models(self):
        """Test OpenAIEmbeddingFunction with different OpenAI models."""
        print("\n✅ Testing OpenAIEmbeddingFunction with different models")

        # Check if litellm is available
        self.test_litellm_env()

        test_doc = "Test document for embedding"

        # Test text-embedding-ada-002 (1536 dimensions)
        ef_ada = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        emb_ada = ef_ada(test_doc)
        assert len(emb_ada[0]) == 1536
        print(f"   text-embedding-ada-002 dimension: {len(emb_ada[0])}")

        # Test text-embedding-3-small (1536 dimensions)
        ef_small = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
        emb_small = ef_small(test_doc)
        assert len(emb_small[0]) == 1536
        print(f"   text-embedding-3-small dimension: {len(emb_small[0])}")

        # Test text-embedding-3-large (3072 dimensions)
        ef_large = OpenAIEmbeddingFunction(model_name="text-embedding-3-large")
        emb_large = ef_large(test_doc)
        assert len(emb_large[0]) == 3072
        print(f"   text-embedding-3-large dimension: {len(emb_large[0])}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
