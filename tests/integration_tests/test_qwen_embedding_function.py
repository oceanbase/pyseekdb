"""
Test QwenEmbeddingFunction using LiteLLM.

This test is marked to skip in normal test scenarios as it requires:
- litellm package to be installed
- QWEN_API_KEY environment variable to be set
- External API access to Qwen

To run this test manually:
    pytest tests/integration_tests/test_qwen_embedding_function.py -v -s
    # Or with environment variable:
    QWEN_API_KEY=your-key pytest tests/integration_tests/test_qwen_embedding_function.py -v -s
"""
import pytest
import os
import time

import pyseekdb
from pyseekdb.utils.embedding_functions import QwenEmbeddingFunction
from pyseekdb.client.embedding_function import dimension_of


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(
    not os.environ.get("QWEN_API_KEY"),
    reason="QWEN_API_KEY environment variable must be set"
)
class TestQwenEmbeddingFunction:
    """Test QwenEmbeddingFunction - skipped by default, requires manual execution"""

    def test_litellm_env(self):
        """Test if litellm is installed and required environment variables are set."""
        try:
            import litellm
        except ImportError:
            print("litellm package is not installed")
            assert False, "litellm package is not installed"

        if not os.environ.get("QWEN_API_KEY"):
            print("QWEN_API_KEY environment variable is not set")
            assert False, "QWEN_API_KEY environment variable is not set"

    def test_qwen_embedding_function_initialization(self):
        """Test QwenEmbeddingFunction initialization."""
        print("\n✅ Testing QwenEmbeddingFunction initialization")

        # Check if litellm is available and env vars are set
        self.test_litellm_env()

        # Test initialization with default api_key_env and default api_base
        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        assert ef is not None
        assert ef.model_name == "text-embedding-v1"
        assert ef.api_key_env == "QWEN_API_KEY"
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")

        # Test initialization with custom api_key_env (api_base uses default)
        custom_key_env = "CUSTOM_QWEN_KEY"
        ef_custom = QwenEmbeddingFunction(
            model_name="text-embedding-v1",
            api_key_env=custom_key_env
        )
        assert ef_custom.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef_custom.api_key_env}")

    def test_qwen_embedding_function_generate_embeddings(self):
        """Test QwenEmbeddingFunction embedding generation."""
        print("\n✅ Testing QwenEmbeddingFunction embedding generation")

        # Check if litellm is available and env vars are set
        self.test_litellm_env()

        # Initialize embedding function (uses default api_base)
        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")

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
            "机器学习是人工智能的一个子集",
            "Python是一种编程语言",
            "深度学习使用神经网络"
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

    def test_qwen_embedding_function_dimension(self):
        """Test QwenEmbeddingFunction dimension detection."""
        print("\n✅ Testing QwenEmbeddingFunction dimension detection")

        # Check if litellm is available and env vars are set
        self.test_litellm_env()

        # Test dimension detection (uses default api_base)
        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        dim = dimension_of(ef)
        assert dim > 0
        print(f"   Detected dimension: {dim}")

    def test_qwen_embedding_function_with_collection(self, db_client):
        """Test QwenEmbeddingFunction with a collection."""
        print("\n✅ Testing QwenEmbeddingFunction with collection")

        # Check if litellm is available and env vars are set
        self.test_litellm_env()

        collection_name = f"test_qwen_ef_{int(time.time() * 1000)}"
        print(f"   Creating collection: {collection_name}")

        # Create embedding function (uses default api_base)
        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")

        # Get dimension first
        dim = dimension_of(ef)
        print(f"   Embedding dimension: {dim}")

        # Create collection with Qwen embedding function
        collection = db_client.create_collection(
            name=collection_name,
            embedding_function=ef
        )

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, QwenEmbeddingFunction)
        assert collection.dimension == dim
        print(f"   Collection dimension: {collection.dimension}")

        # Add documents
        test_documents = [
            "Qwen提供强大的语言模型",
            "嵌入是文本的向量表示",
            "向量数据库高效存储和搜索嵌入"
        ]
        test_ids = ["1", "2", "3"]
        test_metadatas = [
            {"source": "qwen"},
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
        query_text = "什么是嵌入？"
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

    def test_qwen_embedding_function_with_parameters(self):
        """Test QwenEmbeddingFunction with additional parameters."""
        print("\n✅ Testing QwenEmbeddingFunction with additional parameters")

        # Check if litellm is available and env vars are set
        self.test_litellm_env()

        # Test with timeout and max_retries (api_base uses default)
        ef = QwenEmbeddingFunction(
            model_name="text-embedding-v1",
            timeout=30,
            max_retries=3
        )

        test_doc = "测试文档用于嵌入"
        embeddings = ef(test_doc)
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
        print(f"   Embedding generated successfully with custom parameters")
        print(f"   Embedding dimension: {len(embeddings[0])}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
