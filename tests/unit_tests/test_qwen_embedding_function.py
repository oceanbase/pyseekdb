"""
Unit tests for QwenEmbeddingFunction.

Tests Qwen embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires DASHSCOPE_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_qwen_embedding_function.py -v -s
    # Or with environment variable:
    DASHSCOPE_API_KEY=your-key pytest tests/unit_tests/test_qwen_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import QwenEmbeddingFunction

from .test_utils import env_guard


def is_openai_available() -> bool:
    """
    Check if openai is available for testing.

    Returns:
        True if openai is available, False otherwise.
    """
    return importlib.util.find_spec("openai") is not None


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY") or not is_openai_available(),
    reason="DASHSCOPE_API_KEY environment variable must be set",
)
class TestQwenEmbeddingFunction:
    """Test QwenEmbeddingFunction - skipped by default, requires manual execution"""

    def test_qwen_env(self):
        """Test if openai package is installed and required environment variables are set."""
        if not is_openai_available():
            print("openai package is not installed")
            raise AssertionError("openai package is not installed")

        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("DASHSCOPE_API_KEY environment variable is not set")
            raise AssertionError("DASHSCOPE_API_KEY environment variable is not set")

    def test_initialization_with_defaults(self):
        """Test QwenEmbeddingFunction initialization with default values"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with defaults")

        # Check if openai is available and env vars are set
        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")

        assert ef is not None
        assert ef.model_name == "text-embedding-v1"
        assert ef.api_key_env == "DASHSCOPE_API_KEY"
        assert ef.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_different_models(self):
        """Test QwenEmbeddingFunction initialization with different models"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with different models")

        self.test_qwen_env()

        models = [
            "text-embedding-v1",
            "text-embedding-v2",
            "text-embedding-v3",
            "text-embedding-v4",
        ]

        for model in models:
            ef = QwenEmbeddingFunction(model_name=model)
            assert ef.model_name == model
            assert ef.api_key_env == "DASHSCOPE_API_KEY"
            assert ef.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_custom_api_key_env(self):
        """Test QwenEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with custom API key env")

        self.test_qwen_env()

        custom_key_env = "CUSTOM_QWEN_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = "your-custom-key"

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test QwenEmbeddingFunction initialization with custom API base"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with custom API base")

        self.test_qwen_env()

        # Use Qwen's actual API base for testing
        custom_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ef = QwenEmbeddingFunction(model_name="text-embedding-v1", api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_dimensions(self):
        """Test QwenEmbeddingFunction initialization with dimensions parameter"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with dimensions")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v3", dimensions=512)
        assert ef._dimensions_param == 512
        print(f"   Dimensions parameter: {ef._dimensions_param}")

    def test_initialization_with_kwargs(self):
        """Test QwenEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with kwargs")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1", timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing QwenEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                QwenEmbeddingFunction(model_name="text-embedding-v1")
        finally:
            # Restore API key
            if original_key:
                os.environ["DASHSCOPE_API_KEY"] = original_key

    def test_dimension_property_known_models(self):
        """Test dimension property for known Qwen models"""
        print("\n✅ Testing QwenEmbeddingFunction dimension property for known models")

        self.test_qwen_env()

        # Test v1 and v2 (1536 dimensions)
        ef_v1 = QwenEmbeddingFunction(model_name="text-embedding-v1")
        dim_v1 = ef_v1.dimension
        assert dim_v1 == 1536, f"Expected dimension 1536 for text-embedding-v1, got {dim_v1}"
        print(f"   text-embedding-v1 dimension: {dim_v1}")

        ef_v2 = QwenEmbeddingFunction(model_name="text-embedding-v2")
        dim_v2 = ef_v2.dimension
        assert dim_v2 == 1536, f"Expected dimension 1536 for text-embedding-v2, got {dim_v2}"
        print(f"   text-embedding-v2 dimension: {dim_v2}")

        # Test v3 and v4 (1024 dimensions)
        ef_v3 = QwenEmbeddingFunction(model_name="text-embedding-v3")
        dim_v3 = ef_v3.dimension
        assert dim_v3 == 1024, f"Expected dimension 1024 for text-embedding-v3, got {dim_v3}"
        print(f"   text-embedding-v3 dimension: {dim_v3}")

        ef_v4 = QwenEmbeddingFunction(model_name="text-embedding-v4")
        dim_v4 = ef_v4.dimension
        assert dim_v4 == 1024, f"Expected dimension 1024 for text-embedding-v4, got {dim_v4}"
        print(f"   text-embedding-v4 dimension: {dim_v4}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing QwenEmbeddingFunction dimension property for unknown model")

        self.test_qwen_env()

        # This will make an actual API call to get dimension
        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Unknown model dimension (via API call): {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing QwenEmbeddingFunction embedding generation (single document)")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing QwenEmbeddingFunction embedding generation (multiple documents)")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v2")
        multiple_docs = [
            "机器学习是人工智能的一个子集",
            "Python是一种编程语言",
            "深度学习使用神经网络",
        ]
        embeddings = ef(multiple_docs)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(multiple_docs)
        for _i, emb in enumerate(embeddings):
            assert isinstance(emb, list)
            assert len(emb) == len(embeddings[0]), "All embeddings should have same dimension"
        print(f"   Multiple documents embedding dimension: {len(embeddings[0])}")
        print(f"   Number of embeddings: {len(embeddings)}")

    def test_call_empty_input(self):
        """Test __call__ with empty input"""
        print("\n✅ Testing QwenEmbeddingFunction with empty input")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_dimensions_parameter(self):
        """Test __call__ with dimensions parameter"""
        print("\n✅ Testing QwenEmbeddingFunction with dimensions parameter")

        self.test_qwen_env()

        # Test with text-embedding-v3 and custom dimensions
        ef_512 = QwenEmbeddingFunction(model_name="text-embedding-v3", dimensions=512)
        test_doc = "测试文档用于嵌入"
        embeddings_512 = ef_512(test_doc)

        assert len(embeddings_512) == 1
        assert len(embeddings_512[0]) == 512, f"Expected 512 dimensions, got {len(embeddings_512[0])}"
        print(f"   Verified: embeddings have {len(embeddings_512[0])} dimensions")

        # Test with different dimensions
        ef_256 = QwenEmbeddingFunction(model_name="text-embedding-v3", dimensions=256)
        embeddings_256 = ef_256(test_doc)
        assert len(embeddings_256[0]) == 256, f"Expected 256 dimensions, got {len(embeddings_256[0])}"
        print(f"   Verified: embeddings have {len(embeddings_256[0])} dimensions")

        # Test with v4 model
        ef_v4_1024 = QwenEmbeddingFunction(model_name="text-embedding-v4", dimensions=1024)
        embeddings_1024 = ef_v4_1024(test_doc)
        assert len(embeddings_1024[0]) == 1024, f"Expected 1024 dimensions, got {len(embeddings_1024[0])}"
        print(f"   Verified: embeddings have {len(embeddings_1024[0])} dimensions")

    def test_dimension_of_function(self):
        """Test dimension_of function with QwenEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with QwenEmbeddingFunction")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        dim = dimension_of(ef)
        assert dim == 1536
        print(f"   dimension_of result for v1: {dim}")

        ef_v3 = QwenEmbeddingFunction(model_name="text-embedding-v3")
        dim_v3 = dimension_of(ef_v3)
        assert dim_v3 == 1024
        print(f"   dimension_of result for v3: {dim_v3}")

    def test_get_default_api_base(self):
        """Test _get_default_api_base method"""
        print("\n✅ Testing _get_default_api_base method")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        api_base = ef._get_default_api_base()
        assert api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        print(f"   Default API base: {api_base}")

    def test_get_default_api_key_env(self):
        """Test _get_default_api_key_env method"""
        print("\n✅ Testing _get_default_api_key_env method")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        api_key_env = ef._get_default_api_key_env()
        assert api_key_env == "DASHSCOPE_API_KEY"
        print(f"   Default API key env: {api_key_env}")

    def test_get_model_dimensions(self):
        """Test _get_model_dimensions method"""
        print("\n✅ Testing _get_model_dimensions method")

        self.test_qwen_env()

        ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
        dimensions = ef._get_model_dimensions()

        assert isinstance(dimensions, dict)
        assert "text-embedding-v1" in dimensions
        assert "text-embedding-v2" in dimensions
        assert "text-embedding-v3" in dimensions
        assert "text-embedding-v4" in dimensions
        assert dimensions["text-embedding-v1"] == 1536
        assert dimensions["text-embedding-v2"] == 1536
        assert dimensions["text-embedding-v3"] == 1024
        assert dimensions["text-embedding-v4"] == 1024
        print(f"   Model dimensions: {dimensions}")


@pytest.mark.skipif(not is_openai_available(), reason="openai is not available on this system")
class TestQwenEmbeddingFunctionPersistence:
    """Test persistence for QwenEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert QwenEmbeddingFunction.name() == "qwen"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(DASHSCOPE_API_KEY="test-key"):
            ef = QwenEmbeddingFunction(model_name="text-embedding-v1")
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "text-embedding-v1"
            assert config["api_key_env"] == "DASHSCOPE_API_KEY"
            assert config["api_base"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            assert config["dimensions"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_QWEN_KEY="test-key"):
            ef = QwenEmbeddingFunction(
                model_name="text-embedding-v3",
                api_key_env="CUSTOM_QWEN_KEY",
                api_base="https://custom-dashscope.com/v1",
                dimensions=512,
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "text-embedding-v3"
            assert config["api_key_env"] == "CUSTOM_QWEN_KEY"
            assert config["api_base"] == "https://custom-dashscope.com/v1"
            assert config["dimensions"] == 512
            assert config["client_kwargs"]["timeout"] == 60

    def test_get_config_with_dimensions(self):
        """Test that get_config() correctly includes dimensions parameter"""
        with env_guard(DASHSCOPE_API_KEY="test-key"):
            ef = QwenEmbeddingFunction(model_name="text-embedding-v3", dimensions=256)
            config = ef.get_config()

            assert config["dimensions"] == 256

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "text-embedding-v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with env_guard(DASHSCOPE_API_KEY="test-key"):
            restored_ef = QwenEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, QwenEmbeddingFunction)
            assert restored_ef.model_name == "text-embedding-v1"
            assert restored_ef.api_key_env == "DASHSCOPE_API_KEY"
            assert restored_ef.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            assert restored_ef._dimensions_param is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "text-embedding-v3",
            "api_key_env": "CUSTOM_QWEN_KEY",
            "api_base": "https://custom-dashscope.com/v1",
            "dimensions": 512,
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_QWEN_KEY="test-key"):
            restored_ef = QwenEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, QwenEmbeddingFunction)
            assert restored_ef.model_name == "text-embedding-v3"
            assert restored_ef.api_key_env == "CUSTOM_QWEN_KEY"
            assert restored_ef.api_base == "https://custom-dashscope.com/v1"
            assert restored_ef._dimensions_param == 512
            assert restored_ef._client_kwargs["timeout"] == 60

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(DASHSCOPE_API_KEY="test-key"):
            original_ef = QwenEmbeddingFunction(model_name="text-embedding-v1", dimensions=256)

            config = original_ef.get_config()
            restored_ef = QwenEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, QwenEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.api_base == original_ef.api_base
            assert restored_ef._dimensions_param == original_ef._dimensions_param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
