"""
Unit tests for TencentHunyuanEmbeddingFunction.

Tests Tencent Hunyuan embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires HUNYUAN_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_tencent_hunyuan_embedding_function.py -v -s
    # Or with environment variable:
    HUNYUAN_API_KEY=your-key pytest tests/unit_tests/test_tencent_hunyuan_embedding_function.py -v -s
"""

import importlib.util
import os
import warnings

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import TencentHunyuanEmbeddingFunction

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
    not os.environ.get("HUNYUAN_API_KEY") or not is_openai_available(),
    reason="HUNYUAN_API_KEY environment variable must be set and openai package must be installed",
)
class TestTencentHunyuanEmbeddingFunction:
    """Test TencentHunyuanEmbeddingFunction - skipped by default, requires manual execution"""

    def test_hunyuan_env(self):
        """Test if openai package is installed and required environment variables are set."""
        assert is_openai_available(), "openai package is not installed"

        assert os.environ.get("HUNYUAN_API_KEY"), "HUNYUAN_API_KEY environment variable is not set"

    def test_initialization_with_defaults(self):
        """Test TencentHunyuanEmbeddingFunction initialization with default values"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with defaults")

        # Check if openai is available and env vars are set
        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "hunyuan-embedding"
        assert ef.api_key_env == "HUNYUAN_API_KEY"
        assert ef.api_base == "https://api.hunyuan.cloud.tencent.com/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_custom_api_key_env(self):
        """Test TencentHunyuanEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with custom API key env")

        self.test_hunyuan_env()

        custom_key_env = "CUSTOM_HUNYUAN_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = os.environ.get("HUNYUAN_API_KEY", "your-custom-key")

        ef = TencentHunyuanEmbeddingFunction(api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test TencentHunyuanEmbeddingFunction initialization with custom API base"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with custom API base")

        self.test_hunyuan_env()

        # Use Tencent Hunyuan's actual API base for testing
        custom_base = "https://api.hunyuan.cloud.tencent.com/v1"
        ef = TencentHunyuanEmbeddingFunction(api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_dimensions_warning(self):
        """Test TencentHunyuanEmbeddingFunction initialization with dimensions parameter (should warn)"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with dimensions (warning)")

        self.test_hunyuan_env()

        # Should issue a warning when dimensions is provided
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ef = TencentHunyuanEmbeddingFunction(dimensions=512)

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "dimensions parameter is not supported" in str(w[0].message)
            print(f"   Warning issued: {w[0].message}")

        # Dimensions should still be None internally
        assert ef._dimensions_param is None
        # But dimension property should return 1024 (fixed)
        assert ef.dimension == 1024

    def test_initialization_with_custom_model_name_warning(self):
        """Test TencentHunyuanEmbeddingFunction initialization with custom model name (should warn)"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with custom model name (warning)")

        self.test_hunyuan_env()

        # Should issue a warning when model_name is not the default
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ef = TencentHunyuanEmbeddingFunction(model_name="custom-model")

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "may not be supported" in str(w[0].message)
            print(f"   Warning issued: {w[0].message}")

        assert ef.model_name == "custom-model"

    def test_initialization_with_kwargs(self):
        """Test TencentHunyuanEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with kwargs")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction(timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("HUNYUAN_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                TencentHunyuanEmbeddingFunction()
        finally:
            # Restore API key
            if original_key:
                os.environ["HUNYUAN_API_KEY"] = original_key

    def test_dimension_property(self):
        """Test dimension property (always returns 1024)"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction dimension property")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        dim = ef.dimension
        assert dim == 1024, f"Expected dimension 1024, got {dim}"
        print(f"   Dimension: {dim}")

        # Test that dimension is always 1024 regardless of initialization
        ef2 = TencentHunyuanEmbeddingFunction(dimensions=512)  # Should warn but dimension still 1024
        dim2 = ef2.dimension
        assert dim2 == 1024, f"Expected dimension 1024 even with dimensions=512, got {dim2}"
        print(f"   Dimension (with dimensions=512): {dim2}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction embedding generation (single document)")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == 1024, "Tencent Hunyuan embeddings should be 1024 dimensions"
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction embedding generation (multiple documents)")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        multiple_docs = [
            "机器学习是人工智能的一个子集",
            "Python是一种编程语言",
            "深度学习使用神经网络",
        ]
        embeddings = ef(multiple_docs)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(multiple_docs)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 1024, "All embeddings should be 1024 dimensions"
            assert len(emb) == len(embeddings[0]), "All embeddings should have same dimension"
        print(f"   Multiple documents embedding dimension: {len(embeddings[0])}")
        print(f"   Number of embeddings: {len(embeddings)}")

    def test_call_empty_input(self):
        """Test __call__ with empty input"""
        print("\n✅ Testing TencentHunyuanEmbeddingFunction with empty input")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_dimension_of_function(self):
        """Test dimension_of function with TencentHunyuanEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with TencentHunyuanEmbeddingFunction")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of result: {dim}")

    def test_get_default_api_base(self):
        """Test _get_default_api_base method"""
        print("\n✅ Testing _get_default_api_base method")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        api_base = ef._get_default_api_base()
        assert api_base == "https://api.hunyuan.cloud.tencent.com/v1"
        print(f"   Default API base: {api_base}")

    def test_get_default_api_key_env(self):
        """Test _get_default_api_key_env method"""
        print("\n✅ Testing _get_default_api_key_env method")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        api_key_env = ef._get_default_api_key_env()
        assert api_key_env == "HUNYUAN_API_KEY"
        print(f"   Default API key env: {api_key_env}")

    def test_get_model_dimensions(self):
        """Test _get_model_dimensions method"""
        print("\n✅ Testing _get_model_dimensions method")

        self.test_hunyuan_env()

        ef = TencentHunyuanEmbeddingFunction()
        dimensions = ef._get_model_dimensions()

        assert isinstance(dimensions, dict)
        assert "hunyuan-embedding" in dimensions
        assert dimensions["hunyuan-embedding"] == 1024
        print(f"   Model dimensions: {dimensions}")


@pytest.mark.skipif(not is_openai_available(), reason="openai is not available on this system")
class TestTencentHunyuanEmbeddingFunctionPersistence:
    """Test persistence for TencentHunyuanEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert TencentHunyuanEmbeddingFunction.name() == "tencent_hunyuan"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(HUNYUAN_API_KEY="test-key"):
            ef = TencentHunyuanEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "hunyuan-embedding"
            assert config["api_key_env"] == "HUNYUAN_API_KEY"
            assert config["api_base"] == "https://api.hunyuan.cloud.tencent.com/v1"
            assert config["dimensions"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_HUNYUAN_KEY="test-key"):
            ef = TencentHunyuanEmbeddingFunction(
                api_key_env="CUSTOM_HUNYUAN_KEY",
                api_base="https://custom-hunyuan.com/v1",
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "hunyuan-embedding"
            assert config["api_key_env"] == "CUSTOM_HUNYUAN_KEY"
            assert config["api_base"] == "https://custom-hunyuan.com/v1"
            assert config["dimensions"] is None
            assert config["client_kwargs"]["timeout"] == 60

    def test_get_config_with_dimensions(self):
        """Test that get_config() correctly ignores dimensions parameter"""
        with env_guard(HUNYUAN_API_KEY="test-key"):
            # Even if dimensions is provided, it should be None in config
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ef = TencentHunyuanEmbeddingFunction(dimensions=512)
            config = ef.get_config()

            assert config["dimensions"] is None

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "hunyuan-embedding",
            "api_key_env": "HUNYUAN_API_KEY",
            "api_base": "https://api.hunyuan.cloud.tencent.com/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with env_guard(HUNYUAN_API_KEY="test-key"):
            restored_ef = TencentHunyuanEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, TencentHunyuanEmbeddingFunction)
            assert restored_ef.model_name == "hunyuan-embedding"
            assert restored_ef.api_key_env == "HUNYUAN_API_KEY"
            assert restored_ef.api_base == "https://api.hunyuan.cloud.tencent.com/v1"
            assert restored_ef._dimensions_param is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "hunyuan-embedding",
            "api_key_env": "CUSTOM_HUNYUAN_KEY",
            "api_base": "https://custom-hunyuan.com/v1",
            "dimensions": None,
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_HUNYUAN_KEY="test-key"):
            restored_ef = TencentHunyuanEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, TencentHunyuanEmbeddingFunction)
            assert restored_ef.model_name == "hunyuan-embedding"
            assert restored_ef.api_key_env == "CUSTOM_HUNYUAN_KEY"
            assert restored_ef.api_base == "https://custom-hunyuan.com/v1"
            assert restored_ef._dimensions_param is None
            assert restored_ef._client_kwargs["timeout"] == 60

    def test_build_from_config_missing_model_name(self):
        """Test that build_from_config() raises ValueError when model_name is missing"""
        config = {
            "api_key_env": "HUNYUAN_API_KEY",
            "api_base": "https://api.hunyuan.cloud.tencent.com/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with pytest.raises(ValueError, match="Missing required field 'model_name'"):
            TencentHunyuanEmbeddingFunction.build_from_config(config)

    def test_build_from_config_invalid_client_kwargs(self):
        """Test that build_from_config() raises TypeError when client_kwargs is not a dict"""
        config = {
            "model_name": "hunyuan-embedding",
            "api_key_env": "HUNYUAN_API_KEY",
            "api_base": "https://api.hunyuan.cloud.tencent.com/v1",
            "dimensions": None,
            "client_kwargs": "not-a-dict",
        }

        with pytest.raises(TypeError, match="client_kwargs must be a dictionary"):
            TencentHunyuanEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(HUNYUAN_API_KEY="test-key"):
            original_ef = TencentHunyuanEmbeddingFunction()

            config = original_ef.get_config()
            restored_ef = TencentHunyuanEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, TencentHunyuanEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.api_base == original_ef.api_base
            assert restored_ef._dimensions_param == original_ef._dimensions_param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
