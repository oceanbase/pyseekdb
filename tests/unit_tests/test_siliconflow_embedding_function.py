"""
Unit tests for SiliconflowEmbeddingFunction.

Tests SiliconFlow embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires SILICONFLOW_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_siliconflow_embedding_function.py -v -s
    # Or with environment variable:
    SILICONFLOW_API_KEY=your-key pytest tests/unit_tests/test_siliconflow_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import SiliconflowEmbeddingFunction

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
    not os.environ.get("SILICONFLOW_API_KEY") or not is_openai_available(),
    reason="SILICONFLOW_API_KEY environment variable must be set and openai package must be installed",
)
class TestSiliconflowEmbeddingFunction:
    """Test SiliconflowEmbeddingFunction - skipped by default, requires manual execution"""

    def test_siliconflow_env(self):
        """Test if openai package is installed and required environment variables are set."""
        assert is_openai_available(), "openai package is not installed"

        assert os.environ.get("SILICONFLOW_API_KEY"), "SILICONFLOW_API_KEY environment variable is not set"

    def test_initialization_with_defaults(self):
        """Test SiliconflowEmbeddingFunction initialization with default values"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with defaults")

        # Check if openai is available and env vars are set
        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "BAAI/bge-large-zh-v1.5"
        assert ef.api_key_env == "SILICONFLOW_API_KEY"
        assert ef.api_base == "https://api.siliconflow.cn/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_different_models(self):
        """Test SiliconflowEmbeddingFunction initialization with different models"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with different models")

        self.test_siliconflow_env()

        models = [
            "BAAI/bge-large-zh-v1.5",
            "BAAI/bge-large-en-v1.5",
            "netease-youdao/bce-embedding-base_v1",
            "BAAI/bge-m3",
            "Pro/BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-8B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-0.6B",
        ]

        for model in models:
            ef = SiliconflowEmbeddingFunction(model_name=model)
            assert ef.model_name == model
            assert ef.api_key_env == "SILICONFLOW_API_KEY"
            assert ef.api_base == "https://api.siliconflow.cn/v1"
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_custom_api_key_env(self):
        """Test SiliconflowEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with custom API key env")

        self.test_siliconflow_env()

        custom_key_env = "CUSTOM_SILICONFLOW_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = os.environ.get("SILICONFLOW_API_KEY", "your-custom-key")

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test SiliconflowEmbeddingFunction initialization with custom API base"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with custom API base")

        self.test_siliconflow_env()

        # Use SiliconFlow's actual API base for testing
        custom_base = "https://api.siliconflow.cn/v1"
        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5", api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_dimensions(self):
        """Test SiliconflowEmbeddingFunction initialization with dimensions parameter"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with dimensions")

        self.test_siliconflow_env()

        # Test with Qwen model that supports custom dimensions
        ef = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024)
        assert ef._dimensions_param == 1024
        print(f"   Dimensions parameter: {ef._dimensions_param}")

    def test_initialization_with_kwargs(self):
        """Test SiliconflowEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with kwargs")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5", timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing SiliconflowEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("SILICONFLOW_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        finally:
            # Restore API key
            if original_key:
                os.environ["SILICONFLOW_API_KEY"] = original_key

    def test_dimension_property_known_models(self):
        """Test dimension property for known SiliconFlow models"""
        print("\n✅ Testing SiliconflowEmbeddingFunction dimension property for known models")

        self.test_siliconflow_env()

        # Test BAAI/bge-large-zh-v1.5 (1024 dimensions)
        ef_bge_zh = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        dim_bge_zh = ef_bge_zh.dimension
        assert dim_bge_zh == 1024, f"Expected dimension 1024 for BAAI/bge-large-zh-v1.5, got {dim_bge_zh}"
        print(f"   BAAI/bge-large-zh-v1.5 dimension: {dim_bge_zh}")

        # Test BAAI/bge-large-en-v1.5 (1024 dimensions)
        ef_bge_en = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
        dim_bge_en = ef_bge_en.dimension
        assert dim_bge_en == 1024, f"Expected dimension 1024 for BAAI/bge-large-en-v1.5, got {dim_bge_en}"
        print(f"   BAAI/bge-large-en-v1.5 dimension: {dim_bge_en}")

        # Test netease-youdao/bce-embedding-base_v1 (768 dimensions)
        ef_youdao = SiliconflowEmbeddingFunction(model_name="netease-youdao/bce-embedding-base_v1")
        dim_youdao = ef_youdao.dimension
        assert dim_youdao == 768, f"Expected dimension 768 for netease-youdao/bce-embedding-base_v1, got {dim_youdao}"
        print(f"   netease-youdao/bce-embedding-base_v1 dimension: {dim_youdao}")

        # Test BAAI/bge-m3 (1024 dimensions)
        ef_m3 = SiliconflowEmbeddingFunction(model_name="BAAI/bge-m3")
        dim_m3 = ef_m3.dimension
        assert dim_m3 == 1024, f"Expected dimension 1024 for BAAI/bge-m3, got {dim_m3}"
        print(f"   BAAI/bge-m3 dimension: {dim_m3}")

        # Test Qwen/Qwen3-Embedding-8B (4096 dimensions default)
        ef_qwen8b = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-8B")
        dim_qwen8b = ef_qwen8b.dimension
        assert dim_qwen8b == 4096, f"Expected dimension 4096 for Qwen/Qwen3-Embedding-8B, got {dim_qwen8b}"
        print(f"   Qwen/Qwen3-Embedding-8B dimension: {dim_qwen8b}")

        # Test Qwen/Qwen3-Embedding-4B (2560 dimensions default)
        ef_qwen4b = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-4B")
        dim_qwen4b = ef_qwen4b.dimension
        assert dim_qwen4b == 2560, f"Expected dimension 2560 for Qwen/Qwen3-Embedding-4B, got {dim_qwen4b}"
        print(f"   Qwen/Qwen3-Embedding-4B dimension: {dim_qwen4b}")

        # Test Qwen/Qwen3-Embedding-0.6B (1024 dimensions default)
        ef_qwen06b = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-0.6B")
        dim_qwen06b = ef_qwen06b.dimension
        assert dim_qwen06b == 1024, f"Expected dimension 1024 for Qwen/Qwen3-Embedding-0.6B, got {dim_qwen06b}"
        print(f"   Qwen/Qwen3-Embedding-0.6B dimension: {dim_qwen06b}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing SiliconflowEmbeddingFunction dimension property for unknown model")

        self.test_siliconflow_env()

        # This will make an actual API call to get dimension
        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Unknown model dimension (via API call): {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing SiliconflowEmbeddingFunction embedding generation (single document)")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing SiliconflowEmbeddingFunction embedding generation (multiple documents)")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        multiple_docs = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
        ]
        embeddings = ef(multiple_docs)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(multiple_docs)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == len(embeddings[0]), "All embeddings should have same dimension"
        print(f"   Multiple documents embedding dimension: {len(embeddings[0])}")
        print(f"   Number of embeddings: {len(embeddings)}")

    def test_call_empty_input(self):
        """Test __call__ with empty input"""
        print("\n✅ Testing SiliconflowEmbeddingFunction with empty input")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_dimensions_parameter(self):
        """Test __call__ with dimensions parameter for Qwen models"""
        print("\n✅ Testing SiliconflowEmbeddingFunction with dimensions parameter")

        self.test_siliconflow_env()

        # Test with Qwen/Qwen3-Embedding-8B and custom dimensions
        ef_1024 = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024)
        test_doc = "测试文档用于嵌入"
        embeddings_1024 = ef_1024(test_doc)

        assert len(embeddings_1024) == 1
        assert len(embeddings_1024[0]) == 1024, f"Expected 1024 dimensions, got {len(embeddings_1024[0])}"
        print(f"   Verified: embeddings have {len(embeddings_1024[0])} dimensions")

        # Test with different dimensions
        ef_512 = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-8B", dimensions=512)
        embeddings_512 = ef_512(test_doc)
        assert len(embeddings_512[0]) == 512, f"Expected 512 dimensions, got {len(embeddings_512[0])}"
        print(f"   Verified: embeddings have {len(embeddings_512[0])} dimensions")

        # Test with Qwen/Qwen3-Embedding-4B model
        ef_4b_1024 = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-4B", dimensions=1024)
        embeddings_4b_1024 = ef_4b_1024(test_doc)
        assert len(embeddings_4b_1024[0]) == 1024, f"Expected 1024 dimensions, got {len(embeddings_4b_1024[0])}"
        print(f"   Verified: embeddings have {len(embeddings_4b_1024[0])} dimensions")

    def test_dimension_of_function(self):
        """Test dimension_of function with SiliconflowEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with SiliconflowEmbeddingFunction")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of result for BAAI/bge-large-zh-v1.5: {dim}")

        ef_m3 = SiliconflowEmbeddingFunction(model_name="BAAI/bge-m3")
        dim_m3 = dimension_of(ef_m3)
        assert dim_m3 == 1024
        print(f"   dimension_of result for BAAI/bge-m3: {dim_m3}")

        ef_qwen8b = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-8B")
        dim_qwen8b = dimension_of(ef_qwen8b)
        assert dim_qwen8b == 4096
        print(f"   dimension_of result for Qwen/Qwen3-Embedding-8B: {dim_qwen8b}")

    def test_get_default_api_base(self):
        """Test _get_default_api_base method"""
        print("\n✅ Testing _get_default_api_base method")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        api_base = ef._get_default_api_base()
        assert api_base == "https://api.siliconflow.cn/v1"
        print(f"   Default API base: {api_base}")

    def test_get_default_api_key_env(self):
        """Test _get_default_api_key_env method"""
        print("\n✅ Testing _get_default_api_key_env method")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        api_key_env = ef._get_default_api_key_env()
        assert api_key_env == "SILICONFLOW_API_KEY"
        print(f"   Default API key env: {api_key_env}")

    def test_get_model_dimensions(self):
        """Test _get_model_dimensions method"""
        print("\n✅ Testing _get_model_dimensions method")

        self.test_siliconflow_env()

        ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5")
        dimensions = ef._get_model_dimensions()

        assert isinstance(dimensions, dict)
        assert "BAAI/bge-large-zh-v1.5" in dimensions
        assert "BAAI/bge-large-en-v1.5" in dimensions
        assert "netease-youdao/bce-embedding-base_v1" in dimensions
        assert "BAAI/bge-m3" in dimensions
        assert "Pro/BAAI/bge-m3" in dimensions
        assert "Qwen/Qwen3-Embedding-8B" in dimensions
        assert "Qwen/Qwen3-Embedding-4B" in dimensions
        assert "Qwen/Qwen3-Embedding-0.6B" in dimensions
        assert dimensions["BAAI/bge-large-zh-v1.5"] == 1024
        assert dimensions["BAAI/bge-large-en-v1.5"] == 1024
        assert dimensions["netease-youdao/bce-embedding-base_v1"] == 768
        assert dimensions["BAAI/bge-m3"] == 1024
        assert dimensions["Pro/BAAI/bge-m3"] == 1024
        assert dimensions["Qwen/Qwen3-Embedding-8B"] == 4096
        assert dimensions["Qwen/Qwen3-Embedding-4B"] == 2560
        assert dimensions["Qwen/Qwen3-Embedding-0.6B"] == 1024
        print(f"   Model dimensions: {dimensions}")


@pytest.mark.skipif(not is_openai_available(), reason="openai is not available on this system")
class TestSiliconflowEmbeddingFunctionPersistence:
    """Test persistence for SiliconflowEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert SiliconflowEmbeddingFunction.name() == "siliconflow"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(SILICONFLOW_API_KEY="test-key"):
            ef = SiliconflowEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "BAAI/bge-large-zh-v1.5"
            assert config["api_key_env"] == "SILICONFLOW_API_KEY"
            assert config["api_base"] == "https://api.siliconflow.cn/v1"
            assert config["dimensions"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_SILICONFLOW_KEY="test-key"):
            ef = SiliconflowEmbeddingFunction(
                model_name="BAAI/bge-m3",
                api_key_env="CUSTOM_SILICONFLOW_KEY",
                api_base="https://custom-siliconflow.com/v1",
                dimensions=512,
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "BAAI/bge-m3"
            assert config["api_key_env"] == "CUSTOM_SILICONFLOW_KEY"
            assert config["api_base"] == "https://custom-siliconflow.com/v1"
            assert config["dimensions"] == 512
            assert config["client_kwargs"]["timeout"] == 60

    def test_get_config_with_dimensions(self):
        """Test that get_config() correctly includes dimensions parameter"""
        with env_guard(SILICONFLOW_API_KEY="test-key"):
            ef = SiliconflowEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024)
            config = ef.get_config()

            assert config["dimensions"] == 1024

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "BAAI/bge-large-zh-v1.5",
            "api_key_env": "SILICONFLOW_API_KEY",
            "api_base": "https://api.siliconflow.cn/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with env_guard(SILICONFLOW_API_KEY="test-key"):
            restored_ef = SiliconflowEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, SiliconflowEmbeddingFunction)
            assert restored_ef.model_name == "BAAI/bge-large-zh-v1.5"
            assert restored_ef.api_key_env == "SILICONFLOW_API_KEY"
            assert restored_ef.api_base == "https://api.siliconflow.cn/v1"
            assert restored_ef._dimensions_param is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "BAAI/bge-m3",
            "api_key_env": "CUSTOM_SILICONFLOW_KEY",
            "api_base": "https://custom-siliconflow.com/v1",
            "dimensions": 512,
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_SILICONFLOW_KEY="test-key"):
            restored_ef = SiliconflowEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, SiliconflowEmbeddingFunction)
            assert restored_ef.model_name == "BAAI/bge-m3"
            assert restored_ef.api_key_env == "CUSTOM_SILICONFLOW_KEY"
            assert restored_ef.api_base == "https://custom-siliconflow.com/v1"
            assert restored_ef._dimensions_param == 512
            assert restored_ef._client_kwargs["timeout"] == 60

    def test_build_from_config_missing_model_name(self):
        """Test that build_from_config() raises ValueError when model_name is missing"""
        config = {
            "api_key_env": "SILICONFLOW_API_KEY",
            "api_base": "https://api.siliconflow.cn/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with pytest.raises(ValueError, match="Missing required field 'model_name'"):
            SiliconflowEmbeddingFunction.build_from_config(config)

    def test_build_from_config_invalid_client_kwargs(self):
        """Test that build_from_config() raises TypeError when client_kwargs is not a dict"""
        config = {
            "model_name": "BAAI/bge-large-zh-v1.5",
            "api_key_env": "SILICONFLOW_API_KEY",
            "api_base": "https://api.siliconflow.cn/v1",
            "dimensions": None,
            "client_kwargs": "not-a-dict",
        }

        with pytest.raises(TypeError, match="client_kwargs must be a dictionary"):
            SiliconflowEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(SILICONFLOW_API_KEY="test-key"):
            original_ef = SiliconflowEmbeddingFunction(model_name="BAAI/bge-m3", dimensions=256)

            config = original_ef.get_config()
            restored_ef = SiliconflowEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, SiliconflowEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.api_base == original_ef.api_base
            assert restored_ef._dimensions_param == original_ef._dimensions_param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
