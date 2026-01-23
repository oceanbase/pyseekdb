"""
Unit tests for OpenAIEmbeddingFunction.

Tests OpenAI embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires OPENAI_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_openai_embedding_function.py -v -s
    # Or with environment variable:
    OPENAI_API_KEY=your-key pytest tests/unit_tests/test_openai_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import OpenAIEmbeddingFunction

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
    not os.environ.get("OPENAI_API_KEY") or not is_openai_available(),
    reason="OPENAI_API_KEY environment variable must be set",
)
class TestOpenAIEmbeddingFunction:
    """Test OpenAIEmbeddingFunction - skipped by default, requires manual execution"""

    def test_openai_env(self):
        """Test if openai package is installed and required environment variables are set."""
        if not is_openai_available():
            print("openai package is not installed")
            raise AssertionError("openai package is not installed")

        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY environment variable is not set")
            raise AssertionError("OPENAI_API_KEY environment variable is not set")

    def test_initialization_with_defaults(self):
        """Test OpenAIEmbeddingFunction initialization with default values"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with defaults")

        # Check if openai is available and env vars are set
        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "text-embedding-3-small"
        assert ef.api_key_env == "OPENAI_API_KEY"
        assert ef.api_base == "https://api.openai.com/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_custom_model(self):
        """Test OpenAIEmbeddingFunction initialization with custom model"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with custom model")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

        assert ef.model_name == "text-embedding-3-small"
        assert ef.api_key_env == "OPENAI_API_KEY"
        assert ef.api_base == "https://api.openai.com/v1"
        print(f"   Model name: {ef.model_name}")

    def test_initialization_with_custom_api_key_env(self):
        """Test OpenAIEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with custom API key env")

        self.test_openai_env()

        custom_key_env = "CUSTOM_OPENAI_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = "your-custom-key"

        ef = OpenAIEmbeddingFunction(api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test OpenAIEmbeddingFunction initialization with custom API base"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with custom API base")

        self.test_openai_env()

        # Use OpenAI's actual API base for testing
        custom_base = "https://api.openai.com/v1"
        ef = OpenAIEmbeddingFunction(api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_dimensions(self):
        """Test OpenAIEmbeddingFunction initialization with dimensions parameter"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with dimensions")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=512)
        assert ef._dimensions_param == 512
        print(f"   Dimensions parameter: {ef._dimensions_param}")

    def test_initialization_with_kwargs(self):
        """Test OpenAIEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with kwargs")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction(timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing OpenAIEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                OpenAIEmbeddingFunction()
        finally:
            # Restore API key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_dimension_property_known_models(self):
        """Test dimension property for known models"""
        print("\n✅ Testing OpenAIEmbeddingFunction dimension property for known models")

        self.test_openai_env()

        # Test text-embedding-ada-002
        ef_ada = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        dim_ada = ef_ada.dimension
        assert dim_ada == 1536, f"Expected dimension 1536 for text-embedding-ada-002, got {dim_ada}"
        print(f"   text-embedding-ada-002 dimension: {dim_ada}")

        # Test text-embedding-3-small
        ef_small = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
        dim_small = ef_small.dimension
        assert dim_small == 1536, f"Expected dimension 1536 for text-embedding-3-small, got {dim_small}"
        print(f"   text-embedding-3-small dimension: {dim_small}")

        # Test text-embedding-3-large
        ef_large = OpenAIEmbeddingFunction(model_name="text-embedding-3-large")
        dim_large = ef_large.dimension
        assert dim_large == 3072, f"Expected dimension 3072 for text-embedding-3-large, got {dim_large}"
        print(f"   text-embedding-3-large dimension: {dim_large}")

    def test_dimension_property_with_custom_dimensions(self):
        """Test dimension property when dimensions parameter is set"""
        print("\n✅ Testing OpenAIEmbeddingFunction dimension property with custom dimensions")

        self.test_openai_env()

        # Test with text-embedding-3-small and custom dimensions
        ef_512 = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=512)
        assert ef_512.dimension == 512
        print(f"   text-embedding-3-small with dimensions=512: {ef_512.dimension}")

        ef_256 = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=256)
        assert ef_256.dimension == 256
        print(f"   text-embedding-3-small with dimensions=256: {ef_256.dimension}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing OpenAIEmbeddingFunction dimension property for unknown model")

        self.test_openai_env()

        # This will make an actual API call to get dimension
        ef = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Unknown model dimension (via API call): {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing OpenAIEmbeddingFunction embedding generation (single document)")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing OpenAIEmbeddingFunction embedding generation (multiple documents)")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()
        multiple_docs = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
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
        print("\n✅ Testing OpenAIEmbeddingFunction with empty input")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_dimensions_parameter(self):
        """Test __call__ with dimensions parameter"""
        print("\n✅ Testing OpenAIEmbeddingFunction with dimensions parameter")

        self.test_openai_env()

        # Test with text-embedding-3-small and custom dimensions
        ef_512 = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=512)
        test_doc = "Test document for embedding"
        embeddings_512 = ef_512(test_doc)

        assert len(embeddings_512) == 1
        assert len(embeddings_512[0]) == 512, f"Expected 512 dimensions, got {len(embeddings_512[0])}"
        print(f"   Verified: embeddings have {len(embeddings_512[0])} dimensions")

        # Test with different dimensions
        ef_256 = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=256)
        embeddings_256 = ef_256(test_doc)
        assert len(embeddings_256[0]) == 256, f"Expected 256 dimensions, got {len(embeddings_256[0])}"
        print(f"   Verified: embeddings have {len(embeddings_256[0])} dimensions")

    def test_dimension_of_function(self):
        """Test dimension_of function with OpenAIEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with OpenAIEmbeddingFunction")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        dim = dimension_of(ef)
        assert dim == 1536
        print(f"   dimension_of result: {dim}")

    def test_dimension_of_with_custom_dimensions(self):
        """Test dimension_of function with custom dimensions"""
        print("\n✅ Testing dimension_of function with custom dimensions")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=512)
        dim = dimension_of(ef)
        assert dim == 512
        print(f"   dimension_of result with custom dimensions: {dim}")

    def test_get_default_api_base(self):
        """Test _get_default_api_base method"""
        print("\n✅ Testing _get_default_api_base method")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()
        api_base = ef._get_default_api_base()
        assert api_base == "https://api.openai.com/v1"
        print(f"   Default API base: {api_base}")

    def test_get_default_api_key_env(self):
        """Test _get_default_api_key_env method"""
        print("\n✅ Testing _get_default_api_key_env method")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()
        api_key_env = ef._get_default_api_key_env()
        assert api_key_env == "OPENAI_API_KEY"
        print(f"   Default API key env: {api_key_env}")

    def test_get_model_dimensions(self):
        """Test _get_model_dimensions method"""
        print("\n✅ Testing _get_model_dimensions method")

        self.test_openai_env()

        ef = OpenAIEmbeddingFunction()
        dimensions = ef._get_model_dimensions()

        assert isinstance(dimensions, dict)
        assert "text-embedding-ada-002" in dimensions
        assert "text-embedding-3-small" in dimensions
        assert "text-embedding-3-large" in dimensions
        assert dimensions["text-embedding-ada-002"] == 1536
        assert dimensions["text-embedding-3-small"] == 1536
        assert dimensions["text-embedding-3-large"] == 3072
        print(f"   Model dimensions: {dimensions}")

    def test_different_models_dimensions(self):
        """Test that different OpenAI models return correct dimensions"""
        print("\n✅ Testing different OpenAI models return correct dimensions")

        self.test_openai_env()

        test_cases = [
            ("text-embedding-ada-002", 1536),
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
        ]

        for model_name, expected_dim in test_cases:
            ef = OpenAIEmbeddingFunction(model_name=model_name)
            assert ef.dimension == expected_dim, f"Model {model_name} should have dimension {expected_dim}"
            print(f"   {model_name}: {ef.dimension} dimensions")


@pytest.mark.skipif(not is_openai_available(), reason="openai is not available on this system")
class TestOpenAIEmbeddingFunctionPersistence:
    """Test persistence for OpenAIEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert OpenAIEmbeddingFunction.name() == "openai"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(OPENAI_API_KEY="test-key"):
            ef = OpenAIEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "text-embedding-3-small"
            assert config["api_key_env"] == "OPENAI_API_KEY"
            assert config["api_base"] == "https://api.openai.com/v1"
            assert config["dimensions"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_OPENAI_KEY="test-key"):
            ef = OpenAIEmbeddingFunction(
                model_name="text-embedding-3-large",
                api_key_env="CUSTOM_OPENAI_KEY",
                api_base="https://custom-api.openai.com/v1",
                dimensions=512,
                timeout=60,
                max_retries=5,
            )
            config = ef.get_config()

            assert config["model_name"] == "text-embedding-3-large"
            assert config["api_key_env"] == "CUSTOM_OPENAI_KEY"
            assert config["api_base"] == "https://custom-api.openai.com/v1"
            assert config["dimensions"] == 512
            assert config["client_kwargs"]["timeout"] == 60
            assert config["client_kwargs"]["max_retries"] == 5

    def test_get_config_with_dimensions(self):
        """Test that get_config() correctly includes dimensions parameter"""
        with env_guard(OPENAI_API_KEY="test-key"):
            ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=256)
            config = ef.get_config()

            assert config["dimensions"] == 256

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
            "api_base": "https://api.openai.com/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with env_guard(OPENAI_API_KEY="test-key"):
            restored_ef = OpenAIEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, OpenAIEmbeddingFunction)
            assert restored_ef.model_name == "text-embedding-3-small"
            assert restored_ef.api_key_env == "OPENAI_API_KEY"
            assert restored_ef.api_base == "https://api.openai.com/v1"
            assert restored_ef._dimensions_param is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "text-embedding-3-large",
            "api_key_env": "CUSTOM_OPENAI_KEY",
            "api_base": "https://custom-api.openai.com/v1",
            "dimensions": 512,
            "client_kwargs": {"timeout": 60, "max_retries": 5},
        }

        with env_guard(CUSTOM_OPENAI_KEY="test-key"):
            restored_ef = OpenAIEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, OpenAIEmbeddingFunction)
            assert restored_ef.model_name == "text-embedding-3-large"
            assert restored_ef.api_key_env == "CUSTOM_OPENAI_KEY"
            assert restored_ef.api_base == "https://custom-api.openai.com/v1"
            assert restored_ef._dimensions_param == 512
            assert restored_ef._client_kwargs["timeout"] == 60
            assert restored_ef._client_kwargs["max_retries"] == 5

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(OPENAI_API_KEY="test-key"):
            original_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", dimensions=256)

            config = original_ef.get_config()
            restored_ef = OpenAIEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, OpenAIEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.api_base == original_ef.api_base
            assert restored_ef._dimensions_param == original_ef._dimensions_param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
