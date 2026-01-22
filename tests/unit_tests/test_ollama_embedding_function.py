"""
Unit tests for OllamaEmbeddingFunction.

Tests Ollama embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires Ollama to be running locally and models to be pulled.

To run this test manually:
    pytest tests/unit_tests/test_ollama_embedding_function.py -v -s
    # Make sure Ollama is running: ollama serve
    # Pull required models: ollama pull nomic-embed-text
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import OllamaEmbeddingFunction

from .test_utils import env_guard


def is_openai_available() -> bool:
    """
    Check if openai is available for testing.

    Returns:
        True if openai is available, False otherwise.
    """
    return importlib.util.find_spec("openai") is not None


def is_ollama_available() -> bool:
    """
    Check if Ollama is available (running and accessible).

    Returns:
        True if Ollama is accessible, False otherwise.
    """
    if not is_openai_available():
        return False

    try:
        import openai

        # Try to connect to Ollama's default endpoint
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama ignores the key but requires it
        )
        # Try to list models (this will fail if Ollama is not running)
        client.models.list()
    except Exception:
        return False
    return True


# Skip this test by default - it requires Ollama to be running locally
@pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama must be running locally and openai package must be installed",
)
class TestOllamaEmbeddingFunction:
    """Test OllamaEmbeddingFunction - skipped by default, requires Ollama to be running"""

    def test_ollama_env(self):
        """Test if openai package is installed and Ollama is accessible."""
        assert is_openai_available(), "openai package is not installed"
        assert is_ollama_available(), "Ollama is not running or not accessible"

    def test_initialization_with_defaults(self):
        """Test OllamaEmbeddingFunction initialization with default values"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization with defaults")

        # Check if openai is available and Ollama is accessible
        self.test_ollama_env()

        ef = OllamaEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "nomic-embed-text"
        assert ef.api_key_env == "OLLAMA_API_KEY"
        assert ef.api_base == "http://localhost:11434/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_different_models(self):
        """Test OllamaEmbeddingFunction initialization with different models"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization with different models")

        self.test_ollama_env()

        models = [
            "nomic-embed-text",
            "all-minilm",
        ]

        for model in models:
            ef = OllamaEmbeddingFunction(model_name=model)
            assert ef.model_name == model
            assert ef.api_key_env == "OLLAMA_API_KEY"
            assert ef.api_base == "http://localhost:11434/v1"
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_custom_api_key_env(self):
        """Test OllamaEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization with custom API key env")

        self.test_ollama_env()

        custom_key_env = "CUSTOM_OLLAMA_KEY"
        # Ollama ignores the key, but we need to set it
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = "ollama"

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test OllamaEmbeddingFunction initialization with custom API base"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization with custom API base")

        self.test_ollama_env()

        # Use Ollama's default API base for testing
        custom_base = "http://localhost:11434/v1"
        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_dimensions(self):
        """Test OllamaEmbeddingFunction initialization with dimensions parameter"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization with dimensions")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", dimensions=512)
        assert ef._dimensions_param == 512
        print(f"   Dimensions parameter: {ef._dimensions_param}")

    def test_initialization_with_kwargs(self):
        """Test OllamaEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization with kwargs")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_without_api_key(self):
        """Test that initialization works even without API key (Ollama ignores it)"""
        print("\n✅ Testing OllamaEmbeddingFunction initialization without API key")

        self.test_ollama_env()

        # Temporarily remove API key
        original_key = os.environ.pop("OLLAMA_API_KEY", None)
        try:
            # Should work because OllamaEmbeddingFunction sets a default
            ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
            assert ef is not None
            # The API key should be set to "ollama" by default
            assert os.environ.get("OLLAMA_API_KEY") == "ollama"
            print("   Initialization succeeded without API key (defaulted to 'ollama')")
        finally:
            # Restore API key
            if original_key:
                os.environ["OLLAMA_API_KEY"] = original_key
            elif "OLLAMA_API_KEY" in os.environ:
                os.environ.pop("OLLAMA_API_KEY")

    def test_dimension_property_known_models(self):
        """Test dimension property for known Ollama models"""
        print("\n✅ Testing OllamaEmbeddingFunction dimension property for known models")

        self.test_ollama_env()

        # Test nomic-embed-text (768 dimensions)
        ef_nomic = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        dim_nomic = ef_nomic.dimension
        assert dim_nomic == 768, f"Expected dimension 768 for nomic-embed-text, got {dim_nomic}"
        print(f"   nomic-embed-text dimension: {dim_nomic}")

        # Test all-minilm (384 dimensions)
        ef_minilm = OllamaEmbeddingFunction(model_name="all-minilm")
        dim_minilm = ef_minilm.dimension
        assert dim_minilm == 384, f"Expected dimension 384 for all-minilm, got {dim_minilm}"
        print(f"   all-minilm dimension: {dim_minilm}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing OllamaEmbeddingFunction dimension property for unknown model")

        self.test_ollama_env()

        # This will make an actual API call to get dimension
        ef = OllamaEmbeddingFunction(model_name="embeddinggemma")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Unknown model dimension (via API call): {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing OllamaEmbeddingFunction embedding generation (single document)")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing OllamaEmbeddingFunction embedding generation (multiple documents)")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
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
        print("\n✅ Testing OllamaEmbeddingFunction with empty input")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_dimensions_parameter(self):
        """Test __call__ with dimensions parameter"""
        print("\n✅ Testing OllamaEmbeddingFunction with dimensions parameter")

        self.test_ollama_env()

        # Test with nomic-embed-text and custom dimensions (if supported)
        ef_512 = OllamaEmbeddingFunction(model_name="nomic-embed-text", dimensions=512)
        test_doc = "Test document for embedding"
        embeddings_512 = ef_512(test_doc)

        assert len(embeddings_512) == 1
        # Note: Some models may not support custom dimensions, so we just check it's valid
        assert len(embeddings_512[0]) > 0
        print(f"   Verified: embeddings have {len(embeddings_512[0])} dimensions")

    def test_dimension_of_function(self):
        """Test dimension_of function with OllamaEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with OllamaEmbeddingFunction")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        dim = dimension_of(ef)
        assert dim == 768
        print(f"   dimension_of result for nomic-embed-text: {dim}")

        ef_minilm = OllamaEmbeddingFunction(model_name="all-minilm")
        dim_minilm = dimension_of(ef_minilm)
        assert dim_minilm == 384
        print(f"   dimension_of result for all-minilm: {dim_minilm}")

    def test_get_default_api_base(self):
        """Test _get_default_api_base method"""
        print("\n✅ Testing _get_default_api_base method")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        api_base = ef._get_default_api_base()
        assert api_base == "http://localhost:11434/v1"
        print(f"   Default API base: {api_base}")

    def test_get_default_api_key_env(self):
        """Test _get_default_api_key_env method"""
        print("\n✅ Testing _get_default_api_key_env method")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        api_key_env = ef._get_default_api_key_env()
        assert api_key_env == "OLLAMA_API_KEY"
        print(f"   Default API key env: {api_key_env}")

    def test_get_model_dimensions(self):
        """Test _get_model_dimensions method"""
        print("\n✅ Testing _get_model_dimensions method")

        self.test_ollama_env()

        ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        dimensions = ef._get_model_dimensions()

        assert isinstance(dimensions, dict)
        assert "nomic-embed-text" in dimensions
        assert "all-minilm" in dimensions
        assert dimensions["nomic-embed-text"] == 768
        assert dimensions["all-minilm"] == 384
        print(f"   Model dimensions: {dimensions}")


@pytest.mark.skipif(not is_openai_available(), reason="openai is not available on this system")
class TestOllamaEmbeddingFunctionPersistence:
    """Test persistence for OllamaEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert OllamaEmbeddingFunction.name() == "ollama"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(OLLAMA_API_KEY="ollama"):
            ef = OllamaEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "nomic-embed-text"
            assert config["api_key_env"] == "OLLAMA_API_KEY"
            assert config["api_base"] == "http://localhost:11434/v1"
            assert config["dimensions"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_OLLAMA_KEY="ollama"):
            ef = OllamaEmbeddingFunction(
                model_name="all-minilm",
                api_key_env="CUSTOM_OLLAMA_KEY",
                api_base="http://remote-server:11434/v1",
                dimensions=512,
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "all-minilm"
            assert config["api_key_env"] == "CUSTOM_OLLAMA_KEY"
            assert config["api_base"] == "http://remote-server:11434/v1"
            assert config["dimensions"] == 512
            assert config["client_kwargs"]["timeout"] == 60

    def test_get_config_with_dimensions(self):
        """Test that get_config() correctly includes dimensions parameter"""
        with env_guard(OLLAMA_API_KEY="ollama"):
            ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", dimensions=256)
            config = ef.get_config()

            assert config["dimensions"] == 256

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "nomic-embed-text",
            "api_key_env": "OLLAMA_API_KEY",
            "api_base": "http://localhost:11434/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with env_guard(OLLAMA_API_KEY="ollama"):
            restored_ef = OllamaEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, OllamaEmbeddingFunction)
            assert restored_ef.model_name == "nomic-embed-text"
            assert restored_ef.api_key_env == "OLLAMA_API_KEY"
            assert restored_ef.api_base == "http://localhost:11434/v1"
            assert restored_ef._dimensions_param is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "all-minilm",
            "api_key_env": "CUSTOM_OLLAMA_KEY",
            "api_base": "http://remote-server:11434/v1",
            "dimensions": 512,
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_OLLAMA_KEY="ollama"):
            restored_ef = OllamaEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, OllamaEmbeddingFunction)
            assert restored_ef.model_name == "all-minilm"
            assert restored_ef.api_key_env == "CUSTOM_OLLAMA_KEY"
            assert restored_ef.api_base == "http://remote-server:11434/v1"
            assert restored_ef._dimensions_param == 512
            assert restored_ef._client_kwargs["timeout"] == 60

    def test_build_from_config_missing_model_name(self):
        """Test that build_from_config() raises ValueError when model_name is missing"""
        config = {
            "api_key_env": "OLLAMA_API_KEY",
            "api_base": "http://localhost:11434/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with pytest.raises(ValueError, match="Missing required field 'model_name'"):
            OllamaEmbeddingFunction.build_from_config(config)

    def test_build_from_config_invalid_client_kwargs(self):
        """Test that build_from_config() raises ValueError when client_kwargs is not a dict"""
        config = {
            "model_name": "nomic-embed-text",
            "api_key_env": "OLLAMA_API_KEY",
            "api_base": "http://localhost:11434/v1",
            "dimensions": None,
            "client_kwargs": "not-a-dict",
        }

        with pytest.raises(ValueError, match="client_kwargs must be a dictionary"):
            OllamaEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(OLLAMA_API_KEY="ollama"):
            original_ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", dimensions=256)

            config = original_ef.get_config()
            restored_ef = OllamaEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, OllamaEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.api_base == original_ef.api_base
            assert restored_ef._dimensions_param == original_ef._dimensions_param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
