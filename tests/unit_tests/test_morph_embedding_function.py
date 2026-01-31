"""
Unit tests for MorphEmbeddingFunction.

Tests Morph embedding function initialization, embedding generation, and config handling.
Uses real API calls - requires MORPH_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_morph_embedding_function.py -v -s
    # Or with environment variable:
    MORPH_API_KEY=your-key pytest tests/unit_tests/test_morph_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import MorphEmbeddingFunction

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
    not os.environ.get("MORPH_API_KEY") or not is_openai_available(),
    reason="MORPH_API_KEY environment variable must be set",
)
class TestMorphEmbeddingFunction:
    """Test MorphEmbeddingFunction - skipped by default, requires manual execution"""

    def test_morph_env(self):
        """Test if openai package is installed and required environment variables are set."""
        if not is_openai_available():
            print("openai package is not installed")
            raise AssertionError("openai package is not installed")

        if not os.environ.get("MORPH_API_KEY"):
            print("MORPH_API_KEY environment variable is not set")
            raise AssertionError("MORPH_API_KEY environment variable is not set")

    def test_initialization_with_model_name(self):
        """Test MorphEmbeddingFunction initialization with required model_name"""
        print("\nTesting MorphEmbeddingFunction initialization with required model_name")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")

        assert ef is not None
        assert ef.model_name == "morph-embedding-v4"
        assert ef.api_key_env == "MORPH_API_KEY"
        assert ef.api_base == "https://api.morphllm.com/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_custom_api_key_env(self):
        """Test MorphEmbeddingFunction initialization with custom API key env"""
        print("\nTesting MorphEmbeddingFunction initialization with custom API key env")

        self.test_morph_env()

        custom_key_env = "CUSTOM_MORPH_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = "your-custom-key"

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test MorphEmbeddingFunction initialization with custom API base"""
        print("\nTesting MorphEmbeddingFunction initialization with custom API base")

        self.test_morph_env()

        custom_base = "https://api.morphllm.com/v1"
        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4", api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_kwargs(self):
        """Test MorphEmbeddingFunction initialization with additional kwargs"""
        print("\nTesting MorphEmbeddingFunction initialization with kwargs")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4", timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\nTesting MorphEmbeddingFunction initialization with missing API key")

        original_key = os.environ.pop("MORPH_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                MorphEmbeddingFunction(model_name="morph-embedding-v4")
        finally:
            if original_key:
                os.environ["MORPH_API_KEY"] = original_key

    def test_initialization_missing_model_name(self):
        """Test that missing model_name raises TypeError"""
        print("\nTesting MorphEmbeddingFunction initialization with missing model_name")

        self.test_morph_env()

        with pytest.raises(TypeError):
            MorphEmbeddingFunction()

    def test_dimension_property_known_model(self):
        """Test dimension property for known Morph model"""
        print("\nTesting MorphEmbeddingFunction dimension property for known model")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        dim = ef.dimension
        assert dim == 1536, f"Expected dimension 1536 for morph-embedding-v4, got {dim}"
        print(f"   morph-embedding-v4 dimension: {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\nTesting MorphEmbeddingFunction embedding generation (single document)")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        single_doc = "def add(a, b): return a + b"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\nTesting MorphEmbeddingFunction embedding generation (multiple documents)")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        multiple_docs = [
            "def foo():\n    return 1",
            "class Bar:\n    pass",
            "print('hello')",
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
        print("\nTesting MorphEmbeddingFunction with empty input")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_dimension_of_function(self):
        """Test dimension_of function with MorphEmbeddingFunction"""
        print("\nTesting dimension_of function with MorphEmbeddingFunction")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        dim = dimension_of(ef)
        assert dim == 1536
        print(f"   dimension_of result: {dim}")

    def test_get_default_api_base(self):
        """Test _get_default_api_base method"""
        print("\nTesting _get_default_api_base method")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        api_base = ef._get_default_api_base()
        assert api_base == "https://api.morphllm.com/v1"
        print(f"   Default API base: {api_base}")

    def test_get_default_api_key_env(self):
        """Test _get_default_api_key_env method"""
        print("\nTesting _get_default_api_key_env method")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        api_key_env = ef._get_default_api_key_env()
        assert api_key_env == "MORPH_API_KEY"
        print(f"   Default API key env: {api_key_env}")

    def test_get_model_dimensions(self):
        """Test _get_model_dimensions method"""
        print("\nTesting _get_model_dimensions method")

        self.test_morph_env()

        ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
        dimensions = ef._get_model_dimensions()

        assert isinstance(dimensions, dict)
        assert "morph-embedding-v4" in dimensions
        assert dimensions["morph-embedding-v4"] == 1536
        print(f"   Model dimensions: {dimensions}")


@pytest.mark.skipif(not is_openai_available(), reason="openai is not available on this system")
class TestMorphEmbeddingFunctionPersistence:
    """Test persistence for MorphEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert MorphEmbeddingFunction.name() == "morph"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(MORPH_API_KEY="test-key"):
            ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "morph-embedding-v4"
            assert config["api_key_env"] == "MORPH_API_KEY"
            assert config["api_base"] == "https://api.morphllm.com/v1"
            assert config["dimensions"] is None
            assert isinstance(config["client_kwargs"], dict)
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_MORPH_KEY="test-key"):
            ef = MorphEmbeddingFunction(
                model_name="morph-embedding-v4",
                api_key_env="CUSTOM_MORPH_KEY",
                api_base="https://custom-morph.example.com/v1",
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "morph-embedding-v4"
            assert config["api_key_env"] == "CUSTOM_MORPH_KEY"
            assert config["api_base"] == "https://custom-morph.example.com/v1"
            assert config["dimensions"] is None
            assert config["client_kwargs"]["timeout"] == 60

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "morph-embedding-v4",
            "api_key_env": "MORPH_API_KEY",
            "api_base": "https://api.morphllm.com/v1",
            "dimensions": None,
            "client_kwargs": {},
        }

        with env_guard(MORPH_API_KEY="test-key"):
            restored_ef = MorphEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, MorphEmbeddingFunction)
            assert restored_ef.model_name == "morph-embedding-v4"
            assert restored_ef.api_key_env == "MORPH_API_KEY"
            assert restored_ef.api_base == "https://api.morphllm.com/v1"
            assert restored_ef._dimensions_param is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "morph-embedding-v4",
            "api_key_env": "CUSTOM_MORPH_KEY",
            "api_base": "https://custom-morph.example.com/v1",
            "dimensions": None,
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_MORPH_KEY="test-key"):
            restored_ef = MorphEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, MorphEmbeddingFunction)
            assert restored_ef.model_name == "morph-embedding-v4"
            assert restored_ef.api_key_env == "CUSTOM_MORPH_KEY"
            assert restored_ef.api_base == "https://custom-morph.example.com/v1"
            assert restored_ef._dimensions_param is None
            assert restored_ef._client_kwargs["timeout"] == 60

    def test_build_from_config_with_dimensions_ignored(self, caplog):
        """Test that build_from_config() ignores dimensions parameter"""
        config = {
            "model_name": "morph-embedding-v4",
            "api_key_env": "MORPH_API_KEY",
            "api_base": "https://api.morphllm.com/v1",
            "dimensions": 1536,
            "client_kwargs": {},
        }

        with env_guard(MORPH_API_KEY="test-key"):
            restored_ef = MorphEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, MorphEmbeddingFunction)
        assert restored_ef._dimensions_param is None

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(MORPH_API_KEY="test-key"):
            original_ef = MorphEmbeddingFunction(model_name="morph-embedding-v4")

            config = original_ef.get_config()
            restored_ef = MorphEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, MorphEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.api_base == original_ef.api_base
            assert restored_ef._dimensions_param == original_ef._dimensions_param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
