"""
Unit tests for JinaEmbeddingFunction.

Tests Jina AI embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires JINA_AI_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_jina_embedding_function.py -v -s
    # Or with environment variable:
    JINA_AI_API_KEY=your-key pytest tests/unit_tests/test_jina_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import JinaEmbeddingFunction

from .test_utils import env_guard


def is_litellm_available() -> bool:
    """
    Check if litellm is available for testing.

    Returns:
        True if litellm is available, False otherwise.
    """
    return importlib.util.find_spec("litellm") is not None


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(
    not os.environ.get("JINA_AI_API_KEY") or not is_litellm_available(),
    reason="JINA_AI_API_KEY environment variable must be set and litellm must be installed",
)
class TestJinaEmbeddingFunction:
    """Test JinaEmbeddingFunction - skipped by default, requires manual execution"""

    def test_jina_env(self):
        """Test if litellm package is installed and required environment variables are set."""
        assert is_litellm_available(), "litellm package is not installed"

        assert os.environ.get("JINA_AI_API_KEY"), "JINA_AI_API_KEY environment variable is not set"

    def test_initialization_with_defaults(self):
        """Test JinaEmbeddingFunction initialization with default values"""
        print("\n✅ Testing JinaEmbeddingFunction initialization with defaults")

        # Check if litellm is available and env vars are set
        self.test_jina_env()

        ef = JinaEmbeddingFunction()

        assert ef is not None
        assert ef._base_model_name == "jina-embeddings-v3"
        assert ef.api_key_env == "JINA_AI_API_KEY"
        print(f"   Model name: {ef._base_model_name}")
        print(f"   API key env: {ef.api_key_env}")

    def test_initialization_with_different_models(self):
        """Test JinaEmbeddingFunction initialization with different models"""
        print("\n✅ Testing JinaEmbeddingFunction initialization with different models")

        self.test_jina_env()

        models = [
            "jina-embeddings-v3",
            "jina-embeddings-v4",
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-base-multilingual",
            "jina-embeddings-v2-small-en",
        ]

        for model in models:
            ef = JinaEmbeddingFunction(model_name=model)
            assert ef._base_model_name == model
            assert ef.api_key_env == "JINA_AI_API_KEY"
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_custom_api_key_env(self):
        """Test JinaEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing JinaEmbeddingFunction initialization with custom API key env")

        self.test_jina_env()

        custom_key_env = "CUSTOM_JINA_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = "your-custom-key"

        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_kwargs(self):
        """Test JinaEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing JinaEmbeddingFunction initialization with kwargs")

        self.test_jina_env()

        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3", timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing JinaEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("JINA_AI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                JinaEmbeddingFunction(model_name="jina-embeddings-v3")
        finally:
            # Restore API key
            if original_key:
                os.environ["JINA_AI_API_KEY"] = original_key

    def test_dimension_property_known_models(self):
        """Test dimension property for known Jina models"""
        print("\n✅ Testing JinaEmbeddingFunction dimension property for known models")

        self.test_jina_env()

        # Test v3 (1024 dimensions)
        ef_v3 = JinaEmbeddingFunction(model_name="jina-embeddings-v3")
        dim_v3 = ef_v3.dimension
        assert dim_v3 == 1024, f"Expected dimension 1024 for jina-embeddings-v3, got {dim_v3}"
        print(f"   jina-embeddings-v3 dimension: {dim_v3}")

        # Test v4 (2048 dimensions)
        ef_v4 = JinaEmbeddingFunction(model_name="jina-embeddings-v4")
        dim_v4 = ef_v4.dimension
        assert dim_v4 == 2048, f"Expected dimension 2048 for jina-embeddings-v4, got {dim_v4}"
        print(f"   jina-embeddings-v4 dimension: {dim_v4}")

        # Test v2-base-en (768 dimensions)
        ef_v2_base = JinaEmbeddingFunction(model_name="jina-embeddings-v2-base-en")
        dim_v2_base = ef_v2_base.dimension
        assert dim_v2_base == 768, f"Expected dimension 768 for jina-embeddings-v2-base-en, got {dim_v2_base}"
        print(f"   jina-embeddings-v2-base-en dimension: {dim_v2_base}")

        # Test v2-small-en (512 dimensions)
        ef_v2_small = JinaEmbeddingFunction(model_name="jina-embeddings-v2-small-en")
        dim_v2_small = ef_v2_small.dimension
        assert dim_v2_small == 512, f"Expected dimension 512 for jina-embeddings-v2-small-en, got {dim_v2_small}"
        print(f"   jina-embeddings-v2-small-en dimension: {dim_v2_small}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing JinaEmbeddingFunction dimension property for unknown model")

        self.test_jina_env()

        # This will make an actual API call to get dimension
        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Unknown model dimension (via API call): {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing JinaEmbeddingFunction embedding generation (single document)")

        self.test_jina_env()

        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing JinaEmbeddingFunction embedding generation (multiple documents)")

        self.test_jina_env()

        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3")
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
        print("\n✅ Testing JinaEmbeddingFunction with empty input")

        self.test_jina_env()

        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_dimension_of_function(self):
        """Test dimension_of function with JinaEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with JinaEmbeddingFunction")

        self.test_jina_env()

        ef = JinaEmbeddingFunction(model_name="jina-embeddings-v3")
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of result for v3: {dim}")

        ef_v4 = JinaEmbeddingFunction(model_name="jina-embeddings-v4")
        dim_v4 = dimension_of(ef_v4)
        assert dim_v4 == 2048
        print(f"   dimension_of result for v4: {dim_v4}")


@pytest.mark.skipif(not is_litellm_available(), reason="litellm is not available on this system")
class TestJinaEmbeddingFunctionPersistence:
    """Test persistence for JinaEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert JinaEmbeddingFunction.name() == "jina"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(JINA_AI_API_KEY="test-key"):
            ef = JinaEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "jina-embeddings-v3"
            assert config["api_key_env"] == "JINA_AI_API_KEY"
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_JINA_KEY="test-key"):
            ef = JinaEmbeddingFunction(
                model_name="jina-embeddings-v4",
                api_key_env="CUSTOM_JINA_KEY",
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "jina-embeddings-v4"
            assert config["api_key_env"] == "CUSTOM_JINA_KEY"
            assert isinstance(config["client_kwargs"], dict)

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "jina-embeddings-v3",
            "api_key_env": "JINA_AI_API_KEY",
            "client_kwargs": {},
        }

        with env_guard(JINA_AI_API_KEY="test-key"):
            restored_ef = JinaEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, JinaEmbeddingFunction)
            assert restored_ef._base_model_name == "jina-embeddings-v3"
            assert restored_ef.api_key_env == "JINA_AI_API_KEY"

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "jina-embeddings-v4",
            "api_key_env": "CUSTOM_JINA_KEY",
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_JINA_KEY="test-key"):
            restored_ef = JinaEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, JinaEmbeddingFunction)
            assert restored_ef._base_model_name == "jina-embeddings-v4"
            assert restored_ef.api_key_env == "CUSTOM_JINA_KEY"

    def test_build_from_config_missing_model_name(self):
        """Test that build_from_config() raises ValueError when model_name is missing"""
        config = {
            "api_key_env": "JINA_AI_API_KEY",
            "client_kwargs": {},
        }

        with pytest.raises(ValueError, match="Missing required field 'model_name'"):
            JinaEmbeddingFunction.build_from_config(config)

    def test_build_from_config_invalid_kwargs(self):
        """Test that build_from_config() raises ValueError when kwargs is not a dict"""
        config = {
            "model_name": "jina-embeddings-v3",
            "api_key_env": "JINA_AI_API_KEY",
            "client_kwargs": "not-a-dict",
        }

        with pytest.raises(ValueError, match="kwargs must be a dictionary"):
            JinaEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(JINA_AI_API_KEY="test-key"):
            original_ef = JinaEmbeddingFunction(model_name="jina-embeddings-v4", api_key_env="JINA_AI_API_KEY")

            config = original_ef.get_config()
            restored_ef = JinaEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, JinaEmbeddingFunction)
            assert restored_ef._base_model_name == original_ef._base_model_name
            assert restored_ef.api_key_env == original_ef.api_key_env


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
