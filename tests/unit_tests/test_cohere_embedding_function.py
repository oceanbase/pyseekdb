"""
Unit tests for CohereEmbeddingFunction.

Tests Cohere embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires COHERE_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_cohere_embedding_function.py -v -s
    # Or with environment variable:
    COHERE_API_KEY=your-key pytest tests/unit_tests/test_cohere_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import CohereEmbeddingFunction

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
    not os.environ.get("COHERE_API_KEY") or not is_litellm_available(),
    reason="COHERE_API_KEY environment variable must be set and litellm must be installed",
)
class TestCohereEmbeddingFunction:
    """Test CohereEmbeddingFunction - skipped by default, requires manual execution"""

    def test_cohere_env(self):
        """Test if litellm package is installed and required environment variables are set."""
        assert is_litellm_available(), "litellm package is not installed"

        assert os.environ.get("COHERE_API_KEY"), "COHERE_API_KEY environment variable is not set"

    def test_initialization_with_defaults(self):
        """Test CohereEmbeddingFunction initialization with default values"""
        print("\n✅ Testing CohereEmbeddingFunction initialization with defaults")

        # Check if litellm is available and env vars are set
        self.test_cohere_env()

        ef = CohereEmbeddingFunction()

        assert ef is not None
        assert ef._base_model_name == "embed-english-v3.0"
        assert ef.api_key_env == "COHERE_API_KEY"
        assert ef.input_type is None
        print(f"   Model name: {ef._base_model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   Input type: {ef.input_type}")

    def test_initialization_with_different_models(self):
        """Test CohereEmbeddingFunction initialization with different models"""
        print("\n✅ Testing CohereEmbeddingFunction initialization with different models")

        self.test_cohere_env()

        models = [
            "embed-v4.0",
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-light-v3.0",
            "embed-english-v2.0",
            "embed-multilingual-v2.0",
            "embed-english-light-v2.0",
            "embed-multilingual-light-v2.0",
        ]

        for model in models:
            ef = CohereEmbeddingFunction(model_name=model)
            assert ef._base_model_name == model
            assert ef.api_key_env == "COHERE_API_KEY"
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_custom_api_key_env(self):
        """Test CohereEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing CohereEmbeddingFunction initialization with custom API key env")

        self.test_cohere_env()

        custom_key_env = "CUSTOM_COHERE_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = os.environ.get("COHERE_API_KEY", "your-custom-key")

        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_input_type(self):
        """Test CohereEmbeddingFunction initialization with input_type parameter"""
        print("\n✅ Testing CohereEmbeddingFunction initialization with input_type")

        self.test_cohere_env()

        # Test with search_document
        ef_doc = CohereEmbeddingFunction(model_name="embed-english-v3.0", input_type="search_document")
        assert ef_doc.input_type == "search_document"
        print(f"   Input type (search_document): {ef_doc.input_type}")

        # Test with search_query
        ef_query = CohereEmbeddingFunction(model_name="embed-english-v3.0", input_type="search_query")
        assert ef_query.input_type == "search_query"
        print(f"   Input type (search_query): {ef_query.input_type}")

        # Test with None (default)
        ef_none = CohereEmbeddingFunction(model_name="embed-english-v3.0", input_type=None)
        assert ef_none.input_type is None
        print(f"   Input type (None): {ef_none.input_type}")

    def test_initialization_with_kwargs(self):
        """Test CohereEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing CohereEmbeddingFunction initialization with kwargs")

        self.test_cohere_env()

        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0", timeout=30, max_retries=3)
        assert ef is not None
        print("   Initialized with timeout and max_retries")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing CohereEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("COHERE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key environment variable"):
                CohereEmbeddingFunction(model_name="embed-english-v3.0")
        finally:
            # Restore API key
            if original_key:
                os.environ["COHERE_API_KEY"] = original_key

    def test_dimension_property_known_models(self):
        """Test dimension property for known Cohere models"""
        print("\n✅ Testing CohereEmbeddingFunction dimension property for known models")

        self.test_cohere_env()

        # Test embed-v4.0 (1536 dimensions)
        ef_v4 = CohereEmbeddingFunction(model_name="embed-v4.0")
        dim_v4 = ef_v4.dimension
        assert dim_v4 == 1536, f"Expected dimension 1536 for embed-v4.0, got {dim_v4}"
        print(f"   embed-v4.0 dimension: {dim_v4}")

        # Test embed-english-v3.0 (1024 dimensions)
        ef_en_v3 = CohereEmbeddingFunction(model_name="embed-english-v3.0")
        dim_en_v3 = ef_en_v3.dimension
        assert dim_en_v3 == 1024, f"Expected dimension 1024 for embed-english-v3.0, got {dim_en_v3}"
        print(f"   embed-english-v3.0 dimension: {dim_en_v3}")

        # Test embed-multilingual-v3.0 (1024 dimensions)
        ef_multi_v3 = CohereEmbeddingFunction(model_name="embed-multilingual-v3.0")
        dim_multi_v3 = ef_multi_v3.dimension
        assert dim_multi_v3 == 1024, f"Expected dimension 1024 for embed-multilingual-v3.0, got {dim_multi_v3}"
        print(f"   embed-multilingual-v3.0 dimension: {dim_multi_v3}")

        # Test embed-english-light-v3.0 (384 dimensions)
        ef_en_light_v3 = CohereEmbeddingFunction(model_name="embed-english-light-v3.0")
        dim_en_light_v3 = ef_en_light_v3.dimension
        assert dim_en_light_v3 == 384, f"Expected dimension 384 for embed-english-light-v3.0, got {dim_en_light_v3}"
        print(f"   embed-english-light-v3.0 dimension: {dim_en_light_v3}")

        # Test embed-multilingual-light-v3.0 (384 dimensions)
        ef_multi_light_v3 = CohereEmbeddingFunction(model_name="embed-multilingual-light-v3.0")
        dim_multi_light_v3 = ef_multi_light_v3.dimension
        assert dim_multi_light_v3 == 384, (
            f"Expected dimension 384 for embed-multilingual-light-v3.0, got {dim_multi_light_v3}"
        )
        print(f"   embed-multilingual-light-v3.0 dimension: {dim_multi_light_v3}")

        # Test embed-english-v2.0 (4096 dimensions)
        ef_en_v2 = CohereEmbeddingFunction(model_name="embed-english-v2.0")
        dim_en_v2 = ef_en_v2.dimension
        assert dim_en_v2 == 4096, f"Expected dimension 4096 for embed-english-v2.0, got {dim_en_v2}"
        print(f"   embed-english-v2.0 dimension: {dim_en_v2}")

        # Test embed-multilingual-v2.0 (768 dimensions)
        ef_multi_v2 = CohereEmbeddingFunction(model_name="embed-multilingual-v2.0")
        dim_multi_v2 = ef_multi_v2.dimension
        assert dim_multi_v2 == 768, f"Expected dimension 768 for embed-multilingual-v2.0, got {dim_multi_v2}"
        print(f"   embed-multilingual-v2.0 dimension: {dim_multi_v2}")

        # Test embed-english-light-v2.0 (1024 dimensions)
        ef_en_light_v2 = CohereEmbeddingFunction(model_name="embed-english-light-v2.0")
        dim_en_light_v2 = ef_en_light_v2.dimension
        assert dim_en_light_v2 == 1024, f"Expected dimension 1024 for embed-english-light-v2.0, got {dim_en_light_v2}"
        print(f"   embed-english-light-v2.0 dimension: {dim_en_light_v2}")

        # Test embed-multilingual-light-v2.0 (384 dimensions)
        ef_multi_light_v2 = CohereEmbeddingFunction(model_name="embed-multilingual-light-v2.0")
        dim_multi_light_v2 = ef_multi_light_v2.dimension
        assert dim_multi_light_v2 == 384, (
            f"Expected dimension 384 for embed-multilingual-light-v2.0, got {dim_multi_light_v2}"
        )
        print(f"   embed-multilingual-light-v2.0 dimension: {dim_multi_light_v2}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing CohereEmbeddingFunction dimension property for unknown model")

        self.test_cohere_env()

        # This will make an actual API call to get dimension
        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Unknown model dimension (via API call): {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing CohereEmbeddingFunction embedding generation (single document)")

        self.test_cohere_env()

        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing CohereEmbeddingFunction embedding generation (multiple documents)")

        self.test_cohere_env()

        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0")
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
        print("\n✅ Testing CohereEmbeddingFunction with empty input")

        self.test_cohere_env()

        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_input_type(self):
        """Test __call__ with input_type parameter"""
        print("\n✅ Testing CohereEmbeddingFunction with input_type parameter")

        self.test_cohere_env()

        # Test with search_document
        ef_doc = CohereEmbeddingFunction(model_name="embed-english-v3.0", input_type="search_document")
        test_doc = "This is a document to be indexed"
        embeddings_doc = ef_doc(test_doc)

        assert len(embeddings_doc) == 1
        assert len(embeddings_doc[0]) > 0
        print(f"   Verified: embeddings with search_document have {len(embeddings_doc[0])} dimensions")

        # Test with search_query
        ef_query = CohereEmbeddingFunction(model_name="embed-english-v3.0", input_type="search_query")
        test_query = "What is machine learning?"
        embeddings_query = ef_query(test_query)

        assert len(embeddings_query) == 1
        assert len(embeddings_query[0]) > 0
        print(f"   Verified: embeddings with search_query have {len(embeddings_query[0])} dimensions")

    def test_dimension_of_function(self):
        """Test dimension_of function with CohereEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with CohereEmbeddingFunction")

        self.test_cohere_env()

        ef = CohereEmbeddingFunction(model_name="embed-english-v3.0")
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of result for embed-english-v3.0: {dim}")

        ef_v4 = CohereEmbeddingFunction(model_name="embed-v4.0")
        dim_v4 = dimension_of(ef_v4)
        assert dim_v4 == 1536
        print(f"   dimension_of result for embed-v4.0: {dim_v4}")

        ef_en_light = CohereEmbeddingFunction(model_name="embed-english-light-v3.0")
        dim_en_light = dimension_of(ef_en_light)
        assert dim_en_light == 384
        print(f"   dimension_of result for embed-english-light-v3.0: {dim_en_light}")


@pytest.mark.skipif(not is_litellm_available(), reason="litellm is not available on this system")
class TestCohereEmbeddingFunctionPersistence:
    """Test persistence for CohereEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert CohereEmbeddingFunction.name() == "cohere"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(COHERE_API_KEY="test-key"):
            ef = CohereEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "embed-english-v3.0"
            assert config["api_key_env"] == "COHERE_API_KEY"
            assert config["input_type"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_COHERE_KEY="test-key"):
            ef = CohereEmbeddingFunction(
                model_name="embed-multilingual-v3.0",
                api_key_env="CUSTOM_COHERE_KEY",
                input_type="search_document",
                timeout=60,
            )
            config = ef.get_config()

            assert config["model_name"] == "embed-multilingual-v3.0"
            assert config["api_key_env"] == "CUSTOM_COHERE_KEY"
            assert config["input_type"] == "search_document"
            assert config["client_kwargs"]["timeout"] == 60

    def test_get_config_with_input_type(self):
        """Test that get_config() correctly includes input_type parameter"""
        with env_guard(COHERE_API_KEY="test-key"):
            ef = CohereEmbeddingFunction(model_name="embed-english-v3.0", input_type="search_query")
            config = ef.get_config()

            assert config["input_type"] == "search_query"

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "embed-english-v3.0",
            "api_key_env": "COHERE_API_KEY",
            "input_type": None,
            "client_kwargs": {},
        }

        with env_guard(COHERE_API_KEY="test-key"):
            restored_ef = CohereEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, CohereEmbeddingFunction)
            assert restored_ef._base_model_name == "embed-english-v3.0"
            assert restored_ef.api_key_env == "COHERE_API_KEY"
            assert restored_ef.input_type is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "embed-multilingual-v3.0",
            "api_key_env": "CUSTOM_COHERE_KEY",
            "input_type": "search_document",
            "client_kwargs": {"timeout": 60},
        }

        with env_guard(CUSTOM_COHERE_KEY="test-key"):
            restored_ef = CohereEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, CohereEmbeddingFunction)
            assert restored_ef._base_model_name == "embed-multilingual-v3.0"
            assert restored_ef.api_key_env == "CUSTOM_COHERE_KEY"
            assert restored_ef.input_type == "search_document"
            assert restored_ef._client_kwargs["timeout"] == 60

    def test_build_from_config_with_input_type(self):
        """Test that build_from_config() correctly restores input_type"""
        config = {
            "model_name": "embed-english-v3.0",
            "api_key_env": "COHERE_API_KEY",
            "input_type": "search_query",
            "client_kwargs": {},
        }

        with env_guard(COHERE_API_KEY="test-key"):
            restored_ef = CohereEmbeddingFunction.build_from_config(config)

            assert restored_ef.input_type == "search_query"

    def test_build_from_config_invalid_kwargs(self):
        """Test that build_from_config() raises TypeError when kwargs is not a dict"""
        config = {
            "model_name": "embed-english-v3.0",
            "api_key_env": "COHERE_API_KEY",
            "input_type": None,
            "client_kwargs": "not-a-dict",
        }

        with pytest.raises(TypeError, match="kwargs must be a dictionary"):
            CohereEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(COHERE_API_KEY="test-key"):
            original_ef = CohereEmbeddingFunction(model_name="embed-multilingual-v3.0", input_type="search_document")

            config = original_ef.get_config()
            restored_ef = CohereEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, CohereEmbeddingFunction)
            assert restored_ef._base_model_name == original_ef._base_model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.input_type == original_ef.input_type


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
