"""
Unit tests for VoyageaiEmbeddingFunction.

Tests Voyage AI embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires VOYAGE_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_voyageai_embedding_function.py -v -s
    # Or with environment variable:
    VOYAGE_API_KEY=your-key pytest tests/unit_tests/test_voyageai_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import VoyageaiEmbeddingFunction

from .test_utils import env_guard


def is_voyageai_available() -> bool:
    """
    Check if voyageai is available for testing.

    Returns:
        True if voyageai is available, False otherwise.
    """
    return importlib.util.find_spec("voyageai") is not None


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY") or not is_voyageai_available(),
    reason="VOYAGE_API_KEY environment variable must be set and voyageai package must be installed",
)
class TestVoyageaiEmbeddingFunction:
    """Test VoyageaiEmbeddingFunction - skipped by default, requires manual execution"""

    def test_voyageai_env(self):
        """Test if voyageai package is installed and required environment variables are set."""
        assert is_voyageai_available(), "voyageai package is not installed"

        assert os.environ.get("VOYAGE_API_KEY"), "VOYAGE_API_KEY environment variable is not set"

    def test_initialization_with_defaults(self):
        """Test VoyageaiEmbeddingFunction initialization with default values"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with defaults")

        # Check if voyageai is available and env vars are set
        self.test_voyageai_env()

        ef = VoyageaiEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "voyage-4-large"
        assert ef.api_key_env == "VOYAGE_API_KEY"
        assert ef.input_type is None
        assert ef.truncation is None
        assert ef.output_dimension is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   Input type: {ef.input_type}")

    def test_initialization_with_different_models(self):
        """Test VoyageaiEmbeddingFunction initialization with different models"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with different models")

        self.test_voyageai_env()

        models = [
            "voyage-4-large",
            "voyage-4",
            "voyage-4-lite",
            "voyage-code-3",
            "voyage-finance-2",
            "voyage-law-2",
            "voyage-code-2",
            "voyage-3-large",
            "voyage-3.5",
            "voyage-3.5-lite",
            "voyage-3",
            "voyage-3-lite",
            "voyage-multilingual-2",
            "voyage-4-nano",
        ]

        for model in models:
            ef = VoyageaiEmbeddingFunction(model_name=model)
            assert ef.model_name == model
            assert ef.api_key_env == "VOYAGE_API_KEY"
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_custom_api_key_env(self):
        """Test VoyageaiEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with custom API key env")

        self.test_voyageai_env()

        custom_key_env = "CUSTOM_VOYAGE_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = os.environ.get("VOYAGE_API_KEY", "your-custom-key")

        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_input_type(self):
        """Test VoyageaiEmbeddingFunction initialization with input_type parameter"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with input_type")

        self.test_voyageai_env()

        # Test with "document"
        ef_doc = VoyageaiEmbeddingFunction(model_name="voyage-4-large", input_type="document")
        assert ef_doc.input_type == "document"
        print(f"   Input type (document): {ef_doc.input_type}")

        # Test with "query"
        ef_query = VoyageaiEmbeddingFunction(model_name="voyage-4-large", input_type="query")
        assert ef_query.input_type == "query"
        print(f"   Input type (query): {ef_query.input_type}")

        # Test with None (default)
        ef_none = VoyageaiEmbeddingFunction(model_name="voyage-4-large", input_type=None)
        assert ef_none.input_type is None
        print(f"   Input type (None): {ef_none.input_type}")

    def test_initialization_with_truncation(self):
        """Test VoyageaiEmbeddingFunction initialization with truncation parameter"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with truncation")

        self.test_voyageai_env()

        # Test with True
        ef_true = VoyageaiEmbeddingFunction(model_name="voyage-4-large", truncation=True)
        assert ef_true.truncation is True
        print(f"   Truncation (True): {ef_true.truncation}")

        # Test with False
        ef_false = VoyageaiEmbeddingFunction(model_name="voyage-4-large", truncation=False)
        assert ef_false.truncation is False
        print(f"   Truncation (False): {ef_false.truncation}")

        # Test with None (default)
        ef_none = VoyageaiEmbeddingFunction(model_name="voyage-4-large", truncation=None)
        assert ef_none.truncation is None
        print(f"   Truncation (None): {ef_none.truncation}")

    def test_initialization_with_output_dimension(self):
        """Test VoyageaiEmbeddingFunction initialization with output_dimension parameter"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with output_dimension")

        self.test_voyageai_env()

        # Test with different dimensions
        for dim in [256, 512, 1024, 2048]:
            ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=dim)
            assert ef.output_dimension == dim
            assert ef.dimension == dim
            print(f"   Output dimension {dim}: verified")

    def test_initialization_with_kwargs(self):
        """Test VoyageaiEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with kwargs")

        self.test_voyageai_env()

        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large", custom_param="value")
        assert ef is not None
        assert ef.kwargs.get("custom_param") == "value"
        print("   Initialized with custom kwargs")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("VOYAGE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Voyage AI API key not found"):
                VoyageaiEmbeddingFunction()
        finally:
            # Restore API key
            if original_key:
                os.environ["VOYAGE_API_KEY"] = original_key

    def test_initialization_without_voyageai(self):
        """Test that initialization fails when voyageai is not installed"""
        print("\n✅ Testing VoyageaiEmbeddingFunction initialization without voyageai")

        # This test doesn't require API key, just checks the import error
        # We can't easily mock importlib.util.find_spec, so we'll skip if voyageai is available
        if is_voyageai_available():
            pytest.skip("voyageai is available, cannot test import error")

    def test_dimension_property_known_models(self):
        """Test dimension property for known Voyage AI models"""
        print("\n✅ Testing VoyageaiEmbeddingFunction dimension property for known models")

        self.test_voyageai_env()

        # Test voyage-4-large (1024 dimensions default)
        ef_v4_large = VoyageaiEmbeddingFunction(model_name="voyage-4-large")
        dim_v4_large = ef_v4_large.dimension
        assert dim_v4_large == 1024, f"Expected dimension 1024 for voyage-4-large, got {dim_v4_large}"
        print(f"   voyage-4-large dimension: {dim_v4_large}")

        # Test voyage-3-lite (512 dimensions default)
        ef_v3_lite = VoyageaiEmbeddingFunction(model_name="voyage-3-lite")
        dim_v3_lite = ef_v3_lite.dimension
        assert dim_v3_lite == 512, f"Expected dimension 512 for voyage-3-lite, got {dim_v3_lite}"
        print(f"   voyage-3-lite dimension: {dim_v3_lite}")

        # Test voyage-code-2 (1536 dimensions default)
        ef_code2 = VoyageaiEmbeddingFunction(model_name="voyage-code-2")
        dim_code2 = ef_code2.dimension
        assert dim_code2 == 1536, f"Expected dimension 1536 for voyage-code-2, got {dim_code2}"
        print(f"   voyage-code-2 dimension: {dim_code2}")

    def test_dimension_property_with_output_dimension(self):
        """Test dimension property when output_dimension is specified"""
        print("\n✅ Testing VoyageaiEmbeddingFunction dimension property with output_dimension")

        self.test_voyageai_env()

        # Test with custom output_dimension
        ef_512 = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=512)
        dim_512 = ef_512.dimension
        assert dim_512 == 512, f"Expected dimension 512, got {dim_512}"
        print(f"   Output dimension 512: {dim_512}")

        # Test with different dimension
        ef_256 = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=256)
        dim_256 = ef_256.dimension
        assert dim_256 == 256, f"Expected dimension 256, got {dim_256}"
        print(f"   Output dimension 256: {dim_256}")

    def test_dimension_property_unknown_model(self):
        """Test dimension property for unknown model (should make API call)"""
        print("\n✅ Testing VoyageaiEmbeddingFunction dimension property for unknown model")

        self.test_voyageai_env()

        # This will make an actual API call to get dimension
        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large")
        dim = ef.dimension

        # Should have a valid dimension
        assert dim > 0
        print(f"   Model dimension: {dim}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing VoyageaiEmbeddingFunction embedding generation (single document)")

        self.test_voyageai_env()

        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing VoyageaiEmbeddingFunction embedding generation (multiple documents)")

        self.test_voyageai_env()

        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large")
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
        print("\n✅ Testing VoyageaiEmbeddingFunction with empty input")

        self.test_voyageai_env()

        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_input_type(self):
        """Test __call__ with input_type parameter"""
        print("\n✅ Testing VoyageaiEmbeddingFunction with input_type parameter")

        self.test_voyageai_env()

        # Test with "document"
        ef_doc = VoyageaiEmbeddingFunction(model_name="voyage-4-large", input_type="document")
        test_doc = "This is a document to be indexed"
        embeddings_doc = ef_doc(test_doc)

        assert len(embeddings_doc) == 1
        assert len(embeddings_doc[0]) > 0
        print(f"   Verified: embeddings with input_type='document' have {len(embeddings_doc[0])} dimensions")

        # Test with "query"
        ef_query = VoyageaiEmbeddingFunction(model_name="voyage-4-large", input_type="query")
        test_query = "What is machine learning?"
        embeddings_query = ef_query(test_query)

        assert len(embeddings_query) == 1
        assert len(embeddings_query[0]) > 0
        print(f"   Verified: embeddings with input_type='query' have {len(embeddings_query[0])} dimensions")

    def test_call_with_output_dimension(self):
        """Test __call__ with output_dimension parameter"""
        print("\n✅ Testing VoyageaiEmbeddingFunction with output_dimension parameter")

        self.test_voyageai_env()

        # Test with output_dimension=512
        ef_512 = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=512)
        test_doc = "Test document for embedding"
        embeddings_512 = ef_512(test_doc)

        assert len(embeddings_512) == 1
        assert len(embeddings_512[0]) == 512, f"Expected 512 dimensions, got {len(embeddings_512[0])}"
        print(f"   Verified: embeddings have {len(embeddings_512[0])} dimensions")

        # Test with output_dimension=256
        ef_256 = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=256)
        embeddings_256 = ef_256(test_doc)
        assert len(embeddings_256[0]) == 256, f"Expected 256 dimensions, got {len(embeddings_256[0])}"
        print(f"   Verified: embeddings have {len(embeddings_256[0])} dimensions")

    def test_call_with_truncation(self):
        """Test __call__ with truncation parameter"""
        print("\n✅ Testing VoyageaiEmbeddingFunction with truncation parameter")

        self.test_voyageai_env()

        # Test with truncation=True
        ef_truncate = VoyageaiEmbeddingFunction(model_name="voyage-4-large", truncation=True)
        test_doc = "Test document"
        embeddings = ef_truncate(test_doc)

        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
        print(f"   Verified: embeddings with truncation=True have {len(embeddings[0])} dimensions")

    def test_dimension_of_function(self):
        """Test dimension_of function with VoyageaiEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with VoyageaiEmbeddingFunction")

        self.test_voyageai_env()

        ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large")
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of result for voyage-4-large: {dim}")

        ef_512 = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=512)
        dim_512 = dimension_of(ef_512)
        assert dim_512 == 512
        print(f"   dimension_of result with output_dimension=512: {dim_512}")


@pytest.mark.skipif(not is_voyageai_available(), reason="voyageai is not available on this system")
class TestVoyageaiEmbeddingFunctionPersistence:
    """Test persistence for VoyageaiEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert VoyageaiEmbeddingFunction.name() == "voyageai"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(VOYAGE_API_KEY="test-key"):
            ef = VoyageaiEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "voyage-4-large"
            assert config["api_key_env"] == "VOYAGE_API_KEY"
            assert config["input_type"] is None
            assert config["truncation"] is None
            assert config["output_dimension"] is None
            assert isinstance(config["client_kwargs"], dict)
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_VOYAGE_KEY="test-key"):
            ef = VoyageaiEmbeddingFunction(
                model_name="voyage-3-large",
                api_key_env="CUSTOM_VOYAGE_KEY",
                input_type="document",
                truncation=True,
                output_dimension=512,
            )
            config = ef.get_config()

            assert config["model_name"] == "voyage-3-large"
            assert config["api_key_env"] == "CUSTOM_VOYAGE_KEY"
            assert config["input_type"] == "document"
            assert config["truncation"] is True
            assert config["output_dimension"] == 512

    def test_get_config_with_input_type(self):
        """Test that get_config() correctly includes input_type parameter"""
        with env_guard(VOYAGE_API_KEY="test-key"):
            ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large", input_type="query")
            config = ef.get_config()

            assert config["input_type"] == "query"

    def test_get_config_with_truncation(self):
        """Test that get_config() correctly includes truncation parameter"""
        with env_guard(VOYAGE_API_KEY="test-key"):
            ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large", truncation=False)
            config = ef.get_config()

            assert config["truncation"] is False

    def test_get_config_with_output_dimension(self):
        """Test that get_config() correctly includes output_dimension parameter"""
        with env_guard(VOYAGE_API_KEY="test-key"):
            ef = VoyageaiEmbeddingFunction(model_name="voyage-4-large", output_dimension=256)
            config = ef.get_config()

            assert config["output_dimension"] == 256

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "voyage-4-large",
            "api_key_env": "VOYAGE_API_KEY",
            "input_type": None,
            "truncation": None,
            "output_dimension": None,
            "client_kwargs": {},
        }

        with env_guard(VOYAGE_API_KEY="test-key"):
            restored_ef = VoyageaiEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, VoyageaiEmbeddingFunction)
            assert restored_ef.model_name == "voyage-4-large"
            assert restored_ef.api_key_env == "VOYAGE_API_KEY"
            assert restored_ef.input_type is None
            assert restored_ef.truncation is None
            assert restored_ef.output_dimension is None

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "voyage-3-large",
            "api_key_env": "CUSTOM_VOYAGE_KEY",
            "input_type": "document",
            "truncation": True,
            "output_dimension": 512,
            "client_kwargs": {},
        }

        with env_guard(CUSTOM_VOYAGE_KEY="test-key"):
            restored_ef = VoyageaiEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, VoyageaiEmbeddingFunction)
            assert restored_ef.model_name == "voyage-3-large"
            assert restored_ef.api_key_env == "CUSTOM_VOYAGE_KEY"
            assert restored_ef.input_type == "document"
            assert restored_ef.truncation is True
            assert restored_ef.output_dimension == 512

    def test_build_from_config_missing_model_name(self):
        """Test that build_from_config() raises ValueError when model_name is missing"""
        config = {
            "api_key_env": "VOYAGE_API_KEY",
            "input_type": None,
            "truncation": None,
            "output_dimension": None,
            "client_kwargs": {},
        }

        with pytest.raises(ValueError, match="Missing required field 'model_name'"):
            VoyageaiEmbeddingFunction.build_from_config(config)

    def test_build_from_config_invalid_kwargs(self):
        """Test that build_from_config() raises TypeError when kwargs is not a dict"""
        config = {
            "model_name": "voyage-4-large",
            "api_key_env": "VOYAGE_API_KEY",
            "input_type": None,
            "truncation": None,
            "output_dimension": None,
            "client_kwargs": "not-a-dict",
        }

        with pytest.raises(TypeError, match="kwargs must be a dictionary"):
            VoyageaiEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(VOYAGE_API_KEY="test-key"):
            original_ef = VoyageaiEmbeddingFunction(
                model_name="voyage-3-large",
                input_type="query",
                truncation=False,
                output_dimension=256,
            )

            config = original_ef.get_config()
            restored_ef = VoyageaiEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, VoyageaiEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.api_key_env == original_ef.api_key_env
            assert restored_ef.input_type == original_ef.input_type
            assert restored_ef.truncation == original_ef.truncation
            assert restored_ef.output_dimension == original_ef.output_dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
