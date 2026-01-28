"""
Unit tests for MistralEmbeddingFunction.

Tests Mistral embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires MISTRAL_API_KEY environment variable to be set.

To run this test manually:
    pytest tests/unit_tests/test_mistral_embedding_function.py -v -s
    # Or with environment variable:
    MISTRAL_API_KEY=your-key pytest tests/unit_tests/test_mistral_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import MistralEmbeddingFunction

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
    not os.environ.get("MISTRAL_API_KEY") or not is_openai_available(),
    reason="MISTRAL_API_KEY environment variable must be set",
)
class TestMistralEmbeddingFunction:
    """Test MistralEmbeddingFunction - skipped by default, requires manual execution"""

    def test_mistral_env(self):
        """Test if openai package is installed and required environment variables are set."""
        if not is_openai_available():
            print("openai package is not installed")
            raise AssertionError("openai package is not installed")

        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY environment variable is not set")
            raise AssertionError("MISTRAL_API_KEY environment variable is not set")

    def test_initialization_with_defaults(self):
        """Test MistralEmbeddingFunction initialization with default values"""
        print("\nTesting MistralEmbeddingFunction initialization with defaults")

        # Check if openai is available and env vars are set
        self.test_mistral_env()

        ef = MistralEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "mistral-embed"
        assert ef.api_key_env == "MISTRAL_API_KEY"
        assert ef.api_base == "https://api.mistral.ai/v1"
        assert ef._dimensions_param is None
        print(f"   Model name: {ef.model_name}")
        print(f"   API key env: {ef.api_key_env}")
        print(f"   API base: {ef.api_base}")

    def test_initialization_with_custom_api_key_env(self):
        """Test MistralEmbeddingFunction initialization with custom API key env"""
        print("\nTesting MistralEmbeddingFunction initialization with custom API key env")

        self.test_mistral_env()

        custom_key_env = "CUSTOM_MISTRAL_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = "your-custom-key"

        ef = MistralEmbeddingFunction(model_name="mistral-embed", api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_custom_api_base(self):
        """Test MistralEmbeddingFunction initialization with custom API base"""
        print("\nTesting MistralEmbeddingFunction initialization with custom API base")

        self.test_mistral_env()

        custom_base = "https://api.mistral.ai/v1"
        ef = MistralEmbeddingFunction(model_name="mistral-embed", api_base=custom_base)
        assert ef.api_base == custom_base
        print(f"   Custom API base: {ef.api_base}")

    def test_initialization_with_additional_kwargs(self):
        """Test MistralEmbeddingFunction initialization with additional kwargs"""
        print("\nTesting MistralEmbeddingFunction initialization with kwargs")

        self.test_mistral_env()

        ef = MistralEmbeddingFunction(model_name="mistral-embed", timeout=30, max_retries=3)
        assert ef._client_kwargs.get("timeout") == 30
        assert ef._client_kwargs.get("max_retries") == 3
        print(f"   Client kwargs: {ef._client_kwargs}")

    def test_initialization_with_missing_api_key(self):
        """Test MistralEmbeddingFunction initialization with missing API key"""
        print("\nTesting MistralEmbeddingFunction initialization with missing API key")

        with env_guard(MISTRAL_API_KEY=None):
            try:
                MistralEmbeddingFunction(model_name="mistral-embed")
                raise AssertionError("Expected ValueError for missing API key")
            except ValueError as e:
                print(f"   Caught expected error: {e}")

    def test_dimension_property_for_known_model(self):
        """Test MistralEmbeddingFunction dimension property for known model"""
        print("\nTesting MistralEmbeddingFunction dimension property for known model")

        self.test_mistral_env()

        ef = MistralEmbeddingFunction(model_name="mistral-embed")
        assert ef.dimension == 1024
        print(f"   Model mistral-embed dimension: {ef.dimension}")

    def test_embedding_generation_single_document(self):
        """Test MistralEmbeddingFunction embedding generation (single document)"""
        print("\nTesting MistralEmbeddingFunction embedding generation (single document)")

        self.test_mistral_env()

        ef = MistralEmbeddingFunction(model_name="mistral-embed")
        embeddings = ef("Hello world")
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
        print(f"   Embedding dimension: {len(embeddings[0])}")

    def test_embedding_generation_multiple_documents(self):
        """Test MistralEmbeddingFunction embedding generation (multiple documents)"""
        print("\nTesting MistralEmbeddingFunction embedding generation (multiple documents)")

        self.test_mistral_env()

        ef = MistralEmbeddingFunction(model_name="mistral-embed")
        embeddings = ef(["Hello world", "How are you?"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        print(f"   Embedding dimension: {len(embeddings[0])}")

    def test_embedding_with_empty_input(self):
        """Test MistralEmbeddingFunction with empty input"""
        print("\nTesting MistralEmbeddingFunction with empty input")

        self.test_mistral_env()

        ef = MistralEmbeddingFunction(model_name="mistral-embed")
        embeddings = ef([])
        assert embeddings == []
        print("   Empty input returned empty embeddings")

    def test_dimension_of_function(self):
        """Test dimension_of function with MistralEmbeddingFunction"""
        print("\nTesting dimension_of function with MistralEmbeddingFunction")

        self.test_mistral_env()

        ef = MistralEmbeddingFunction(model_name="mistral-embed")
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of returned: {dim}")

    def test_persistence(self):
        """Test persistence for MistralEmbeddingFunction"""
        print("\nTesting MistralEmbeddingFunction persistence")

        self.test_mistral_env()

        assert MistralEmbeddingFunction.name() == "mistral"

        ef = MistralEmbeddingFunction(model_name="mistral-embed")
        config = ef.get_config()

        restored_ef = MistralEmbeddingFunction.build_from_config(config)
        assert isinstance(restored_ef, MistralEmbeddingFunction)
        assert restored_ef.model_name == ef.model_name
        assert restored_ef.api_key_env == ef.api_key_env
        assert restored_ef.api_base == ef.api_base
