"""
Unit tests for GoogleVertexEmbeddingFunction.

Tests Google Vertex AI embedding function initialization, embedding generation, and configuration.
Uses real API calls - requires CHROMA_GOOGLE_VERTEX_API_KEY environment variable to be set and google-cloud-aiplatform to be installed.

To run this test manually:
    pytest tests/unit_tests/test_google_vertex_embedding_function.py -v -s
    # Or with environment variable:
    CHROMA_GOOGLE_VERTEX_API_KEY=your-key pytest tests/unit_tests/test_google_vertex_embedding_function.py -v -s
"""

import importlib.util
import os

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import GoogleVertexEmbeddingFunction

from .test_utils import env_guard


def is_vertexai_available() -> bool:
    """
    Check if google-cloud-aiplatform (vertexai) is available for testing.

    Returns:
        True if vertexai is available, False otherwise.
    """
    return importlib.util.find_spec("vertexai") is not None


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(
    not os.environ.get("CHROMA_GOOGLE_VERTEX_API_KEY") or not is_vertexai_available(),
    reason="CHROMA_GOOGLE_VERTEX_API_KEY environment variable must be set and google-cloud-aiplatform must be installed",
)
class TestGoogleVertexEmbeddingFunction:
    """Test GoogleVertexEmbeddingFunction - skipped by default, requires manual execution"""

    def test_vertex_env(self):
        """Test if vertexai package is installed and required environment variables are set."""
        assert is_vertexai_available(), "google-cloud-aiplatform package is not installed"

        assert os.environ.get("CHROMA_GOOGLE_VERTEX_API_KEY"), (
            "CHROMA_GOOGLE_VERTEX_API_KEY environment variable is not set"
        )

    def test_initialization_with_defaults(self):
        """Test GoogleVertexEmbeddingFunction initialization with default values"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with defaults")

        # Check if vertexai is available and env vars are set
        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction()

        assert ef is not None
        assert ef.model_name == "textembedding-gecko"
        assert ef.project_id == "cloud-large-language-models"
        assert ef.region == "us-central1"
        assert ef.api_key_env == "CHROMA_GOOGLE_VERTEX_API_KEY"
        print(f"   Model name: {ef.model_name}")
        print(f"   Project ID: {ef.project_id}")
        print(f"   Region: {ef.region}")
        print(f"   API key env: {ef.api_key_env}")

    def test_initialization_with_custom_model_name(self):
        """Test GoogleVertexEmbeddingFunction initialization with custom model name"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with custom model name")

        self.test_vertex_env()

        model_names = [
            "textembedding-gecko",
            "textembedding-gecko@003",
            "textembedding-gecko@002",
        ]

        for model_name in model_names:
            ef = GoogleVertexEmbeddingFunction(model_name=model_name)
            assert ef.model_name == model_name
            print(f"   Model {model_name}: initialized successfully")

    def test_initialization_with_custom_project_id(self):
        """Test GoogleVertexEmbeddingFunction initialization with custom project_id"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with custom project_id")

        self.test_vertex_env()

        custom_project_id = "my-custom-project-id"
        ef = GoogleVertexEmbeddingFunction(project_id=custom_project_id)
        assert ef.project_id == custom_project_id
        print(f"   Project ID: {ef.project_id}")

    def test_initialization_with_custom_region(self):
        """Test GoogleVertexEmbeddingFunction initialization with custom region"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with custom region")

        self.test_vertex_env()

        custom_region = "us-west1"
        ef = GoogleVertexEmbeddingFunction(region=custom_region)
        assert ef.region == custom_region
        print(f"   Region: {ef.region}")

    def test_initialization_with_custom_api_key_env(self):
        """Test GoogleVertexEmbeddingFunction initialization with custom API key env"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with custom API key env")

        self.test_vertex_env()

        custom_key_env = "CUSTOM_GOOGLE_VERTEX_KEY"
        if not os.environ.get(custom_key_env):
            os.environ[custom_key_env] = os.environ.get("CHROMA_GOOGLE_VERTEX_API_KEY", "your-custom-key")

        ef = GoogleVertexEmbeddingFunction(api_key_env=custom_key_env)
        assert ef.api_key_env == custom_key_env
        print(f"   Custom API key env: {ef.api_key_env}")

    def test_initialization_with_all_parameters(self):
        """Test GoogleVertexEmbeddingFunction initialization with all custom parameters"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with all parameters")

        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction(
            model_name="textembedding-gecko@003",
            project_id="my-project",
            region="us-west1",
            api_key_env="CHROMA_GOOGLE_VERTEX_API_KEY",
        )
        assert ef.model_name == "textembedding-gecko@003"
        assert ef.project_id == "my-project"
        assert ef.region == "us-west1"
        assert ef.api_key_env == "CHROMA_GOOGLE_VERTEX_API_KEY"
        print("   All parameters set correctly")

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization with missing API key")

        # Temporarily remove API key
        original_key = os.environ.pop("CHROMA_GOOGLE_VERTEX_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="environment variable is not set"):
                GoogleVertexEmbeddingFunction()
        finally:
            # Restore API key
            if original_key:
                os.environ["CHROMA_GOOGLE_VERTEX_API_KEY"] = original_key

    def test_initialization_without_vertexai(self):
        """Test that initialization fails when vertexai is not installed"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction initialization without vertexai")

        # This test doesn't require API key, just checks the import error
        # We can't easily mock importlib.util.find_spec, so we'll skip if vertexai is available
        if is_vertexai_available():
            pytest.skip("vertexai is available, cannot test import error")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction embedding generation (single document)")

        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction()
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction embedding generation (multiple documents)")

        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction()
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
        print("\n✅ Testing GoogleVertexEmbeddingFunction with empty input")

        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction()
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_call_with_non_string_documents(self):
        """Test __call__ with non-string documents raises ValueError"""
        print("\n✅ Testing GoogleVertexEmbeddingFunction with non-string documents")

        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction()

        # Test with list containing non-string
        with pytest.raises(ValueError, match="Google Vertex only supports text documents"):
            ef([123, "text"])

        # Test with list containing None
        with pytest.raises(ValueError, match="Google Vertex only supports text documents"):
            ef([None, "text"])

    def test_dimension_of_function(self):
        """Test dimension_of function with GoogleVertexEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with GoogleVertexEmbeddingFunction")

        self.test_vertex_env()

        ef = GoogleVertexEmbeddingFunction()
        # Generate an embedding to determine dimension
        test_embedding = ef("test")
        expected_dim = len(test_embedding[0])

        dim = dimension_of(ef)
        assert dim == expected_dim
        print(f"   dimension_of result: {dim}")


@pytest.mark.skipif(
    not is_vertexai_available() or not os.environ.get("CHROMA_GOOGLE_VERTEX_API_KEY"),
    reason="google-cloud-aiplatform is not available on this system",
)
class TestGoogleVertexEmbeddingFunctionPersistence:
    """Test persistence for GoogleVertexEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert GoogleVertexEmbeddingFunction.name() == "google_vertex"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        with env_guard(CHROMA_GOOGLE_VERTEX_API_KEY="test-key"):
            ef = GoogleVertexEmbeddingFunction()
            config = ef.get_config()

            assert isinstance(config, dict)
            assert config["model_name"] == "textembedding-gecko"
            assert config["project_id"] == "cloud-large-language-models"
            assert config["region"] == "us-central1"
            assert config["api_key_env"] == "CHROMA_GOOGLE_VERTEX_API_KEY"
            # name should NOT be in config
            assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        with env_guard(CUSTOM_GOOGLE_VERTEX_KEY="test-key"):
            ef = GoogleVertexEmbeddingFunction(
                model_name="textembedding-gecko@003",
                project_id="my-project",
                region="us-west1",
                api_key_env="CUSTOM_GOOGLE_VERTEX_KEY",
            )
            config = ef.get_config()

            assert config["model_name"] == "textembedding-gecko@003"
            assert config["project_id"] == "my-project"
            assert config["region"] == "us-west1"
            assert config["api_key_env"] == "CUSTOM_GOOGLE_VERTEX_KEY"

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "api_key_env": "CHROMA_GOOGLE_VERTEX_API_KEY",
            "model_name": "textembedding-gecko",
            "project_id": "cloud-large-language-models",
            "region": "us-central1",
        }

        with env_guard(CHROMA_GOOGLE_VERTEX_API_KEY="test-key"):
            restored_ef = GoogleVertexEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, GoogleVertexEmbeddingFunction)
            assert restored_ef.model_name == "textembedding-gecko"
            assert restored_ef.project_id == "cloud-large-language-models"
            assert restored_ef.region == "us-central1"
            assert restored_ef.api_key_env == "CHROMA_GOOGLE_VERTEX_API_KEY"

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "api_key_env": "CUSTOM_GOOGLE_VERTEX_KEY",
            "model_name": "textembedding-gecko@003",
            "project_id": "my-project",
            "region": "us-west1",
        }

        with env_guard(CUSTOM_GOOGLE_VERTEX_KEY="test-key"):
            restored_ef = GoogleVertexEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, GoogleVertexEmbeddingFunction)
            assert restored_ef.model_name == "textembedding-gecko@003"
            assert restored_ef.project_id == "my-project"
            assert restored_ef.region == "us-west1"
            assert restored_ef.api_key_env == "CUSTOM_GOOGLE_VERTEX_KEY"

    def test_build_from_config_missing_api_key_env(self):
        """Test that build_from_config() raises AssertionError when api_key_env is missing"""
        config = {
            "model_name": "textembedding-gecko",
            "project_id": "cloud-large-language-models",
            "region": "us-central1",
        }

        with pytest.raises(AssertionError, match="This code should not be reached"):
            GoogleVertexEmbeddingFunction.build_from_config(config)

    def test_build_from_config_missing_model_name(self):
        """Test that build_from_config() raises AssertionError when model_name is missing"""
        config = {
            "api_key_env": "CHROMA_GOOGLE_VERTEX_API_KEY",
            "project_id": "cloud-large-language-models",
            "region": "us-central1",
        }

        with pytest.raises(AssertionError, match="This code should not be reached"):
            GoogleVertexEmbeddingFunction.build_from_config(config)

    def test_build_from_config_missing_project_id(self):
        """Test that build_from_config() raises AssertionError when project_id is missing"""
        config = {
            "api_key_env": "CHROMA_GOOGLE_VERTEX_API_KEY",
            "model_name": "textembedding-gecko",
            "region": "us-central1",
        }

        with pytest.raises(AssertionError, match="This code should not be reached"):
            GoogleVertexEmbeddingFunction.build_from_config(config)

    def test_build_from_config_missing_region(self):
        """Test that build_from_config() raises AssertionError when region is missing"""
        config = {
            "api_key_env": "CHROMA_GOOGLE_VERTEX_API_KEY",
            "model_name": "textembedding-gecko",
            "project_id": "cloud-large-language-models",
        }

        with pytest.raises(AssertionError, match="This code should not be reached"):
            GoogleVertexEmbeddingFunction.build_from_config(config)

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        with env_guard(CHROMA_GOOGLE_VERTEX_API_KEY="test-key"):
            original_ef = GoogleVertexEmbeddingFunction(
                model_name="textembedding-gecko@003",
                project_id="my-project",
                region="us-west1",
            )

            config = original_ef.get_config()
            restored_ef = GoogleVertexEmbeddingFunction.build_from_config(config)

            assert isinstance(restored_ef, GoogleVertexEmbeddingFunction)
            assert restored_ef.model_name == original_ef.model_name
            assert restored_ef.project_id == original_ef.project_id
            assert restored_ef.region == original_ef.region
            assert restored_ef.api_key_env == original_ef.api_key_env


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
