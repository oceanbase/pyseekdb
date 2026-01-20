"""
Unit tests for SentenceTransformerEmbeddingFunction persistence (get_config and build_from_config).

Tests the persistence functionality for SentenceTransformerEmbeddingFunction.
These tests verify that SentenceTransformerEmbeddingFunction can be serialized to config dictionaries
and restored from them correctly.

To run this test:
    pytest tests/unit_tests/test_sentence_transformer_embedding_function_persistence.py -v
"""

import importlib.util

import pytest

from pyseekdb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# # This will work on both CPU and CUDA systems
# pip install sentence-transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


def is_cuda_available() -> bool:
    """
    Check if CUDA is available for testing.

    Returns:
        True if CUDA is available, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def is_sentence_transformers_available() -> bool:
    """
    Check if sentence-transformers is available for testing.

    Returns:
        True if sentence-transformers is available, False otherwise.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not is_sentence_transformers_available(),
    reason="sentence-transformers is not available on this system",
)
class TestSentenceTransformerEmbeddingFunctionPersistence:
    """Test persistence for SentenceTransformerEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert SentenceTransformerEmbeddingFunction.name() == "sentence_transformer"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        ef = SentenceTransformerEmbeddingFunction()
        config = ef.get_config()

        assert isinstance(config, dict)
        assert config["model_name"] == "all-MiniLM-L6-v2"
        assert config["device"] == "cpu"
        assert config["normalize_embeddings"] is False
        assert isinstance(config["kwargs"], dict)
        assert config["kwargs"] == {}
        # name should NOT be in config
        assert "name" not in config

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2",
            device="cuda",
            normalize_embeddings=True,
            trust_remote_code=True,
        )
        config = ef.get_config()

        assert config["model_name"] == "all-mpnet-base-v2"
        assert config["device"] == "cuda"
        assert config["normalize_embeddings"] is True
        assert config["kwargs"]["trust_remote_code"] is True

    def test_get_config_with_kwargs(self):
        """Test that get_config() correctly includes kwargs"""
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            normalize_embeddings=False,
            trust_remote_code=True,
            use_auth_token="test-token",  # noqa: S106
        )
        config = ef.get_config()

        assert config["kwargs"]["trust_remote_code"] is True
        assert config["kwargs"]["use_auth_token"] == "test-token"  # noqa: S105

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""
        config = {
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu",
            "normalize_embeddings": False,
            "kwargs": {},
        }

        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, SentenceTransformerEmbeddingFunction)
        assert restored_ef.model_name == "all-MiniLM-L6-v2"
        assert restored_ef.device == "cpu"
        assert restored_ef.normalize_embeddings is False
        assert restored_ef.kwargs == {}

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""
        config = {
            "model_name": "all-mpnet-base-v2",
            "device": "cuda",
            "normalize_embeddings": True,
            "kwargs": {"trust_remote_code": True},
        }

        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, SentenceTransformerEmbeddingFunction)
        assert restored_ef.model_name == "all-mpnet-base-v2"
        assert restored_ef.device == "cuda"
        assert restored_ef.normalize_embeddings is True
        assert restored_ef.kwargs["trust_remote_code"] is True

    def test_build_from_config_with_missing_fields_uses_defaults(self):
        """Test that build_from_config() uses defaults for missing fields"""
        # Test with minimal config
        config = {"model_name": "all-MiniLM-L6-v2"}

        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, SentenceTransformerEmbeddingFunction)
        assert restored_ef.model_name == "all-MiniLM-L6-v2"
        # Should use defaults
        assert restored_ef.device == "cpu"
        assert restored_ef.normalize_embeddings is False
        assert restored_ef.kwargs == {}

    def test_build_from_config_with_partial_config(self):
        """Test that build_from_config() handles partial configuration"""
        config = {"model_name": "all-mpnet-base-v2", "normalize_embeddings": True}

        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert restored_ef.model_name == "all-mpnet-base-v2"
        assert restored_ef.normalize_embeddings is True
        # Should use defaults for missing fields
        assert restored_ef.device == "cpu"
        assert restored_ef.kwargs == {}

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        original_ef = SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2",
            device="cpu",
            normalize_embeddings=True,
            trust_remote_code=True,
        )

        config = original_ef.get_config()
        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, SentenceTransformerEmbeddingFunction)
        assert restored_ef.model_name == original_ef.model_name
        assert restored_ef.device == original_ef.device
        assert restored_ef.normalize_embeddings == original_ef.normalize_embeddings
        assert restored_ef.kwargs == original_ef.kwargs

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_initialization_with_cuda(self):
        """Test that SentenceTransformerEmbeddingFunction can be initialized with CUDA device"""
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda")

        assert ef.device == "cuda"
        assert ef.model_name == "all-MiniLM-L6-v2"

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_embeddings_with_cuda(self):
        """Test that embeddings can be generated using CUDA"""
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda")

        # Generate embeddings
        embeddings = ef(["Hello world", "How are you?"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        assert len(embeddings[1]) > 0
        assert len(embeddings[0]) == len(embeddings[1])  # Same dimension

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_get_config_with_cuda(self):
        """Test that get_config() correctly saves CUDA device setting"""
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda", normalize_embeddings=True
        )

        config = ef.get_config()

        assert config["device"] == "cuda"
        assert config["model_name"] == "all-MiniLM-L6-v2"
        assert config["normalize_embeddings"] is True

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_build_from_config_with_cuda(self):
        """Test that build_from_config() correctly restores CUDA device setting"""
        config = {
            "model_name": "all-MiniLM-L6-v2",
            "device": "cuda",
            "normalize_embeddings": False,
            "kwargs": {},
        }

        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, SentenceTransformerEmbeddingFunction)
        assert restored_ef.device == "cuda"
        assert restored_ef.model_name == "all-MiniLM-L6-v2"

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_persistence_roundtrip_with_cuda(self):
        """Test complete roundtrip with CUDA: get_config -> build_from_config"""
        original_ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda", normalize_embeddings=True
        )

        config = original_ef.get_config()
        restored_ef = SentenceTransformerEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, SentenceTransformerEmbeddingFunction)
        assert restored_ef.model_name == original_ef.model_name
        assert restored_ef.device == original_ef.device
        assert restored_ef.normalize_embeddings == original_ef.normalize_embeddings

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this system")
    def test_cuda_vs_cpu_config_consistency(self):
        """Test that CUDA and CPU configs are handled consistently"""
        ef_cuda = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda")
        ef_cpu = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cpu")

        config_cuda = ef_cuda.get_config()
        config_cpu = ef_cpu.get_config()

        # Both should have same structure, different device
        assert config_cuda["device"] == "cuda"
        assert config_cpu["device"] == "cpu"
        assert config_cuda["model_name"] == config_cpu["model_name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
