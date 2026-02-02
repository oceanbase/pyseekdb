"""
Unit tests for OnnxEmbeddingFunction.
"""

from __future__ import annotations

import importlib

import pytest

from pyseekdb.utils.embedding_functions.onnx_embedding_function import OnnxEmbeddingFunction


def is_onnx_available() -> bool:
    """
    Check if openai is available for testing.

    Returns:
        True if openai is available, False otherwise.
    """
    return importlib.util.find_spec("onnxruntime") is not None


# Skip this test by default - it requires external API access and API keys
@pytest.mark.skipif(
    not is_onnx_available(),
    reason="onnxruntime is not available",
)
class TestOnnxEmbeddingFunction:
    def _make_onnx(self) -> OnnxEmbeddingFunction:
        return OnnxEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            hf_model_id="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            preferred_providers=None,
        )

    def test_init_validates_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            OnnxEmbeddingFunction(model_name="", hf_model_id="org/test", dimension=3)

    def test_init_validates_hf_model_id(self) -> None:
        with pytest.raises(ValueError, match="hf_model_id must be a non-empty string"):
            OnnxEmbeddingFunction(model_name="test", hf_model_id="", dimension=3)

    def test_init_validates_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension must be a positive integer"):
            OnnxEmbeddingFunction(model_name="test", hf_model_id="org/test", dimension=0)

    def test_init_validates_preferred_providers(self) -> None:
        with pytest.raises(ValueError, match="Preferred providers must be a list of strings"):
            OnnxEmbeddingFunction(model_name="test", hf_model_id="org/test", dimension=3, preferred_providers=[1])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="Preferred providers must be unique"):
            OnnxEmbeddingFunction(
                model_name="test",
                hf_model_id="org/test",
                dimension=3,
                preferred_providers=["CPUExecutionProvider", "CPUExecutionProvider"],
            )

    def test_dimension_property(self) -> None:
        ef = self._make_onnx()
        assert ef.dimension == 384

    def test_call_empty_returns_empty(self) -> None:
        ef = self._make_onnx()
        assert ef([]) == []

    def test_call_generates_embeddings(self) -> None:
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")
        pytest.importorskip("tqdm")

        ef = self._make_onnx()
        embeddings = ef("hello world")

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == ef.dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
