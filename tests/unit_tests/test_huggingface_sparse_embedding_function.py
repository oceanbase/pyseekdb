"""
Unit tests for HuggingFaceSparseEmbeddingFunction.

Tests persistence (get_config / build_from_config), protocol compliance,
and the sparse-vector conversion logic.

The SparseEncoder model loading is mocked to avoid downloading large models.

To run:
    pytest tests/unit_tests/test_huggingface_sparse_embedding_function.py -v
"""

import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyseekdb.client.sparse_embedding_function import (
    SparseEmbeddingFunctionRegistry,
    SparseVector,
)


def _sparse_encoder_available() -> bool:
    """Check if sentence_transformers with SparseEncoder is available."""
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        from sentence_transformers import SparseEncoder  # noqa: F401
    except ImportError:
        return False
    else:
        return True


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clear the class-level model cache before each test."""
    from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
        HuggingFaceSparseEmbeddingFunction,
    )

    saved = HuggingFaceSparseEmbeddingFunction.models.copy()
    HuggingFaceSparseEmbeddingFunction.models.clear()
    yield
    HuggingFaceSparseEmbeddingFunction.models.clear()
    HuggingFaceSparseEmbeddingFunction.models.update(saved)


@pytest.fixture()
def mock_sparse_encoder():
    """Mock SparseEncoder to avoid real model download/loading."""
    with patch("sentence_transformers.SparseEncoder") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


@pytest.mark.skipif(
    not _sparse_encoder_available(),
    reason="sentence-transformers with SparseEncoder is not available",
)
class TestHuggingFaceSparseEFInit:
    """Test initialization and basic properties."""

    def test_default_init(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        ef = HuggingFaceSparseEmbeddingFunction()
        assert ef.model_name == "prithivida/Splade_PP_en_v1"
        assert ef.device == "cpu"
        assert ef.task == "document"
        assert ef.kwargs == {}

    def test_custom_init(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        ef = HuggingFaceSparseEmbeddingFunction(
            model_name="naver/splade-v3",
            device="cuda",
            task="query",
            trust_remote_code=True,
        )
        assert ef.model_name == "naver/splade-v3"
        assert ef.device == "cuda"
        assert ef.task == "query"
        assert ef.kwargs == {"trust_remote_code": True}

    def test_invalid_kwarg_type_raises(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        with pytest.raises(TypeError, match="primitive type"):
            HuggingFaceSparseEmbeddingFunction(bad_arg=object())

    def test_model_cached_across_instances(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        mock_cls, _ = mock_sparse_encoder
        HuggingFaceSparseEmbeddingFunction(model_name="model-a")
        HuggingFaceSparseEmbeddingFunction(model_name="model-a")
        mock_cls.assert_called_once()

    def test_different_models_loaded_separately(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        mock_cls, _ = mock_sparse_encoder
        HuggingFaceSparseEmbeddingFunction(model_name="model-a")
        HuggingFaceSparseEmbeddingFunction(model_name="model-b")
        assert mock_cls.call_count == 2


@pytest.mark.skipif(
    not _sparse_encoder_available(),
    reason="sentence-transformers with SparseEncoder is not available",
)
class TestHuggingFaceSparseEFCall:
    """Test __call__ and embed_query with mocked model output."""

    def _make_dense_output(self, nonzero_map: list[dict[int, float]], dim: int = 30000):
        """Build fake model output tensors from nonzero specs."""
        results = []
        for nz in nonzero_map:
            arr = np.zeros(dim, dtype=np.float32)
            for idx, val in nz.items():
                arr[idx] = val
            mock_tensor = MagicMock()
            mock_tensor.to_dense.return_value = MagicMock(numpy=MagicMock(return_value=arr))
            results.append(mock_tensor)
        return results

    def test_call_document_mode(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        _, mock_model = mock_sparse_encoder
        mock_model.encode_document.return_value = self._make_dense_output([{100: 0.5, 200: 0.3}, {500: 0.8}])

        ef = HuggingFaceSparseEmbeddingFunction(task="document")
        result = ef(["doc1", "doc2"])

        assert len(result) == 2
        assert isinstance(result[0], SparseVector)
        assert result[0].embeddings == {100: pytest.approx(0.5, abs=1e-5), 200: pytest.approx(0.3, abs=1e-5)}
        assert result[1].embeddings == {500: pytest.approx(0.8, abs=1e-5)}
        mock_model.encode_document.assert_called_once()

    def test_call_query_mode(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        _, mock_model = mock_sparse_encoder
        mock_model.encode_query.return_value = self._make_dense_output([{10: 1.0}])

        ef = HuggingFaceSparseEmbeddingFunction(task="query")
        result = ef(["search term"])

        assert len(result) == 1
        assert result[0].embeddings == {10: pytest.approx(1.0, abs=1e-5)}
        mock_model.encode_query.assert_called_once()

    def test_call_single_string_input(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        _, mock_model = mock_sparse_encoder
        mock_model.encode_document.return_value = self._make_dense_output([{42: 0.7}])

        ef = HuggingFaceSparseEmbeddingFunction()
        result = ef("single string")

        assert len(result) == 1
        mock_model.encode_document.assert_called_once_with(["single string"])

    def test_embed_query_always_uses_encode_query(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        _, mock_model = mock_sparse_encoder
        mock_model.encode_query.return_value = self._make_dense_output([{5: 0.9}])

        ef = HuggingFaceSparseEmbeddingFunction(task="document")
        result = ef.embed_query(["query text"])

        assert len(result) == 1
        mock_model.encode_query.assert_called_once()
        mock_model.encode_document.assert_not_called()

    def test_numpy_array_fallback(self, mock_sparse_encoder):
        """Test conversion when model returns plain numpy arrays (no to_dense)."""
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        _, mock_model = mock_sparse_encoder
        arr = np.zeros(100, dtype=np.float32)
        arr[7] = 0.6
        arr[99] = 0.1
        mock_model.encode_document.return_value = [arr]

        ef = HuggingFaceSparseEmbeddingFunction()
        result = ef(["test"])

        assert len(result) == 1
        assert 7 in result[0].embeddings
        assert 99 in result[0].embeddings
        assert result[0].embeddings[7] == pytest.approx(0.6, abs=1e-5)


@pytest.mark.skipif(
    not _sparse_encoder_available(),
    reason="sentence-transformers with SparseEncoder is not available",
)
class TestHuggingFaceSparseEFPersistence:
    """Test get_config / build_from_config / name."""

    def test_name(self):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        assert HuggingFaceSparseEmbeddingFunction.name() == "huggingface_sparse"

    def test_get_config_defaults(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        ef = HuggingFaceSparseEmbeddingFunction()
        config = ef.get_config()

        assert config["model_name"] == "prithivida/Splade_PP_en_v1"
        assert config["device"] == "cpu"
        assert config["task"] == "document"
        assert config["kwargs"] == {}
        assert "name" not in config

    def test_get_config_custom(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        ef = HuggingFaceSparseEmbeddingFunction(
            model_name="naver/splade-v3",
            device="cuda:0",
            task="query",
            trust_remote_code=True,
        )
        config = ef.get_config()

        assert config["model_name"] == "naver/splade-v3"
        assert config["device"] == "cuda:0"
        assert config["task"] == "query"
        assert config["kwargs"] == {"trust_remote_code": True}

    def test_build_from_config_defaults(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        config = {
            "model_name": "prithivida/Splade_PP_en_v1",
            "device": "cpu",
            "task": "document",
            "kwargs": {},
        }
        ef = HuggingFaceSparseEmbeddingFunction.build_from_config(config)

        assert isinstance(ef, HuggingFaceSparseEmbeddingFunction)
        assert ef.model_name == "prithivida/Splade_PP_en_v1"
        assert ef.device == "cpu"
        assert ef.task == "document"

    def test_build_from_config_custom(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        config = {
            "model_name": "naver/splade-v3",
            "device": "cuda",
            "task": "query",
            "kwargs": {"trust_remote_code": True},
        }
        ef = HuggingFaceSparseEmbeddingFunction.build_from_config(config)

        assert ef.model_name == "naver/splade-v3"
        assert ef.task == "query"
        assert ef.kwargs == {"trust_remote_code": True}

    def test_build_from_config_minimal(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        config = {"model_name": "some-model"}
        ef = HuggingFaceSparseEmbeddingFunction.build_from_config(config)

        assert ef.model_name == "some-model"
        assert ef.device == "cpu"
        assert ef.task == "document"
        assert ef.kwargs == {}

    def test_build_from_config_invalid_kwargs_type(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        with pytest.raises(TypeError, match="kwargs must be a dictionary"):
            HuggingFaceSparseEmbeddingFunction.build_from_config({"kwargs": "not-a-dict"})

    def test_roundtrip(self, mock_sparse_encoder):
        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        original = HuggingFaceSparseEmbeddingFunction(
            model_name="naver/splade-v3",
            device="cpu",
            task="query",
            trust_remote_code=True,
        )
        config = original.get_config()
        restored = HuggingFaceSparseEmbeddingFunction.build_from_config(config)

        assert restored.model_name == original.model_name
        assert restored.device == original.device
        assert restored.task == original.task
        assert restored.kwargs == original.kwargs


@pytest.mark.skipif(
    not _sparse_encoder_available(),
    reason="sentence-transformers with SparseEncoder is not available",
)
class TestHuggingFaceSparseEFRegistry:
    """Test that the class is auto-registered in SparseEmbeddingFunctionRegistry."""

    def test_registered_in_registry(self):
        assert SparseEmbeddingFunctionRegistry.get_class("huggingface_sparse") is not None

    def test_registry_build_from_config(self, mock_sparse_encoder):
        config = {
            "model_name": "prithivida/Splade_PP_en_v1",
            "device": "cpu",
            "task": "document",
            "kwargs": {},
        }
        ef = SparseEmbeddingFunctionRegistry.build_from_config("huggingface_sparse", config)

        from pyseekdb.utils.embedding_functions.huggingface_sparse_embedding_function import (
            HuggingFaceSparseEmbeddingFunction,
        )

        assert isinstance(ef, HuggingFaceSparseEmbeddingFunction)

    def test_listed_in_registry(self):
        names = SparseEmbeddingFunctionRegistry.list_registered()
        assert "huggingface_sparse" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
