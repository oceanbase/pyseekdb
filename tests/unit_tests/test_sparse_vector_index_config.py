"""
Unit tests for SparseVectorIndexConfig.
"""

from typing import Any

import pytest

from pyseekdb.client.configuration import SparseVectorIndexConfig
from pyseekdb.client.sparse_embedding_function import (
    Documents,
    SparseEmbeddingFunction,
    SparseVector,
    SparseVectors,
)
from pyseekdb.client.types import K


class _FakeSparseEF(SparseEmbeddingFunction):
    """Minimal valid sparse embedding function for testing."""

    def __call__(self, documents: Documents) -> SparseVectors:
        if isinstance(documents, str):
            documents = [documents]
        return [SparseVector.from_dict({0: 1.0})] * len(documents)

    @staticmethod
    def name() -> str:
        return "fake_sparse"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_FakeSparseEF":
        return _FakeSparseEF()


class _NonPersistentSparseEF(SparseEmbeddingFunction):
    """Implements protocol shape but does not support persistence."""

    def __call__(self, documents: Documents) -> SparseVectors:
        if isinstance(documents, str):
            documents = [documents]
        return [SparseVector.from_dict({0: 1.0})] * len(documents)

    @staticmethod
    def name() -> str:
        return ""

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_NonPersistentSparseEF":
        return _NonPersistentSparseEF()


def _make_config(**overrides) -> SparseVectorIndexConfig:
    """Helper to create SparseVectorIndexConfig with a default embedding_function."""
    defaults = {"embedding_function": _FakeSparseEF()}
    defaults.update(overrides)
    return SparseVectorIndexConfig(**defaults)


class TestSparseVectorIndexConfigDefaults:
    """Test default values of SparseVectorIndexConfig"""

    def test_defaults(self):
        config = _make_config()
        assert isinstance(config.embedding_function, _FakeSparseEF)
        assert config.source_key is K.DOCUMENT
        assert config.lib == "vsag"
        assert config.distance == "inner_product"
        assert config.type == "sindi"
        assert config.prune is False
        assert config.refine is False
        assert config.drop_ratio_build == 0.0
        assert config.drop_ratio_search == 0.0
        assert config.refine_k == 4.0
        assert config.properties is None

    def test_with_embedding_function(self):
        ef = _FakeSparseEF()
        config = SparseVectorIndexConfig(embedding_function=ef)
        assert config.embedding_function is ef

    def test_with_metadata_source_key(self):
        config = _make_config(source_key="title")
        assert config.source_key == "title"

    def test_embedding_function_none_raises(self):
        """SparseVectorIndexConfig requires an embedding_function."""
        with pytest.raises(ValueError, match="embedding_function is None"):
            SparseVectorIndexConfig(embedding_function=None)

    def test_embedding_function_must_support_persistence(self):
        with pytest.raises(ValueError, match="must support persistence"):
            SparseVectorIndexConfig(embedding_function=_NonPersistentSparseEF())


class TestSparseVectorIndexConfigValidation:
    """Test validation logic in __post_init__"""

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError, match="only supports inner_product"):
            _make_config(distance="cosine")

    def test_invalid_lib_raises(self):
        with pytest.raises(ValueError, match="only supports 'vsag'"):
            _make_config(lib="faiss")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="only supports 'sindi'"):
            _make_config(type="hnsw")

    def test_drop_ratio_build_too_low(self):
        with pytest.raises(ValueError, match="drop_ratio_build must be between"):
            _make_config(drop_ratio_build=-0.1)

    def test_drop_ratio_build_too_high(self):
        with pytest.raises(ValueError, match="drop_ratio_build must be between"):
            _make_config(drop_ratio_build=0.95)

    def test_drop_ratio_build_valid_boundary(self):
        config = _make_config(drop_ratio_build=0.0)
        assert config.drop_ratio_build == 0.0
        config = _make_config(drop_ratio_build=0.9)
        assert config.drop_ratio_build == 0.9

    def test_drop_ratio_search_too_low(self):
        with pytest.raises(ValueError, match="drop_ratio_search must be between"):
            _make_config(drop_ratio_search=-0.1)

    def test_drop_ratio_search_too_high(self):
        with pytest.raises(ValueError, match="drop_ratio_search must be between"):
            _make_config(drop_ratio_search=0.95)

    def test_refine_k_too_low(self):
        with pytest.raises(ValueError, match="refine_k must be between"):
            _make_config(refine_k=0.5)

    def test_refine_k_too_high(self):
        with pytest.raises(ValueError, match="refine_k must be between"):
            _make_config(refine_k=1001.0)

    def test_refine_k_valid_boundary(self):
        config = _make_config(refine_k=1.0)
        assert config.refine_k == 1.0
        config = _make_config(refine_k=1000.0)
        assert config.refine_k == 1000.0

    def test_properties_invalid_type(self):
        with pytest.raises(TypeError, match="properties must be a dictionary"):
            _make_config(properties={"bad": {"nested": "dict"}})

    def test_properties_valid(self):
        config = _make_config(properties={"key": "value", "num": 42, "flag": True, "rate": 0.5})
        assert config.properties["key"] == "value"
        assert config.properties["num"] == 42
        assert config.properties["flag"] is True
        assert config.properties["rate"] == 0.5


class TestSparseVectorIndexConfigResolveSourceKey:
    """Test resolve_source_key method"""

    def test_resolve_document_key_constant(self):
        config = _make_config(source_key=K.DOCUMENT)
        source_type, meta_key = config.resolve_source_key()
        assert source_type == "document"
        assert meta_key is None

    def test_resolve_document_key_string(self):
        config = _make_config(source_key="#document")
        source_type, meta_key = config.resolve_source_key()
        assert source_type == "document"
        assert meta_key is None

    def test_resolve_metadata_key(self):
        config = _make_config(source_key="title")
        source_type, meta_key = config.resolve_source_key()
        assert source_type == "metadata"
        assert meta_key == "title"

    def test_resolve_metadata_key_custom(self):
        config = _make_config(source_key="description")
        source_type, meta_key = config.resolve_source_key()
        assert source_type == "metadata"
        assert meta_key == "description"

    def test_resolve_none_source_key_defaults_to_document(self):
        config = SparseVectorIndexConfig(embedding_function=_FakeSparseEF(), source_key=None)
        source_type, meta_key = config.resolve_source_key()
        assert config.source_key is K.DOCUMENT
        assert source_type == "document"
        assert meta_key is None


class TestSparseVectorIndexConfigOptionalParams:
    """Test optional parameters (prune, refine, etc.)"""

    def test_prune_true(self):
        config = _make_config(prune=True)
        assert config.prune is True

    def test_refine_true(self):
        config = _make_config(refine=True)
        assert config.refine is True

    def test_all_optional_params(self):
        config = _make_config(
            prune=True,
            refine=True,
            drop_ratio_build=0.1,
            drop_ratio_search=0.2,
            refine_k=4.0,
        )
        assert config.prune is True
        assert config.refine is True
        assert config.drop_ratio_build == 0.1
        assert config.drop_ratio_search == 0.2
        assert config.refine_k == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
