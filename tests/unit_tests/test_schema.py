"""
Unit tests for Schema class.
"""

from typing import Any

import pytest

from pyseekdb.client.configuration import (
    FulltextIndexConfig,
    HNSWConfiguration,
    SparseVectorIndexConfig,
    VectorIndexConfig,
)
from pyseekdb.client.schema import Schema
from pyseekdb.client.sparse_embedding_function import (
    Documents,
    SparseEmbeddingFunction,
    SparseVector,
    SparseVectors,
)


class _FakeSparseEF(SparseEmbeddingFunction):
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


def _make_sparse_config(**overrides) -> SparseVectorIndexConfig:
    defaults = {"embedding_function": _FakeSparseEF()}
    defaults.update(overrides)
    return SparseVectorIndexConfig(**defaults)


class TestSchemaInit:
    """Test Schema __init__ with various argument combinations"""

    def test_defaults(self):
        schema = Schema()
        assert isinstance(schema.vector_index, VectorIndexConfig)
        assert schema.sparse_vector_index is None
        assert schema.fulltext_index is None

    def test_with_vector_index_config(self):
        vic = VectorIndexConfig(hnsw=HNSWConfiguration(dimension=128))
        schema = Schema(vector_index=vic)
        assert schema.vector_index is vic

    def test_with_hnsw_configuration(self):
        hnsw = HNSWConfiguration(dimension=256, distance="cosine")
        schema = Schema(vector_index=hnsw)
        assert isinstance(schema.vector_index, VectorIndexConfig)
        assert schema.vector_index.hnsw is hnsw

    def test_with_none_vector_index(self):
        schema = Schema(vector_index=None)
        assert isinstance(schema.vector_index, VectorIndexConfig)

    def test_with_invalid_vector_index_type(self):
        with pytest.raises(TypeError, match="Unsupported vector index configuration type"):
            Schema(vector_index="bad")

    def test_with_sparse_vector_index(self):
        sparse = _make_sparse_config()
        schema = Schema(sparse_vector_index=sparse)
        assert schema.sparse_vector_index is sparse

    def test_with_fulltext_index(self):
        ft = FulltextIndexConfig(analyzer="space")
        schema = Schema(fulltext_index=ft)
        assert schema.fulltext_index is ft

    def test_with_all_indexes(self):
        hnsw = HNSWConfiguration(dimension=384, distance="cosine")
        sparse = _make_sparse_config()
        fulltext = FulltextIndexConfig(analyzer="ik")
        schema = Schema(
            vector_index=hnsw,
            sparse_vector_index=sparse,
            fulltext_index=fulltext,
        )
        assert schema.vector_index.hnsw is hnsw
        assert schema.sparse_vector_index is sparse
        assert schema.fulltext_index is fulltext


class TestSchemaCreateIndex:
    """Test Schema.create_index fluent API"""

    def test_create_index_hnsw(self):
        hnsw = HNSWConfiguration(dimension=128)
        schema = Schema().create_index(hnsw)
        assert isinstance(schema, Schema)
        assert schema.vector_index.hnsw is hnsw

    def test_create_index_vector_index_config(self):
        vic = VectorIndexConfig(hnsw=HNSWConfiguration(dimension=64))
        schema = Schema().create_index(vic)
        assert schema.vector_index is vic

    def test_create_index_sparse(self):
        sparse = _make_sparse_config()
        schema = Schema().create_index(sparse)
        assert schema.sparse_vector_index is sparse

    def test_create_index_fulltext(self):
        ft = FulltextIndexConfig(analyzer="ngram")
        schema = Schema().create_index(ft)
        assert schema.fulltext_index is ft

    def test_create_index_invalid_type(self):
        with pytest.raises(TypeError, match="Unsupported index configuration type"):
            Schema().create_index("bad_config")

    def test_chaining_multiple_indexes(self):
        hnsw = HNSWConfiguration(dimension=512, distance="cosine")
        sparse = _make_sparse_config()
        ft = FulltextIndexConfig(analyzer="space")

        schema = Schema().create_index(hnsw).create_index(sparse).create_index(ft)

        assert schema.vector_index.hnsw is hnsw
        assert schema.sparse_vector_index is sparse
        assert schema.fulltext_index is ft

    def test_create_index_overwrites_same_type(self):
        sparse1 = _make_sparse_config(prune=True)
        sparse2 = _make_sparse_config(refine=True)
        schema = Schema().create_index(sparse1).create_index(sparse2)
        assert schema.sparse_vector_index is sparse2
        assert schema.sparse_vector_index.refine is True

    def test_returns_self(self):
        schema = Schema()
        result = schema.create_index(_make_sparse_config())
        assert result is schema


class TestSchemaRepr:
    """Test Schema __repr__"""

    def test_repr_defaults(self):
        schema = Schema()
        r = repr(schema)
        assert "Schema(" in r
        assert "vector_index=" in r

    def test_repr_with_sparse(self):
        schema = Schema(sparse_vector_index=_make_sparse_config())
        r = repr(schema)
        assert "sparse_vector_index=" in r

    def test_repr_with_fulltext(self):
        schema = Schema(fulltext_index=FulltextIndexConfig())
        r = repr(schema)
        assert "fulltext_index=" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
