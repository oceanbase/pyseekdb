"""
Unit tests for sparse vector index SQL generation.
"""

from typing import Any

import pytest

from pyseekdb.client.client_base import _get_sparse_vector_index_sql
from pyseekdb.client.configuration import SparseVectorIndexConfig
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


def _make_config(**overrides) -> SparseVectorIndexConfig:
    defaults = {"embedding_function": _FakeSparseEF()}
    defaults.update(overrides)
    return SparseVectorIndexConfig(**defaults)


class TestGetSparseVectorIndexSql:
    """Test _get_sparse_vector_index_sql helper function"""

    def test_basic_defaults(self):
        config = _make_config()
        sql = _get_sparse_vector_index_sql(config)
        assert sql.startswith("WITH (")
        assert sql.endswith(")")
        assert "DISTANCE=inner_product" in sql
        assert "TYPE=sindi" in sql
        assert "LIB=vsag" in sql

    def test_no_optional_params_by_default(self):
        config = _make_config()
        _ = _get_sparse_vector_index_sql(config)

    def test_with_prune_true(self):
        config = _make_config(prune=True)
        sql = _get_sparse_vector_index_sql(config)
        assert "prune=true" in sql

    def test_with_prune_false(self):
        config = _make_config(prune=False)
        sql = _get_sparse_vector_index_sql(config)
        assert "prune=false" in sql

    def test_with_refine_true(self):
        config = _make_config(refine=True)
        sql = _get_sparse_vector_index_sql(config)
        assert "refine=true" in sql

    def test_with_drop_ratio_build(self):
        config = _make_config(drop_ratio_build=0.1)
        sql = _get_sparse_vector_index_sql(config)
        assert "drop_ratio_build=0.1" in sql

    def test_with_drop_ratio_search(self):
        config = _make_config(drop_ratio_search=0.2)
        sql = _get_sparse_vector_index_sql(config)
        assert "drop_ratio_search=0.2" in sql

    def test_with_refine_k(self):
        config = _make_config(refine_k=4.0)
        sql = _get_sparse_vector_index_sql(config)
        assert "refine_k=4.0" in sql

    def test_with_all_optional_params(self):
        config = _make_config(
            prune=True,
            refine=True,
            drop_ratio_build=0.1,
            drop_ratio_search=0.2,
            refine_k=4.0,
        )
        sql = _get_sparse_vector_index_sql(config)
        assert "DISTANCE=inner_product" in sql
        assert "TYPE=sindi" in sql
        assert "LIB=vsag" in sql
        assert "prune=true" in sql
        assert "refine=true" in sql
        assert "drop_ratio_build=0.1" in sql
        assert "drop_ratio_search=0.2" in sql
        assert "refine_k=4.0" in sql

    def test_sql_format_is_comma_separated(self):
        config = _make_config(prune=True, refine=True)
        sql = _get_sparse_vector_index_sql(config)
        inner = sql[len("WITH (") : -1]
        parts = [p.strip() for p in inner.split(",")]
        assert len(parts) >= 3
        for part in parts:
            assert "=" in part


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
