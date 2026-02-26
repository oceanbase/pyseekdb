"""
Unit tests for SparseVector dataclass and _sparse_vector_to_sql helper.
"""

import pytest

from pyseekdb.client.sparse_embedding_function import SparseVector, _sparse_vector_to_sql


class TestSparseVector:
    """Test SparseVector dataclass"""

    def test_from_dict_basic(self):
        sv = SparseVector.from_dict({100: 0.5, 200: 0.3, 500: 0.8})
        assert sv.embeddings == {100: 0.5, 200: 0.3, 500: 0.8}

    def test_from_dict_single_entry(self):
        sv = SparseVector.from_dict({42: 1.0})
        assert sv.embeddings == {42: 1.0}

    def test_from_dict_invalid_type(self):
        with pytest.raises(TypeError, match="embeddings must be a dict"):
            SparseVector.from_dict([1, 2, 3])

    def test_from_dict_invalid_type_string(self):
        with pytest.raises(TypeError, match="embeddings must be a dict"):
            SparseVector.from_dict("not a dict")

    def test_from_indices_basic(self):
        sv = SparseVector.from_indices([100, 200, 500], [0.5, 0.3, 0.8])
        assert sv.embeddings == {100: 0.5, 200: 0.3, 500: 0.8}

    def test_from_indices_single_entry(self):
        sv = SparseVector.from_indices([42], [1.0])
        assert sv.embeddings == {42: 1.0}

    def test_from_indices_empty(self):
        sv = SparseVector.from_indices([], [])
        assert sv.embeddings == {}

    def test_from_indices_length_mismatch(self):
        with pytest.raises(ValueError, match="must have the same length"):
            SparseVector.from_indices([1, 2, 3], [0.5, 0.3])

    def test_default_embeddings_none(self):
        sv = SparseVector()
        assert sv.embeddings is None

    def test_to_sql_string_basic(self):
        sv = SparseVector.from_dict({100: 0.5, 200: 0.3})
        sql = sv.to_sql_string()
        assert sql.startswith("'{")
        assert sql.endswith("}'")
        assert "100:0.5" in sql
        assert "200:0.3" in sql

    def test_to_sql_string_single_entry(self):
        sv = SparseVector.from_dict({42: 1.0})
        sql = sv.to_sql_string()
        assert "42:1.0" in sql

    def test_to_sql_string_empty_raises(self):
        sv = SparseVector(embeddings={})
        with pytest.raises(ValueError, match="Cannot convert empty sparse vector"):
            sv.to_sql_string()

    def test_to_sql_string_none_raises(self):
        sv = SparseVector(embeddings=None)
        with pytest.raises(ValueError, match="Cannot convert empty sparse vector"):
            sv.to_sql_string()

    def test_repr_with_data(self):
        sv = SparseVector.from_dict({1: 0.1, 2: 0.2, 3: 0.3})
        r = repr(sv)
        assert "3 non-zero entries" in r

    def test_repr_none(self):
        sv = SparseVector()
        assert "None" in repr(sv)


class TestSparseVectorToSql:
    """Test _sparse_vector_to_sql helper function"""

    def test_with_sparse_vector(self):
        sv = SparseVector.from_dict({10: 0.5, 20: 0.3})
        sql = _sparse_vector_to_sql(sv)
        assert sql.startswith("'{")
        assert sql.endswith("}'")
        assert "10:0.5" in sql
        assert "20:0.3" in sql

    def test_with_raw_dict(self):
        sql = _sparse_vector_to_sql({10: 0.5, 20: 0.3})
        assert sql.startswith("'{")
        assert sql.endswith("}'")
        assert "10:0.5" in sql
        assert "20:0.3" in sql

    def test_with_empty_dict_raises(self):
        with pytest.raises(ValueError, match="Cannot convert empty sparse vector"):
            _sparse_vector_to_sql({})

    def test_with_empty_sparse_vector_raises(self):
        with pytest.raises(ValueError, match="Cannot convert empty sparse vector"):
            _sparse_vector_to_sql(SparseVector(embeddings={}))

    def test_with_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected SparseVector or dict"):
            _sparse_vector_to_sql([1, 2, 3])

    def test_with_invalid_type_string_raises(self):
        with pytest.raises(TypeError, match="Expected SparseVector or dict"):
            _sparse_vector_to_sql("not valid")

    def test_consistent_output(self):
        """SparseVector and raw dict should produce the same SQL"""
        data = {100: 0.5, 200: 0.3, 500: 0.8}
        sql_from_sv = _sparse_vector_to_sql(SparseVector.from_dict(data))
        sql_from_dict = _sparse_vector_to_sql(data)
        assert sql_from_sv == sql_from_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
