"""
Unit tests for FieldKey (K) type constants.
"""

from typing import Any

import pytest

from pyseekdb.client.types import FieldKey, K


class TestFieldKey:
    """Test FieldKey class and its constants"""

    def test_k_is_field_key(self):
        assert K is FieldKey

    def test_constants_are_field_key_instances(self):
        assert isinstance(K.ID, FieldKey)
        assert isinstance(K.DOCUMENT, FieldKey)
        assert isinstance(K.EMBEDDING, FieldKey)
        assert isinstance(K.SPARSE_EMBEDDING, FieldKey)
        assert isinstance(K.SCORE, FieldKey)

    def test_constant_names(self):
        assert K.ID.name == "#id"
        assert K.DOCUMENT.name == "#document"
        assert K.EMBEDDING.name == "#embedding"
        assert K.SPARSE_EMBEDDING.name == "#sparse_embedding"
        assert K.SCORE.name == "#score"

    def test_identity(self):
        assert K.SPARSE_EMBEDDING is FieldKey.SPARSE_EMBEDDING
        assert K.DOCUMENT is FieldKey.DOCUMENT

    def test_custom_field_key(self):
        fk = FieldKey("custom_field")
        assert fk.name == "custom_field"

    def test_field_key_equality_by_identity(self):
        """FieldKey doesn't define __eq__, so comparison is by identity"""
        fk1 = FieldKey("#sparse_embedding")
        fk2 = FieldKey("#sparse_embedding")
        assert fk1 is not fk2

    def test_field_key_name_check(self):
        """The typical way to compare FieldKey values is via .name"""
        fk = FieldKey("#sparse_embedding")
        assert fk.name == K.SPARSE_EMBEDDING.name


class TestFieldKeyUsagePatterns:
    """Test typical usage patterns of K"""

    def test_as_sparse_vector_index_source_key(self):
        from pyseekdb.client.configuration import SparseVectorIndexConfig
        from pyseekdb.client.sparse_embedding_function import (
            Documents,
            SparseEmbeddingFunction,
            SparseVector,
            SparseVectors,
        )

        class _Fake(SparseEmbeddingFunction):
            def __call__(self, documents: Documents) -> SparseVectors:
                return [SparseVector.from_dict({0: 1.0})]

            @staticmethod
            def name() -> str:
                return "fake"

            def get_config(self) -> dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(config: dict[str, Any]) -> "_Fake":
                return _Fake()

        config = SparseVectorIndexConfig(embedding_function=_Fake(), source_key=K.DOCUMENT)
        assert config.source_key is K.DOCUMENT

    def test_metadata_string_as_source_key(self):
        """Plain strings work as metadata source keys"""
        from pyseekdb.client.configuration import SparseVectorIndexConfig
        from pyseekdb.client.sparse_embedding_function import (
            Documents,
            SparseEmbeddingFunction,
            SparseVector,
            SparseVectors,
        )

        class _Fake(SparseEmbeddingFunction):
            def __call__(self, documents: Documents) -> SparseVectors:
                return [SparseVector.from_dict({0: 1.0})]

            @staticmethod
            def name() -> str:
                return "fake2"

            def get_config(self) -> dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(config: dict[str, Any]) -> "_Fake":
                return _Fake()

        config = SparseVectorIndexConfig(embedding_function=_Fake(), source_key="title")
        assert config.source_key == "title"

    def test_sparse_embedding_identity_check(self):
        query_key = K.SPARSE_EMBEDDING
        assert query_key is K.SPARSE_EMBEDDING
        assert query_key.name == "#sparse_embedding"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
