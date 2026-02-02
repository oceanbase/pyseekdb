"""
Unit tests for DefaultEmbeddingFunction.
"""

import sys
from typing import Any

import pytest

from pyseekdb.client.embedding_function import DefaultEmbeddingFunction


class TestDefaultEmbeddingFunctionPersistence:
    """
    Test persistence for DefaultEmbeddingFunction
    Tests the persistence functionality for DefaultEmbeddingFunction.
    These tests verify that DefaultEmbeddingFunction can be serialized to config dictionaries
    and restored from them correctly.
    """

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert DefaultEmbeddingFunction.name() == "default"

    def test_get_config_returns_empty_dict(self):
        """Test that get_config() returns an empty dictionary (all defaults)"""
        ef = DefaultEmbeddingFunction()
        config = ef.get_config()

        assert isinstance(config, dict)
        assert config == {}
        assert len(config) == 0

    def test_build_from_config_creates_default_instance(self):
        """Test that build_from_config() creates a default instance"""
        config: dict[str, Any] = {}

        restored_ef = DefaultEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, DefaultEmbeddingFunction)
        assert restored_ef.model_name == "all-MiniLM-L6-v2"

    def test_build_from_config_with_empty_dict(self):
        """Test that build_from_config() works with empty dictionary"""
        config = {}

        restored_ef = DefaultEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, DefaultEmbeddingFunction)
        assert restored_ef.model_name == "all-MiniLM-L6-v2"

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        original_ef = DefaultEmbeddingFunction()

        config = original_ef.get_config()
        restored_ef = DefaultEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, DefaultEmbeddingFunction)
        assert restored_ef.model_name == original_ef.model_name


def test_default_embedding_function_on_py314():
    if sys.version_info < (3, 14):
        pytest.skip("Python < 3.14")
    embedding_function = DefaultEmbeddingFunction()
    assert embedding_function.dimension == 384
    assert len(embedding_function("hello")[0]) == 384


def test_default_embedding_function_uses_onnx_on_pre314():
    if sys.version_info >= (3, 14):
        pytest.skip("Python >= 3.14")
    embedding_function = DefaultEmbeddingFunction()
    assert embedding_function.dimension == 384
    assert len(embedding_function("hello")[0]) == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
