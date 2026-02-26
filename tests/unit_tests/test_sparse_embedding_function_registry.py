"""
Unit tests for SparseEmbeddingFunction protocol, SparseEmbeddingFunctionRegistry,
and register_sparse_embedding_function decorator.
"""

from typing import Any

import pytest

from pyseekdb.client.sparse_embedding_function import (
    Documents,
    SparseEmbeddingFunction,
    SparseEmbeddingFunctionRegistry,
    SparseVector,
    SparseVectors,
    register_sparse_embedding_function,
)

# ── helpers ──────────────────────────────────────────────────────────


class _MockBM25(SparseEmbeddingFunction):
    """Minimal valid implementation used across many tests."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def __call__(self, documents: Documents) -> SparseVectors:
        if isinstance(documents, str):
            documents = [documents]
        return [SparseVector.from_dict({i: 0.1 * (i + 1) for i in range(len(doc.split()))}) for doc in documents]

    @staticmethod
    def name() -> str:
        return "mock_bm25"

    def get_config(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_MockBM25":
        return _MockBM25(k1=config.get("k1", 1.5), b=config.get("b", 0.75))


# ── SparseEmbeddingFunction protocol tests ───────────────────────────


class TestSparseEmbeddingFunctionProtocol:
    """Test SparseEmbeddingFunction protocol compliance"""

    def test_mock_implements_protocol(self):
        ef = _MockBM25()
        assert isinstance(ef, SparseEmbeddingFunction)

    def test_call_single_string(self):
        ef = _MockBM25()
        result = ef("hello world")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SparseVector)

    def test_call_list_of_strings(self):
        ef = _MockBM25()
        result = ef(["hello world", "foo bar baz"])
        assert len(result) == 2

    def test_name_returns_string(self):
        assert _MockBM25.name() == "mock_bm25"

    def test_get_config_returns_dict(self):
        ef = _MockBM25(k1=2.0, b=0.5)
        config = ef.get_config()
        assert config == {"k1": 2.0, "b": 0.5}

    def test_build_from_config_round_trip(self):
        original = _MockBM25(k1=2.0, b=0.5)
        config = original.get_config()
        restored = _MockBM25.build_from_config(config)
        assert restored.k1 == original.k1
        assert restored.b == original.b

    def test_support_persistence_valid(self):
        ef = _MockBM25()
        assert SparseEmbeddingFunction.support_persistence(ef) is True

    def test_support_persistence_none(self):
        assert SparseEmbeddingFunction.support_persistence(None) is False

    def test_support_persistence_no_name(self):
        class NoName:
            def __call__(self, docs):
                return []

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return NoName()

        assert SparseEmbeddingFunction.support_persistence(NoName()) is False

    def test_support_persistence_empty_name(self):
        class EmptyName:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return ""

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return EmptyName()

        assert SparseEmbeddingFunction.support_persistence(EmptyName()) is False

    def test_support_persistence_not_implemented_config(self):
        class NotImplConfig:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "test"

            def get_config(self):
                return NotImplemented

            @staticmethod
            def build_from_config(config):
                return NotImplConfig()

        assert SparseEmbeddingFunction.support_persistence(NotImplConfig()) is False


# ── SparseEmbeddingFunctionRegistry tests ────────────────────────────


class TestSparseEmbeddingFunctionRegistry:
    """Test SparseEmbeddingFunctionRegistry class"""

    def setup_method(self):
        SparseEmbeddingFunctionRegistry._registry.clear()
        SparseEmbeddingFunctionRegistry._initialized = False

    def test_register_valid_class(self):
        SparseEmbeddingFunctionRegistry.register(_MockBM25)
        assert "mock_bm25" in SparseEmbeddingFunctionRegistry.list_registered()

    def test_get_class_returns_registered(self):
        SparseEmbeddingFunctionRegistry.register(_MockBM25)
        cls = SparseEmbeddingFunctionRegistry.get_class("mock_bm25")
        assert cls is _MockBM25

    def test_get_class_returns_none_for_unknown(self):
        cls = SparseEmbeddingFunctionRegistry.get_class("nonexistent")
        assert cls is None

    def test_register_missing_name_raises(self):
        class NoName:
            def __call__(self, docs):
                return []

            @staticmethod
            def build_from_config(config):
                return NoName()

        with pytest.raises(ValueError, match="must have a static name"):
            SparseEmbeddingFunctionRegistry.register(NoName)

    def test_register_missing_build_from_config_raises(self):
        class NoBuild:
            @staticmethod
            def name():
                return "no_build"

        with pytest.raises(ValueError, match=r"must have.*build_from_config"):
            SparseEmbeddingFunctionRegistry.register(NoBuild)

    def test_register_duplicate_name_different_class_raises(self):
        SparseEmbeddingFunctionRegistry.register(_MockBM25)

        class AnotherBM25:
            @staticmethod
            def name():
                return "mock_bm25"

            @staticmethod
            def build_from_config(config):
                return AnotherBM25()

        with pytest.raises(ValueError, match="is already registered"):
            SparseEmbeddingFunctionRegistry.register(AnotherBM25)

    def test_register_same_class_twice_allowed(self):
        SparseEmbeddingFunctionRegistry.register(_MockBM25)
        SparseEmbeddingFunctionRegistry.register(_MockBM25)
        assert SparseEmbeddingFunctionRegistry.get_class("mock_bm25") is _MockBM25

    def test_list_registered(self):
        SparseEmbeddingFunctionRegistry.register(_MockBM25)

        class SecondEF:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "second_ef"

            @staticmethod
            def build_from_config(config):
                return SecondEF()

        SparseEmbeddingFunctionRegistry.register(SecondEF)
        names = SparseEmbeddingFunctionRegistry.list_registered()
        assert "mock_bm25" in names
        assert "second_ef" in names

    def test_build_from_config(self):
        SparseEmbeddingFunctionRegistry.register(_MockBM25)
        ef = SparseEmbeddingFunctionRegistry.build_from_config("mock_bm25", {"k1": 2.0, "b": 0.5})
        assert isinstance(ef, _MockBM25)
        assert ef.k1 == 2.0
        assert ef.b == 0.5

    def test_build_from_config_unregistered_raises(self):
        with pytest.raises(ValueError, match="is not registered"):
            SparseEmbeddingFunctionRegistry.build_from_config("unknown", {})

    def test_initialization_is_idempotent(self):
        SparseEmbeddingFunctionRegistry._initialize()
        first = SparseEmbeddingFunctionRegistry._registry.copy()
        SparseEmbeddingFunctionRegistry._initialize()
        second = SparseEmbeddingFunctionRegistry._registry.copy()
        assert first == second


# ── Decorator tests ──────────────────────────────────────────────────


class TestRegisterSparseEmbeddingFunctionDecorator:
    """Test register_sparse_embedding_function decorator"""

    def setup_method(self):
        SparseEmbeddingFunctionRegistry._registry.clear()
        SparseEmbeddingFunctionRegistry._initialized = False

    def test_decorator_registers_class(self):
        @register_sparse_embedding_function
        class DecoratedEF:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "decorated_ef"

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return DecoratedEF()

        assert "decorated_ef" in SparseEmbeddingFunctionRegistry.list_registered()

    def test_decorator_returns_same_class(self):
        @register_sparse_embedding_function
        class DecoratedEF:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "decorated_ef2"

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return DecoratedEF()

        assert DecoratedEF.__name__ == "DecoratedEF"
        assert callable(DecoratedEF)

    def test_decorator_preserves_instantiation(self):
        @register_sparse_embedding_function
        class DecoratedEF:
            def __init__(self, param: str = "default"):
                self.param = param

            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "decorated_ef3"

            def get_config(self):
                return {"param": self.param}

            @staticmethod
            def build_from_config(config):
                return DecoratedEF(param=config.get("param", "default"))

        instance = DecoratedEF(param="custom")
        assert instance.param == "custom"

    def test_decorator_invalid_class_raises(self):
        with pytest.raises(ValueError, match="must have a static name"):

            @register_sparse_embedding_function
            class InvalidEF:
                def __call__(self, docs):
                    return []

                @staticmethod
                def build_from_config(config):
                    return InvalidEF()

    def test_multiple_decorated_classes(self):
        @register_sparse_embedding_function
        class EF1:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "ef1"

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return EF1()

        @register_sparse_embedding_function
        class EF2:
            def __call__(self, docs):
                return []

            @staticmethod
            def name():
                return "ef2"

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return EF2()

        registered = SparseEmbeddingFunctionRegistry.list_registered()
        assert "ef1" in registered
        assert "ef2" in registered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
