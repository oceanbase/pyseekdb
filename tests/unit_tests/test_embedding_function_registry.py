"""
Unit tests for EmbeddingFunctionRegistry and register_embedding_function decorator.
"""

import pytest
from typing import Dict, Any

from pyseekdb.client.embedding_function import (
    EmbeddingFunction,
    EmbeddingFunctionRegistry,
    register_embedding_function,
    Documents,
    Embeddings,
    DefaultEmbeddingFunction,
)


class TestEmbeddingFunctionRegistry:
    """Test EmbeddingFunctionRegistry class"""

    def setup_method(self):
        """Reset registry state before each test"""
        # Clear registry and reset initialization state
        EmbeddingFunctionRegistry._registry.clear()
        EmbeddingFunctionRegistry._initialized = False

    def test_initialization_registers_default(self):
        """Test that initialization registers DefaultEmbeddingFunction"""
        # Force initialization
        EmbeddingFunctionRegistry._initialize()

        assert "default" in EmbeddingFunctionRegistry._registry
        assert (
            EmbeddingFunctionRegistry._registry["default"] == DefaultEmbeddingFunction
        )
        assert EmbeddingFunctionRegistry._initialized is True

    def test_initialization_idempotent(self):
        """Test that initialization is idempotent"""
        EmbeddingFunctionRegistry._initialize()
        first_registry = EmbeddingFunctionRegistry._registry.copy()

        # Initialize again
        EmbeddingFunctionRegistry._initialize()
        second_registry = EmbeddingFunctionRegistry._registry.copy()

        # Should be the same
        assert first_registry == second_registry

    def test_get_class_returns_default(self):
        """Test that get_class returns DefaultEmbeddingFunction for 'default'"""
        cls = EmbeddingFunctionRegistry.get_class("default")

        assert cls is not None
        assert cls == DefaultEmbeddingFunction

    def test_get_class_returns_none_for_unregistered(self):
        """Test that get_class returns None for unregistered names"""
        cls = EmbeddingFunctionRegistry.get_class("nonexistent_embedding")

        assert cls is None

    def test_register_valid_embedding_function(self):
        """Test registering a valid embedding function"""

        class TestEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, model_name: str = "test-model"):
                self.model_name = model_name

            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                return [[0.1, 0.2, 0.3] for _ in input]

            @staticmethod
            def name() -> str:
                return "test_embedding"

            def get_config(self) -> Dict[str, Any]:
                return {"model_name": self.model_name}

            @staticmethod
            def build_from_config(config: Dict[str, Any]) -> "TestEmbeddingFunction":
                return TestEmbeddingFunction(
                    model_name=config.get("model_name", "test-model")
                )

        # Register the class
        EmbeddingFunctionRegistry.register(TestEmbeddingFunction)

        # Verify it's registered
        assert "test_embedding" in EmbeddingFunctionRegistry.list_registered()
        cls = EmbeddingFunctionRegistry.get_class("test_embedding")
        assert cls == TestEmbeddingFunction

    def test_register_missing_name_method(self):
        """Test that registering a class without name() method raises ValueError"""

        class InvalidEmbeddingFunction(EmbeddingFunction[Documents]):
            def __call__(self, input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(config: Dict[str, Any]) -> "InvalidEmbeddingFunction":
                return InvalidEmbeddingFunction()

        with pytest.raises(ValueError, match="must have a static name\\(\\) method"):
            EmbeddingFunctionRegistry.register(InvalidEmbeddingFunction)

    def test_register_missing_build_from_config_method(self):
        """Test that registering a class without build_from_config() method raises ValueError"""

        class InvalidEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "invalid_embedding"

            def __call__(self, input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

        with pytest.raises(
            ValueError, match="must have.*build_from_config\\(\\) method"
        ):
            EmbeddingFunctionRegistry.register(InvalidEmbeddingFunction)

    def test_register_duplicate_name_raises_error(self):
        """Test that registering with duplicate name raises ValueError"""

        class FirstEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "duplicate_name"

            def __call__(self, _input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "FirstEmbeddingFunction":
                return FirstEmbeddingFunction()

        class SecondEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "duplicate_name"  # Same name!

            def __call__(self, input: Documents) -> Embeddings:
                return [[0.4, 0.5, 0.6]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "SecondEmbeddingFunction":
                return SecondEmbeddingFunction()

        # Register first one
        EmbeddingFunctionRegistry.register(FirstEmbeddingFunction)

        # Try to register second one with same name
        with pytest.raises(ValueError, match="is already registered"):
            EmbeddingFunctionRegistry.register(SecondEmbeddingFunction)

    def test_register_same_class_twice_allowed(self):
        """Test that registering the same class twice is allowed (idempotent)"""

        class TestEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "test_embedding"

            def __call__(self, input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "TestEmbeddingFunction":
                return TestEmbeddingFunction()

        # Register first time
        EmbeddingFunctionRegistry.register(TestEmbeddingFunction)
        first_count = len(EmbeddingFunctionRegistry.list_registered())

        # Register again (should be allowed)
        EmbeddingFunctionRegistry.register(TestEmbeddingFunction)
        second_count = len(EmbeddingFunctionRegistry.list_registered())

        # Should still be registered once
        assert first_count == second_count
        assert (
            EmbeddingFunctionRegistry.get_class("test_embedding")
            == TestEmbeddingFunction
        )

    def test_list_registered_returns_all_names(self):
        """Test that list_registered returns all registered names"""

        class TestEmbeddingFunction1(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "test1"

            def __call__(self, input: Documents) -> Embeddings:
                return [[0.1]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "TestEmbeddingFunction1":
                return TestEmbeddingFunction1()

        class TestEmbeddingFunction2(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "test2"

            def __call__(self, input: Documents) -> Embeddings:
                return [[0.2]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "TestEmbeddingFunction2":
                return TestEmbeddingFunction2()

        EmbeddingFunctionRegistry.register(TestEmbeddingFunction1)
        EmbeddingFunctionRegistry.register(TestEmbeddingFunction2)

        registered = EmbeddingFunctionRegistry.list_registered()
        assert "test1" in registered
        assert "test2" in registered
        assert "default" in registered  # Built-in
        assert len(registered) >= 3


class TestRegisterEmbeddingFunctionDecorator:
    """Test register_embedding_function decorator"""

    def setup_method(self):
        """Reset registry state before each test"""
        EmbeddingFunctionRegistry._registry.clear()
        EmbeddingFunctionRegistry._initialized = False

    def test_decorator_registers_class(self):
        """Test that decorator automatically registers the class"""

        @register_embedding_function
        class DecoratedEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "decorated_embedding"

            def __call__(self, _input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(
                _config: Dict[str, Any],
            ) -> type["DecoratedEmbeddingFunction"]:
                return DecoratedEmbeddingFunction()

        # Verify it's registered
        assert "decorated_embedding" in EmbeddingFunctionRegistry.list_registered()
        cls = EmbeddingFunctionRegistry.get_class("decorated_embedding")
        assert cls == DecoratedEmbeddingFunction

    def test_decorator_returns_same_class(self):
        """Test that decorator returns the same class (for chaining)"""

        @register_embedding_function
        class DecoratedEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "decorated_embedding"

            def __call__(self, _input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(
                _config: Dict[str, Any],
            ) -> type["DecoratedEmbeddingFunction"]:
                return DecoratedEmbeddingFunction()

        # The decorator should return the class itself
        assert DecoratedEmbeddingFunction.__name__ == "DecoratedEmbeddingFunction"
        assert callable(DecoratedEmbeddingFunction)

    def test_decorator_preserves_class_type(self):
        """Test that decorator preserves the exact class type for type checking"""

        @register_embedding_function
        class TypedEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, model_name: str = "test"):
                self.model_name = model_name

            @staticmethod
            def name() -> str:
                return "typed_embedding"

            def __call__(self, _input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {"model_name": self.model_name}

            @staticmethod
            def build_from_config(config: Dict[str, Any]) -> "TypedEmbeddingFunction":
                return TypedEmbeddingFunction(
                    model_name=config.get("model_name", "test")
                )

        # Type should be preserved - can instantiate and access attributes
        instance = TypedEmbeddingFunction(model_name="custom")
        assert instance.model_name == "custom"
        assert isinstance(instance, TypedEmbeddingFunction)

    def test_decorator_with_validation_errors(self):
        """Test that decorator raises ValueError for invalid classes"""

        with pytest.raises(ValueError, match="must have a static name\\(\\) method"):

            @register_embedding_function
            class InvalidEmbeddingFunction(EmbeddingFunction[Documents]):
                # Missing name() method
                def __call__(self, _input: Documents) -> Embeddings:
                    return [[0.1, 0.2, 0.3]]

                def get_config(self) -> Dict[str, Any]:
                    return {}

                @staticmethod
                def build_from_config(
                    _config: Dict[str, Any],
                ) -> "InvalidEmbeddingFunction":
                    return InvalidEmbeddingFunction()

    def test_decorator_works_with_initialization(self):
        """Test that decorator works correctly with registry initialization"""

        @register_embedding_function
        class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "custom_embedding"

            def __call__(self, _input: Documents) -> Embeddings:
                return [[0.1, 0.2, 0.3]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "CustomEmbeddingFunction":
                return CustomEmbeddingFunction()

        # Registry should be initialized
        assert EmbeddingFunctionRegistry._initialized is True

        # Both default and custom should be registered
        registered = EmbeddingFunctionRegistry.list_registered()
        assert "default" in registered
        assert "custom_embedding" in registered

    def test_multiple_decorated_classes(self):
        """Test that multiple decorated classes can coexist"""

        @register_embedding_function
        class FirstEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "first_embedding"

            def __call__(self, _input: Documents) -> Embeddings:
                return [[0.1]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "FirstEmbeddingFunction":
                return FirstEmbeddingFunction()

        @register_embedding_function
        class SecondEmbeddingFunction(EmbeddingFunction[Documents]):
            @staticmethod
            def name() -> str:
                return "second_embedding"

            def __call__(self, input: Documents) -> Embeddings:
                return [[0.2]]

            def get_config(self) -> Dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(_config: Dict[str, Any]) -> "SecondEmbeddingFunction":
                return SecondEmbeddingFunction()

        # Both should be registered
        registered = EmbeddingFunctionRegistry.list_registered()
        assert "first_embedding" in registered
        assert "second_embedding" in registered

        # Both should be retrievable
        first_cls = EmbeddingFunctionRegistry.get_class("first_embedding")
        second_cls = EmbeddingFunctionRegistry.get_class("second_embedding")

        assert first_cls == FirstEmbeddingFunction
        assert second_cls == SecondEmbeddingFunction


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
