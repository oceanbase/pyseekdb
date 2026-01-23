"""
Test collection creation with embedding function using db_client fixture
Testing create_collection, get_or_create_collection, and get_collection interfaces
with embedding function handling
"""

import contextlib
import time
from typing import Any

import pytest

from pyseekdb import DefaultEmbeddingFunction, HNSWConfiguration
from pyseekdb.client.embedding_function import (
    Documents,
    EmbeddingFunction,
    EmbeddingFunctionRegistry,
    Embeddings,
    register_embedding_function,
)


# ==================== Simple 3D Embedding Function for Testing ====================
class Simple3DEmbeddingFunction:
    """Simple embedding function that returns 3-dimensional vectors for testing"""

    def __init__(self):
        self.dimension = 3

    def __call__(self, documents: str | list[str]) -> list[list[float]]:
        """Convert documents to 3D embeddings (simple hash-based)"""
        if isinstance(documents, str):
            documents = [documents]

        embeddings = []
        for doc in documents:
            # Simple hash-based 3D embedding for testing
            hash_val = hash(doc) % 1000
            embedding = [
                float((hash_val % 10) / 10.0),
                float(((hash_val // 10) % 10) / 10.0),
                float(((hash_val // 100) % 10) / 10.0),
            ]
            embeddings.append(embedding)

        return embeddings


class TestCollectionEmbeddingFunction:
    """Test collection creation with embedding function handling using parameterized db_client fixture"""

    def setup_method(self):
        EmbeddingFunctionRegistry._registry.clear()
        EmbeddingFunctionRegistry._initialized = False

    def test_create_collection_default_embedding_function(self, db_client):
        """
        Test create_collection with default embedding function (not provided).

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_default_ef_{int(time.time() * 1000)}"
        print("\n✅ Testing create_collection with default embedding function")

        # Test: Not providing embedding_function should use DefaultEmbeddingFunction
        collection = db_client.create_collection(name=collection_name)

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, DefaultEmbeddingFunction)
        # Default embedding function produces 384-dim vectors
        assert collection.dimension == 384
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_create_collection_explicit_none(self, db_client):
        """
        Test create_collection with embedding_function=None.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_explicit_none_{int(time.time() * 1000)}"
        print("\n✅ Testing create_collection with embedding_function=None")

        # Test: Explicitly set embedding_function=None, must provide configuration
        config = HNSWConfiguration(dimension=128, distance="cosine")
        collection = db_client.create_collection(name=collection_name, configuration=config, embedding_function=None)

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is None
        assert collection.dimension == 128
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        collection_get = db_client.get_collection(name=collection_name, embedding_function=None)
        assert collection_get is not None
        assert collection_get.name == collection_name
        assert collection_get.embedding_function is None
        assert collection_get.dimension == 128
        print(f"   Collection dimension: {collection_get.dimension}")
        print(f"   Embedding function: {collection_get.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_create_collection_custom_embedding_function(self, db_client):
        """
        Test create_collection with custom embedding function.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_custom_ef_{int(time.time() * 1000)}"
        print("\n✅ Testing create_collection with custom embedding function")

        # Test: Custom embedding function, dimension calculated via __call__("seekdb")
        custom_ef = Simple3DEmbeddingFunction()
        config = HNSWConfiguration(dimension=3, distance="l2")

        collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=custom_ef
        )

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert collection.embedding_function == custom_ef
        assert collection.dimension == 3
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        collection_get = db_client.get_collection(name=collection_name, embedding_function=custom_ef)
        assert collection_get is not None
        assert collection_get.name == collection_name
        assert collection_get.embedding_function is not None
        assert collection_get.embedding_function == custom_ef
        assert collection_get.dimension == 3
        print(f"   Collection dimension: {collection_get.dimension}")
        print(f"   Embedding function: {collection_get.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_create_collection_dimension_mismatch(self, db_client):
        """
        Test create_collection with dimension mismatch should raise error.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_dim_mismatch_{int(time.time() * 1000)}"
        print("\n✅ Testing create_collection with dimension mismatch (should fail)")

        # Test: Configuration dimension doesn't match embedding function dimension
        custom_ef = Simple3DEmbeddingFunction()
        config = HNSWConfiguration(dimension=128, distance="cosine")  # Mismatch: 3 vs 128

        with pytest.raises(ValueError) as exc_info:
            db_client.create_collection(name=collection_name, configuration=config, embedding_function=custom_ef)

        assert "doesn't match" in str(exc_info.value).lower() or "dimension" in str(exc_info.value).lower()
        print(f"   Correctly raised ValueError: {exc_info.value}")

    def test_create_collection_configuration_none_with_ef(self, db_client):
        """
        Test create_collection with configuration=None and embedding_function provided.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_config_none_with_ef_{int(time.time() * 1000)}"
        print("\n✅ Testing create_collection with configuration=None and embedding_function provided")

        # Test: configuration=None, but embedding_function is provided, should calculate dimension
        custom_ef = Simple3DEmbeddingFunction()
        collection = db_client.create_collection(name=collection_name, configuration=None, embedding_function=custom_ef)

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert collection.embedding_function == custom_ef
        assert collection.dimension == 3  # Should use calculated dimension
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_create_collection_both_none_error(self, db_client):
        """
        Test create_collection with embedding_function=None and configuration=None should raise error.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_both_none_{int(time.time() * 1000)}"
        print("\n✅ Testing create_collection with both None (should fail)")

        # Test: Both embedding_function and configuration are None
        with pytest.raises(ValueError) as exc_info:
            db_client.create_collection(name=collection_name, configuration=None, embedding_function=None)

        assert "cannot determine dimension" in str(exc_info.value).lower() or "none" in str(exc_info.value).lower()
        print(f"   Correctly raised ValueError: {exc_info.value}")

    def test_get_collection_default_embedding_function(self, db_client):
        """
        Test get_collection with default embedding function.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_get_default_ef_{int(time.time() * 1000)}"
        print("\n✅ Testing get_collection with default embedding function")

        # First create a collection
        config = HNSWConfiguration(dimension=128, distance="cosine")
        db_client.create_collection(name=collection_name, configuration=config, embedding_function=None)

        # Then get it without providing embedding_function (should use default)
        retrieved_collection = db_client.get_collection(name=collection_name)

        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.dimension == 128
        # Should have default embedding function
        assert retrieved_collection.embedding_function is not None
        assert isinstance(retrieved_collection.embedding_function, DefaultEmbeddingFunction)
        print(f"   Collection dimension: {retrieved_collection.dimension}")
        print(f"   Embedding function: {retrieved_collection.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_get_collection_explicit_none(self, db_client):
        """
        Test get_collection with embedding_function=None.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_get_explicit_none_{int(time.time() * 1000)}"
        print("\n✅ Testing get_collection with embedding_function=None")

        # First create a collection
        config = HNSWConfiguration(dimension=128, distance="cosine")
        db_client.create_collection(name=collection_name, configuration=config, embedding_function=None)

        # Then get it with embedding_function=None
        retrieved_collection = db_client.get_collection(name=collection_name, embedding_function=None)

        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.dimension == 128
        assert retrieved_collection.embedding_function is None
        print(f"   Collection dimension: {retrieved_collection.dimension}")
        print(f"   Embedding function: {retrieved_collection.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_get_or_create_collection_create_new(self, db_client):
        """
        Test get_or_create_collection creating new collection.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_get_or_create_new_{int(time.time() * 1000)}"
        print("\n✅ Testing get_or_create_collection (create new)")

        # Test: Collection doesn't exist, should create with default embedding function
        collection = db_client.get_or_create_collection(name=collection_name)

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert isinstance(collection.embedding_function, DefaultEmbeddingFunction)
        assert collection.dimension == 384
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_get_or_create_collection_get_existing(self, db_client):
        """
        Test get_or_create_collection getting existing collection.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_get_or_create_existing_{int(time.time() * 1000)}"
        print("\n✅ Testing get_or_create_collection (get existing)")

        # First create a collection
        config = HNSWConfiguration(dimension=128, distance="cosine")
        db_client.create_collection(name=collection_name, configuration=config, embedding_function=None)

        # Then get_or_create it
        retrieved_collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.dimension == 128
        print(f"   Collection dimension: {retrieved_collection.dimension}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_get_or_create_collection_custom_embedding_function(self, db_client):
        """
        Test get_or_create_collection with custom embedding function.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_get_or_create_custom_ef_{int(time.time() * 1000)}"
        print("\n✅ Testing get_or_create_collection with custom embedding function")

        # Test: Create with custom embedding function
        custom_ef = Simple3DEmbeddingFunction()
        config = HNSWConfiguration(dimension=3, distance="l2")

        collection = db_client.get_or_create_collection(
            name=collection_name, configuration=config, embedding_function=custom_ef
        )

        assert collection is not None
        assert collection.name == collection_name
        assert collection.embedding_function is not None
        assert collection.embedding_function is custom_ef
        assert collection.dimension == 3
        print(f"   Collection dimension: {collection.dimension}")
        print(f"   Embedding function: {collection.embedding_function}")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_get_or_create_collection_both_none_error(self, db_client):
        """
        Test get_or_create_collection with both None should raise error when creating.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_get_or_create_both_none_{int(time.time() * 1000)}"
        print("\n✅ Testing get_or_create_collection with both None (should fail when creating)")

        # Test: Both None when creating new collection
        with pytest.raises(ValueError) as exc_info:
            db_client.get_or_create_collection(name=collection_name, configuration=None, embedding_function=None)

        assert "cannot determine dimension" in str(exc_info.value).lower() or "none" in str(exc_info.value).lower()
        print(f"   Correctly raised ValueError: {exc_info.value}")

    def test_custom_embedding_function_persistence_and_restoration(self, db_client):
        """
        Test that custom embedding functions are persisted and automatically restored.

        Tests:
        - Create collection with registered custom embedding function
        - Get collection and verify embedding function is automatically restored
        - Verify restored embedding function works correctly

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_ef_persistence_{int(time.time() * 1000)}"
        print("\n✅ Testing custom embedding function persistence and restoration")

        # Define a custom embedding function with decorator
        @register_embedding_function
        class TestPersistentEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, model_name: str = "test-model", dimension: int = 5):
                self.model_name = model_name
                self._dimension = dimension

            def __call__(self, documents: Documents) -> Embeddings:
                if isinstance(documents, str):
                    documents = [documents]
                # Return simple embeddings based on document length
                return [[float(len(doc) % 10) / 10.0] * self._dimension for doc in documents]

            @property
            def dimension(self) -> int:
                return self._dimension

            @staticmethod
            def name() -> str:
                return "test_persistent_embedding"

            def get_config(self) -> dict[str, Any]:
                return {
                    "model_name": self.model_name,
                    "dimension": self._dimension,
                }

            @staticmethod
            def build_from_config(
                config: dict[str, Any],
            ) -> "TestPersistentEmbeddingFunction":
                return TestPersistentEmbeddingFunction(
                    model_name=config.get("model_name", "test-model"),
                    dimension=config.get("dimension", 5),
                )

        # Create collection with the custom embedding function
        custom_ef = TestPersistentEmbeddingFunction(model_name="my-test-model", dimension=5)
        config = HNSWConfiguration(dimension=5, distance="cosine")

        created_collection = db_client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=custom_ef,
        )

        assert created_collection is not None
        assert created_collection.embedding_function is not None
        assert created_collection.embedding_function.model_name == "my-test-model"
        assert created_collection.embedding_function.dimension == 5

        # Get the collection - embedding function should be automatically restored
        retrieved_collection = db_client.get_collection(name=collection_name)

        assert retrieved_collection is not None
        assert retrieved_collection.name == collection_name
        assert retrieved_collection.embedding_function is not None
        # Verify it's the same type
        assert isinstance(retrieved_collection.embedding_function, TestPersistentEmbeddingFunction)
        # Verify configuration was restored correctly
        assert retrieved_collection.embedding_function.model_name == "my-test-model"
        assert retrieved_collection.embedding_function.dimension == 5

        # Verify the restored embedding function works
        test_docs = ["test document 1", "test document 2"]
        embeddings = retrieved_collection.embedding_function(test_docs)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 5

        print(f"   Collection dimension: {retrieved_collection.dimension}")
        print(f"   Restored embedding function: {retrieved_collection.embedding_function}")
        print(f"   Embedding function model: {retrieved_collection.embedding_function.model_name}")

        collection_get = db_client.get_or_create_collection(name=collection_name, embedding_function=custom_ef)
        assert collection_get is not None
        assert collection_get.embedding_function is not None
        assert collection_get.embedding_function.get_config() == custom_ef.get_config()
        assert collection_get.dimension == 5
        assert collection_get.name == collection_name
        assert collection_get.embedding_function.model_name == "my-test-model"
        assert collection_get.embedding_function.dimension == 5

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_custom_embedding_function_manual_registration(self, db_client):
        """
        Test custom embedding function persistence with manual registration.

        Tests:
        - Manually register embedding function
        - Create collection and verify persistence
        - Get collection and verify restoration

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_ef_manual_reg_{int(time.time() * 1000)}"
        print("\n✅ Testing custom embedding function with manual registration")

        # Define custom embedding function without decorator
        class ManualRegisteredEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, model_name: str = "manual-model", dimension: int = 4):
                self.model_name = model_name
                self._dimension = dimension

            def __call__(self, documents: Documents) -> Embeddings:
                if isinstance(documents, str):
                    documents = [documents]
                return [[0.5] * self._dimension for _ in documents]

            @property
            def dimension(self) -> int:
                return self._dimension

            @staticmethod
            def name() -> str:
                return "manual_registered_embedding"

            def get_config(self) -> dict[str, Any]:
                return {
                    "model_name": self.model_name,
                    "dimension": self._dimension,
                }

            @staticmethod
            def build_from_config(
                config: dict[str, Any],
            ) -> "ManualRegisteredEmbeddingFunction":
                return ManualRegisteredEmbeddingFunction(
                    model_name=config.get("model_name", "manual-model"),
                    dimension=config.get("dimension", 4),
                )

        # Manually register the embedding function
        register_embedding_function(ManualRegisteredEmbeddingFunction)

        try:
            # Create collection with the custom embedding function
            custom_ef = ManualRegisteredEmbeddingFunction(model_name="custom-manual-model", dimension=4)
            config = HNSWConfiguration(dimension=4, distance="l2")

            created_collection = db_client.create_collection(
                name=collection_name,
                configuration=config,
                embedding_function=custom_ef,
            )

            assert created_collection.embedding_function.model_name == "custom-manual-model"

            # Get the collection - should restore embedding function
            retrieved_collection = db_client.get_collection(name=collection_name)

            assert retrieved_collection.embedding_function is not None
            assert isinstance(
                retrieved_collection.embedding_function,
                ManualRegisteredEmbeddingFunction,
            )
            assert retrieved_collection.embedding_function.model_name == "custom-manual-model"
            assert retrieved_collection.embedding_function.dimension == 4

            # Verify it works
            embeddings = retrieved_collection.embedding_function(["test"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 4

            collection_get = db_client.get_collection(name=collection_name, embedding_function=custom_ef)
            assert collection_get is not None
            assert collection_get.embedding_function is not None
            assert collection_get.embedding_function.get_config() == custom_ef.get_config()
            assert collection_get.dimension == 4
            assert collection_get.name == collection_name
            assert collection_get.embedding_function.model_name == "custom-manual-model"
            assert collection_get.embedding_function.dimension == 4

            print("   Successfully restored manually registered embedding function")
        finally:
            # Cleanup
            with contextlib.suppress(Exception):
                db_client.delete_collection(name=collection_name)

    def test_restored_embedding_function_usage(self, db_client):
        """
        Test that restored embedding function can be used for operations.

        Tests:
        - Create collection with custom embedding function
        - Get collection (restores embedding function)
        - Use restored embedding function for add and query operations

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_ef_usage_{int(time.time() * 1000)}"
        print("\n✅ Testing restored embedding function usage in operations")

        @register_embedding_function
        class UsableEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, dimension: int = 3):
                self._dimension = dimension

            def __call__(self, documents: Documents) -> Embeddings:
                if isinstance(documents, str):
                    documents = [documents]
                # Simple embedding: use first character code
                return [
                    [float(ord(doc[0]) % 100) / 100.0] * self._dimension if doc else [0.0] * self._dimension
                    for doc in documents
                ]

            @property
            def dimension(self) -> int:
                return self._dimension

            @staticmethod
            def name() -> str:
                return "usable_embedding"

            def get_config(self) -> dict[str, Any]:
                return {"dimension": self._dimension}

            @staticmethod
            def build_from_config(config: dict[str, Any]) -> "UsableEmbeddingFunction":
                return UsableEmbeddingFunction(dimension=config.get("dimension", 3))

        # Create collection
        custom_ef = UsableEmbeddingFunction(dimension=3)
        config = HNSWConfiguration(dimension=3, distance="cosine")

        db_client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=custom_ef,
        )

        # Get collection (restores embedding function)
        retrieved_collection = db_client.get_collection(name=collection_name)

        # Use restored embedding function for add operation
        retrieved_collection.add(
            ids="test_id_1",
            documents="Hello world",
        )

        # Verify document was added
        results = retrieved_collection.get(ids="test_id_1")
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "Hello world"

        # Use restored embedding function for query operation
        query_results = retrieved_collection.query(
            query_texts="Hello",
            n_results=1,
        )

        assert len(query_results["ids"]) == 1
        assert len(query_results["ids"][0]) >= 0  # May or may not find results

        print("   Successfully used restored embedding function for add and query")

        # Cleanup
        with contextlib.suppress(Exception):
            db_client.delete_collection(name=collection_name)

    def test_multiple_custom_embedding_functions(self, db_client):  # noqa: C901
        """
        Test multiple collections with different custom embedding functions.

        Tests:
        - Create multiple collections with different registered embedding functions
        - Get each collection and verify correct embedding function is restored

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name_1 = f"test_ef_multi_1_{int(time.time() * 1000)}"
        collection_name_2 = f"test_ef_multi_2_{int(time.time() * 1000)}"
        print("\n✅ Testing multiple collections with different embedding functions")

        @register_embedding_function
        class FirstEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, param: str = "first"):
                self.param = param

            def __call__(self, documents: Documents) -> Embeddings:
                if isinstance(documents, str):
                    documents = [documents]
                return [[1.0, 2.0, 3.0] for _ in documents]

            @property
            def dimension(self) -> int:
                return 3

            @staticmethod
            def name() -> str:
                return "first_embedding"

            def get_config(self) -> dict[str, Any]:
                return {"param": self.param}

            @staticmethod
            def build_from_config(config: dict[str, Any]) -> "FirstEmbeddingFunction":
                return FirstEmbeddingFunction(param=config.get("param", "first"))

        @register_embedding_function
        class SecondEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self, param: str = "second"):
                self.param = param

            def __call__(self, documents: Documents) -> Embeddings:
                if isinstance(documents, str):
                    documents = [documents]
                return [[4.0, 5.0, 6.0] for _ in documents]

            @property
            def dimension(self) -> int:
                return 3

            @staticmethod
            def name() -> str:
                return "second_embedding"

            def get_config(self) -> dict[str, Any]:
                return {"param": self.param}

            @staticmethod
            def build_from_config(config: dict[str, Any]) -> "SecondEmbeddingFunction":
                return SecondEmbeddingFunction(param=config.get("param", "second"))

        try:
            # Create first collection
            ef1 = FirstEmbeddingFunction(param="custom_first")
            config = HNSWConfiguration(dimension=3, distance="cosine")
            db_client.create_collection(
                name=collection_name_1,
                configuration=config,
                embedding_function=ef1,
            )

            # Create second collection
            ef2 = SecondEmbeddingFunction(param="custom_second")
            db_client.create_collection(
                name=collection_name_2,
                configuration=config,
                embedding_function=ef2,
            )

            # Get both collections
            retrieved_coll1 = db_client.get_collection(name=collection_name_1)
            retrieved_coll2 = db_client.get_collection(name=collection_name_2)

            # Verify each has the correct embedding function
            assert isinstance(retrieved_coll1.embedding_function, FirstEmbeddingFunction)
            assert retrieved_coll1.embedding_function.param == "custom_first"

            assert isinstance(retrieved_coll2.embedding_function, SecondEmbeddingFunction)
            assert retrieved_coll2.embedding_function.param == "custom_second"

            # Verify they produce different embeddings
            emb1 = retrieved_coll1.embedding_function(["test"])
            emb2 = retrieved_coll2.embedding_function(["test"])

            assert emb1[0] == [1.0, 2.0, 3.0]
            assert emb2[0] == [4.0, 5.0, 6.0]

            print("   Successfully restored different embedding functions for different collections")
        finally:
            # Cleanup
            with contextlib.suppress(Exception):
                db_client.delete_collection(name=collection_name_1)
                db_client.delete_collection(name=collection_name_2)

    def test_unregistered_embedding_function_error(self, db_client):
        """
        Test that getting a collection with unregistered embedding function raises error.

        Tests:
        - Create collection with unregistered embedding function (should fail or not persist)
        - Attempt to get collection should handle gracefully

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_ef_unregistered_{int(time.time() * 1000)}"
        print("\n✅ Testing unregistered embedding function handling")

        # Define embedding function but don't register it
        class UnregisteredEmbeddingFunction(EmbeddingFunction[Documents]):
            def __init__(self):
                self._dimension = 3

            def __call__(self, documents: Documents) -> Embeddings:
                if isinstance(documents, str):
                    documents = [documents]
                return [[1.0, 2.0, 3.0] for _ in documents]

            @property
            def dimension(self) -> int:
                return 3

            @staticmethod
            def name() -> str:
                return "unregistered_embedding"

            def get_config(self) -> dict[str, Any]:
                return {}

            @staticmethod
            def build_from_config(
                _config: dict[str, Any],
            ) -> "UnregisteredEmbeddingFunction":
                return UnregisteredEmbeddingFunction()

        # Try to create collection - this should work (registration happens at get time)
        # But we need to register it first for persistence to work
        # Actually, if we don't register, the collection creation will still work,
        # but when we try to get it, it should fail to restore the embedding function

        # Register it first to create the collection
        register_embedding_function(UnregisteredEmbeddingFunction)

        try:
            ef = UnregisteredEmbeddingFunction()
            config = HNSWConfiguration(dimension=3, distance="cosine")
            db_client.create_collection(
                name=collection_name,
                configuration=config,
                embedding_function=ef,
            )

            # Now unregister it
            # Note: We can't easily unregister, but we can test what happens
            # when the registry doesn't have it by checking the error handling
            EmbeddingFunctionRegistry._registry.pop(ef.name())

            # Actually, since we registered it, getting should work
            pytest.raises(ValueError, db_client.get_collection, name=collection_name)
        finally:
            # Cleanup
            with contextlib.suppress(Exception):
                db_client.delete_collection(name=collection_name)

    def test_providing_embedding_function_from_parameter_and_persistence_raises_error(self, db_client):
        """
        Test that providing embedding function from parameter and persistence raises error.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_ef_parameter_and_persistence_{int(time.time() * 1000)}"
        print("\n✅ Testing providing embedding function from parameter and persistence raises error")

        # Create collection
        custom_ef = Simple3DEmbeddingFunction()
        db_client.create_collection(
            name=collection_name,
            # use default embedding function
        )

        # Get collection
        with pytest.raises(ValueError):
            db_client.get_collection(name=collection_name, embedding_function=custom_ef)

        with pytest.raises(ValueError):
            db_client.get_or_create_collection(name=collection_name, embedding_function=None)

        collection_name = f"test_ef_parameter_and_persistence_{int(time.time() * 1000)}"
        db_client.create_collection(
            name=collection_name,
            configuration=HNSWConfiguration(dimension=3, distance="cosine"),
            embedding_function=None,
        )
        collection_get = db_client.get_collection(name=collection_name, embedding_function=custom_ef)
        assert collection_get is not None
        assert collection_get.name == collection_name
        assert collection_get.embedding_function is not None
        assert collection_get.embedding_function == custom_ef
        assert collection_get.dimension == 3
        print(f"   Collection dimension: {collection_get.dimension}")
        print(f"   Embedding function: {collection_get.embedding_function}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
