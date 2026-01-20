"""
Test collection creation with embedding function using db_client fixture
Testing create_collection, get_or_create_collection, and get_collection interfaces
with embedding function handling
"""

import contextlib
import time

import pytest

from pyseekdb import DefaultEmbeddingFunction, HNSWConfiguration


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
        assert collection.embedding_function == custom_ef
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
