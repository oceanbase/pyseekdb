"""
Official example test case - verifies the documented quick-start workflow.

The scenario mirrors `pyseekdb/examples/official_example.py` and covers:
1. Creating a default client (embedded/server/OceanBase, configurable by env vars)
2. Creating a collection via get_or_create_collection
3. Upserting only documents/metadatas/ids (relying on default embedding function)
4. Querying with query_texts + metadata filter + document filter
"""
import os
import sys
import time
from pathlib import Path

import pytest

# Ensure the project root is importable when tests run in isolation
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb  # noqa: E402


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get("SEEKDB_PATH", os.path.join(project_root, "seekdb_store"))
SEEKDB_DATABASE = os.environ.get("SEEKDB_DATABASE", "test")

# Server mode
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "2881"))
SERVER_DATABASE = os.environ.get("SERVER_DATABASE", "test")
SERVER_USER = os.environ.get("SERVER_USER", "root")
SERVER_PASSWORD = os.environ.get("SERVER_PASSWORD", "")

# OceanBase mode
OB_HOST = os.environ.get("OB_HOST", "localhost")
OB_PORT = int(os.environ.get("OB_PORT", "11202"))
OB_TENANT = os.environ.get("OB_TENANT", "mysql")
OB_DATABASE = os.environ.get("OB_DATABASE", "test")
OB_USER = os.environ.get("OB_USER", "root")
OB_PASSWORD = os.environ.get("OB_PASSWORD", "")


PRODUCT_DOCUMENTS = [
    "Laptop Pro with 16GB RAM, 512GB SSD, and high-speed processor",
    "Gaming Laptop with 32GB RAM, 1TB SSD, and high-performance graphics",
    "Business Ultrabook with 8GB RAM, 256GB SSD, and long battery life",
    "Tablet with 6GB RAM, 128GB storage, and 10-inch display",
]

PRODUCT_METADATA = [
    {"category": "laptop", "ram": 16, "storage": 512, "price": 12000, "type": "professional"},
    {"category": "laptop", "ram": 32, "storage": 1000, "price": 25000, "type": "gaming"},
    {"category": "laptop", "ram": 8, "storage": 256, "price": 9000, "type": "business"},
    {"category": "tablet", "ram": 6, "storage": 128, "price": 6000, "type": "consumer"},
]

PRODUCT_IDS = ["1", "2", "3", "4"]


def _run_official_example(collection):
    """Execute the official example workflow against the provided collection."""
    collection.upsert(
        documents=PRODUCT_DOCUMENTS,
        metadatas=PRODUCT_METADATA,
        ids=PRODUCT_IDS,
    )

    results = collection.query(
        query_texts=["powerful computer for professional work"],
        where={
            "category": "laptop",
            "ram": {"$gte": 16},
        },
        where_document={"$contains": "RAM"},
        n_results=2,
        include=["documents", "metadatas", "ids"],
    )

    assert results is not None
    assert "documents" in results
    assert len(results["documents"]) > 0
    assert len(results["documents"][0]) > 0, "Expected at least one matched document"

    matched_docs = results["documents"][0]
    matched_metadata = results["metadatas"][0]

    for doc in matched_docs:
        assert doc is None or "ram" in doc.lower()

    for metadata in matched_metadata:
        if metadata:
            assert metadata.get("category") == "laptop"
            assert metadata.get("ram", 0) >= 16

    return results


class TestOfficialExample:
    """Test suite that mirrors the official example across deployment modes."""

    def _cleanup_collection(self, client, name: str):
        try:
            client.delete_collection(name=name)
        except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
            print(f"Warning: failed to cleanup collection '{name}': {cleanup_error}")

    def _create_collection(self, client):
        collection_name = f"official_example_{int(time.time())}"
        collection = client.get_or_create_collection(name=collection_name)
        return collection_name, collection

    def test_embedded_official_example(self):
        """Official example using embedded client (SeekdbEmbedded)."""
        try:
            import pylibseekdb  # noqa: F401
        except ImportError:
            pytest.fail("SeekDB embedded package is not installed")

        client = pyseekdb.Client(path=SEEKDB_PATH, database=SEEKDB_DATABASE)
        collection_name, collection = self._create_collection(client)

        try:
            _run_official_example(collection)
        finally:
            self._cleanup_collection(client, collection_name)

    def test_server_official_example(self):
        """Official example using SeekDB server (RemoteServerClient default tenant)."""
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD,
        )

        try:
            result = client._server.execute("SELECT 1 as test")
            assert result and result[0].get("test", 1) == 1
        except Exception as exc:
            pytest.fail(f"SeekDB server connection failed ({SERVER_HOST}:{SERVER_PORT}): {exc}")

        collection_name, collection = self._create_collection(client)

        try:
            _run_official_example(collection)
        finally:
            self._cleanup_collection(client, collection_name)

    def test_oceanbase_official_example(self):
        """Official example using OceanBase deployment."""
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD,
        )

        try:
            result = client._server.execute("SELECT 1 as test")
            assert result and result[0].get("test", 1) == 1
        except Exception as exc:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {exc}")

        collection_name, collection = self._create_collection(client)

        try:
            _run_official_example(collection)
        finally:
            self._cleanup_collection(client, collection_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

