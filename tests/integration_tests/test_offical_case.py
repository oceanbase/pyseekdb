"""
Official example test case using db_client fixture
Verifies the documented quick-start workflow.

The scenario mirrors `pyseekdb/examples/official_example.py` and covers:
1. Creating a default client (embedded/server/OceanBase, configurable by env vars)
2. Creating a collection via get_or_create_collection
3. Upserting only documents/metadatas/ids (relying on default embedding function)
4. Querying with query_texts + metadata filter + document filter
"""

import time
import pytest

import pyseekdb


PRODUCT_DOCUMENTS = [
    "Laptop Pro with 16GB RAM, 512GB SSD, and high-speed processor",
    "Gaming Laptop with 32GB RAM, 1TB SSD, and high-performance graphics",
    "Business Ultrabook with 8GB RAM, 256GB SSD, and long battery life",
    "Tablet with 6GB RAM, 128GB storage, and 10-inch display",
]

PRODUCT_METADATA = [
    {
        "category": "laptop",
        "ram": 16,
        "storage": 512,
        "price": 12000,
        "type": "professional",
    },
    {
        "category": "laptop",
        "ram": 32,
        "storage": 1000,
        "price": 25000,
        "type": "gaming",
    },
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
    """Test suite that mirrors the official example using parameterized db_client fixture."""

    def test_official_example(self, db_client):
        """
        Official example using client (automatic mode selection).

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"official_example_{int(time.time() * 1000)}"
        collection = db_client.get_or_create_collection(name=collection_name)

        # Run the official example workflow
        _run_official_example(collection)

        # Note: cleanup is handled automatically by the db_client fixture


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
