"""
Phase1 of V1-to-V2 upgrade test. Must run under pyseekdb==1.0.0b7 (V1).

Creates a v1-style collection (table c$v1$<name>), inserts data, and writes
state to upgrade_test_state.json for phase2. Uses SEEKDB_PATH and SEEKDB_DATABASE.
"""

import json
import os
import sys
from pathlib import Path

# Ensure phase1 runs with the V1-installed pyseekdb (no local path override)
import pyseekdb  # noqa: E402

SEEKDB_PATH = os.environ.get("SEEKDB_PATH")
SEEKDB_DATABASE = os.environ.get("SEEKDB_DATABASE", "test")

V1_COLLECTION_NAME = "upgrade_test_v1"
V1_IDS = ["v1_id1", "v1_id2"]
V1_EMBEDDINGS = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
V1_DOCUMENTS = ["v1_doc1", "v1_doc2"]


def main():
    if not SEEKDB_PATH:
        print("SEEKDB_PATH is not set. Run via run_upgrade_test.sh or set SEEKDB_PATH.", file=sys.stderr)
        sys.exit(1)

    client = pyseekdb.Client(path=SEEKDB_PATH, database=SEEKDB_DATABASE)

    # Create v1 collection with dimension=3 so add() embeddings (length 3) match.
    # 1.0.0b7 defaults to 384; we must pass dimension or configuration.
    coll = None
    if hasattr(pyseekdb, "HNSWConfiguration"):
        cfg = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")
        try:
            coll = client.create_collection(
                name=V1_COLLECTION_NAME,
                configuration=cfg,
                embedding_function=None,
            )
        except TypeError:
            try:
                coll = client.create_collection(
                    name=V1_COLLECTION_NAME,
                    configuration=cfg,
                )
            except TypeError:
                pass
    if coll is None:
        try:
            coll = client.create_collection(name=V1_COLLECTION_NAME, dimension=3)
        except TypeError:
            try:
                coll = client.create_collection(
                    name=V1_COLLECTION_NAME,
                    metadata={"dimension": 3},
                )
            except TypeError:
                raise RuntimeError(
                    "create_collection with dimension=3 failed; "
                    "1.0.0b7 may need a different signature"
                ) from None

    # Verify collection name matches
    assert coll.name == V1_COLLECTION_NAME, f"Collection name mismatch: expected {V1_COLLECTION_NAME}, got {coll.name}"
    print(f"\n✅ Collection '{V1_COLLECTION_NAME}' created successfully")
    print(f"   Collection name: {coll.name}")
    print(f"   Collection dimension: {coll.dimension}")

    # Insert test data
    print(f"\n✅ Inserting test data into collection '{V1_COLLECTION_NAME}'")
    coll.add(
        ids=V1_IDS,
        embeddings=V1_EMBEDDINGS,
        documents=V1_DOCUMENTS,
    )
    print(f"   Inserted {len(V1_IDS)} items: {V1_IDS}")

    # Verify data after insertion
    print(f"\n✅ Verifying inserted data")
    inserted_data = coll.get(ids=V1_IDS)
    assert "ids" in inserted_data, "get() result must contain 'ids'"
    assert "documents" in inserted_data, "get() result must contain 'documents'"
    assert len(inserted_data["ids"]) == len(V1_IDS), f"Expected {len(V1_IDS)} ids, got {len(inserted_data['ids'])}"
    assert set(inserted_data["ids"]) == set(V1_IDS), f"ID mismatch: expected {set(V1_IDS)}, got {set(inserted_data['ids'])}"
    assert len(inserted_data["documents"]) == len(V1_DOCUMENTS), f"Expected {len(V1_DOCUMENTS)} documents, got {len(inserted_data['documents'])}"
    for i, doc_id in enumerate(V1_IDS):
        idx = inserted_data["ids"].index(doc_id)
        assert inserted_data["documents"][idx] == V1_DOCUMENTS[i], f"Document mismatch for {doc_id}: expected {V1_DOCUMENTS[i]}, got {inserted_data['documents'][idx]}"
    print(f"   ✅ Verified {len(inserted_data['ids'])} items retrieved correctly")
    print(f"   IDs: {inserted_data['ids']}")
    print(f"   Documents: {inserted_data['documents']}")

    # Verify collection count
    item_count = coll.count()
    assert item_count == len(V1_IDS), f"Collection count mismatch: expected {len(V1_IDS)}, got {item_count}"
    print(f"\n✅ Collection count verified: {item_count} items")

    # Verify collection exists via has_collection
    assert client.has_collection(V1_COLLECTION_NAME), f"has_collection() should return True for '{V1_COLLECTION_NAME}'"
    print(f"   ✅ has_collection('{V1_COLLECTION_NAME}') returns True")

    state_dir = Path(SEEKDB_PATH).resolve().parent
    state_path = state_dir / "upgrade_test_state.json"
    state = {
        "v1_collection_name": V1_COLLECTION_NAME,
        "v1_ids": V1_IDS,
        "path": SEEKDB_PATH,
        "database": SEEKDB_DATABASE,
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ Phase1 completed successfully: v1 collection '{V1_COLLECTION_NAME}' created, {item_count} items inserted, state written to {state_path}")


if __name__ == "__main__":
    main()
