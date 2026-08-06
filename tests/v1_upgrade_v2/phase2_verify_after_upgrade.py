"""
Phase2 of V1-to-V2 upgrade test. Run after upgrading to latest pyseekdb.

Reads upgrade_test_state.json, connects to the same DB, and verifies:
- v1 collection (c$v1$<name>) still supports full CRUD; v1 has no row in sdk_collections (inferred from get_collection behavior).
- v2 collection works; sdk_collections behavior is verified via client.get_collection(name) only (same as test_collection_get, no raw SQL).
- v1 and v2 do not affect each other.
"""

import json
import os
import sys
from pathlib import Path

import pyseekdb
from pyseekdb.client.meta_info import CollectionNames

SEEKDB_PATH = os.environ.get("SEEKDB_PATH")
SEEKDB_DATABASE = os.environ.get("SEEKDB_DATABASE", "test")
STATE_FILENAME = "upgrade_test_state.json"
V2_COLLECTION_NAME = "upgrade_test_v2"


def load_state():
    if not SEEKDB_PATH:
        print("SEEKDB_PATH is not set. Run via run_upgrade_test.sh or set SEEKDB_PATH.", file=sys.stderr)
        sys.exit(1)
    state_path = Path(SEEKDB_PATH).resolve().parent / STATE_FILENAME
    if not state_path.exists():
        print(f"State file not found: {state_path}", file=sys.stderr)
        sys.exit(1)
    with open(state_path) as f:
        return json.load(f)


def main():
    state = load_state()
    path = state.get("path") or SEEKDB_PATH
    database = state.get("database") or SEEKDB_DATABASE
    v1_name = state["v1_collection_name"]
    v1_ids = state["v1_ids"]

    client = pyseekdb.Client(path=path, database=database)
    hnsw_config = pyseekdb.HNSWConfiguration(dimension=3, distance="cosine")

    # ---- v1 collection: get and verify data from phase1 ----
    # Use embedding_function=None so update/add do not use default 384-dim embedding
    print(f"\n✅ Retrieving v1 collection '{v1_name}' created in Phase1")
    coll_v1 = client.get_collection(v1_name, embedding_function=None)
    assert coll_v1 is not None, f"v1 collection '{v1_name}' should exist"
    assert coll_v1.name == v1_name, f"Collection name mismatch: expected {v1_name}, got {coll_v1.name}"
    assert coll_v1.id is None, f"v1 collection must have id is None, got {coll_v1.id}"
    print(f"   Collection name: {coll_v1.name}")
    print(f"   Collection id: {coll_v1.id} (None as expected for v1)")
    print(f"   Collection dimension: {coll_v1.dimension}")

    # Verify phase1 data is still accessible
    print(f"\n✅ Verifying v1 collection data from Phase1")
    res = coll_v1.get(ids=v1_ids)
    assert "ids" in res, "get() result must contain 'ids'"
    assert "documents" in res, "get() result must contain 'documents'"
    assert len(res["ids"]) == len(v1_ids), f"Expected {len(v1_ids)} ids, got {len(res['ids'])}"
    assert set(res["ids"]) == set(v1_ids), f"ID mismatch: expected {set(v1_ids)}, got {set(res['ids'])}"
    print(f"   ✅ Retrieved {len(res['ids'])} items from Phase1")
    print(f"   IDs: {res['ids']}")
    print(f"   Documents: {res['documents']}")

    # ---- v1: add one more row ----
    print(f"\n✅ Adding new item to v1 collection '{v1_name}'")
    new_id = "v1_id3"
    new_embedding = [3.0, 4.0, 5.0]
    new_document = "v1_doc3"
    coll_v1.add(
        ids=[new_id],
        embeddings=[new_embedding],
        documents=[new_document],
    )
    print(f"   Inserted item: id={new_id}, document={new_document}")

    # Verify the new item
    print(f"   Verifying inserted item")
    res3 = coll_v1.get(ids=[new_id])
    assert "ids" in res3, "get() result must contain 'ids'"
    assert "documents" in res3, "get() result must contain 'documents'"
    assert len(res3["ids"]) == 1, f"Expected 1 id, got {len(res3['ids'])}"
    assert res3["ids"][0] == new_id, f"ID mismatch: expected {new_id}, got {res3['ids'][0]}"
    assert res3["documents"][0] == new_document, f"Document mismatch: expected {new_document}, got {res3['documents'][0]}"
    print(f"   ✅ Successfully added and verified new item: {res3['ids'][0]} -> {res3['documents'][0]}")

    # Verify collection count increased
    item_count_after_add = coll_v1.count()
    assert item_count_after_add == len(v1_ids) + 1, f"Collection count mismatch: expected {len(v1_ids) + 1}, got {item_count_after_add}"
    print(f"   ✅ Collection count after add: {item_count_after_add} items")

    # ---- v1: update ----
    # v1 table has dimension 3; pass embeddings explicitly so V2 does not use default 384-dim embedding
    print(f"\n✅ Updating item in v1 collection '{v1_name}'")
    update_id = "v1_id1"
    updated_document = "v1_doc1_updated"
    update_embedding = [1.0, 2.0, 3.0]
    coll_v1.update(
        ids=update_id,
        documents=updated_document,
        embeddings=update_embedding,
    )
    print(f"   Updated item: id={update_id}, document={updated_document}")

    # Verify the update
    print(f"   Verifying updated item")
    res_up = coll_v1.get(ids=update_id)
    assert "ids" in res_up, "get() result must contain 'ids'"
    assert "documents" in res_up, "get() result must contain 'documents'"
    assert len(res_up["ids"]) == 1, f"Expected 1 id, got {len(res_up['ids'])}"
    assert res_up["ids"][0] == update_id, f"ID mismatch: expected {update_id}, got {res_up['ids'][0]}"
    assert res_up["documents"][0] == updated_document, f"Document mismatch: expected {updated_document}, got {res_up['documents'][0]}"
    print(f"   ✅ Successfully updated and verified: {res_up['ids'][0]} -> {res_up['documents'][0]}")

    # ---- v2 collection: create and use ----
    # Verify v2 collection does not exist before creation (created after upgrade)
    print(f"\n✅ Verifying v2 collection '{V2_COLLECTION_NAME}' does not exist before creation")
    assert not client.has_collection(V2_COLLECTION_NAME), f"v2 collection '{V2_COLLECTION_NAME}' should not exist before creation (it's created after upgrade)"
    print(f"   ✅ has_collection('{V2_COLLECTION_NAME}') returns False (as expected)")

    # Create v2 collection (this should only work after upgrade to V2; sdk_collections is used internally by create/get_collection)
    print(f"\n✅ Creating v2 collection '{V2_COLLECTION_NAME}' (post-upgrade)")
    coll_v2 = client.create_collection(
        name=V2_COLLECTION_NAME,
        configuration=hnsw_config,
        embedding_function=None,
    )
    assert coll_v2 is not None, f"v2 collection '{V2_COLLECTION_NAME}' should be created"
    assert coll_v2.name == V2_COLLECTION_NAME, f"Collection name mismatch: expected {V2_COLLECTION_NAME}, got {coll_v2.name}"
    print(f"   Collection name: {coll_v2.name}")
    print(f"   Collection id from object: {coll_v2.id}")
    print(f"   Collection dimension: {coll_v2.dimension}")

    v2_collection_id = coll_v2.id
    if v2_collection_id is not None:
        expected_v2_table = CollectionNames.table_name_v2(v2_collection_id)
        assert expected_v2_table.startswith("c$v2$"), f"v2 table name must start with c$v2$, got {expected_v2_table}"
        print(f"   ✅ v2 table name format: {expected_v2_table}")

    # Verify sdk_collections behavior via get_collection(name) only (same as test_collection_get, no raw SQL)
    print(f"\n✅ Verifying sdk_collections via get_collection('{V2_COLLECTION_NAME}') (same as test_collection_get)")
    retrieved_v2 = client.get_collection(V2_COLLECTION_NAME, embedding_function=None)
    assert retrieved_v2 is not None, f"get_collection('{V2_COLLECTION_NAME}') should return a collection"
    assert retrieved_v2.name == V2_COLLECTION_NAME, f"Retrieved collection name mismatch: expected {V2_COLLECTION_NAME}, got {retrieved_v2.name}"
    assert retrieved_v2.dimension == coll_v2.dimension, f"Retrieved dimension mismatch: expected {coll_v2.dimension}, got {retrieved_v2.dimension}"
    assert client.has_collection(V2_COLLECTION_NAME), f"has_collection('{V2_COLLECTION_NAME}') should be True"
    print(f"   ✅ get_collection('{V2_COLLECTION_NAME}') returned: name={retrieved_v2.name}, id={retrieved_v2.id}, dimension={retrieved_v2.dimension}")
    print(f"   (v2 lookup by name uses sdk_collections; get_collection success implies sdk_collections has the row)")

    # Insert data into v2 collection
    print(f"\n✅ Inserting test data into v2 collection '{V2_COLLECTION_NAME}'")
    v2_test_id = "v2_id1"
    v2_test_embedding = [1.0, 2.0, 3.0]
    v2_test_document = "v2_doc1"
    coll_v2.add(
        ids=[v2_test_id],
        embeddings=[v2_test_embedding],
        documents=[v2_test_document],
    )
    print(f"   Inserted item: id={v2_test_id}, document={v2_test_document}")

    # Verify inserted data
    print(f"   Verifying inserted data")
    res_v2 = coll_v2.get(ids=[v2_test_id])
    assert "ids" in res_v2, "get() result must contain 'ids'"
    assert "documents" in res_v2, "get() result must contain 'documents'"
    assert len(res_v2["ids"]) == 1, f"Expected 1 id, got {len(res_v2['ids'])}"
    assert res_v2["ids"][0] == v2_test_id, f"ID mismatch: expected {v2_test_id}, got {res_v2['ids'][0]}"
    assert res_v2["documents"][0] == v2_test_document, f"Document mismatch: expected {v2_test_document}, got {res_v2['documents'][0]}"
    print(f"   ✅ Successfully inserted and verified: {res_v2['ids'][0]} -> {res_v2['documents'][0]}")

    # Verify v2 collection count
    v2_count = coll_v2.count()
    assert v2_count == 1, f"v2 collection count mismatch: expected 1, got {v2_count}"
    print(f"   ✅ v2 collection count: {v2_count} items")

    # ---- Create multiple v2 collections; verify via get_collection (sdk_collections behavior) ----
    print(f"\n✅ Creating additional v2 collections; verifying via get_collection (same as test_collection_get)")
    additional_v2_names = [f"{V2_COLLECTION_NAME}_extra_{i}" for i in range(1, 4)]
    additional_v2_collections = []
    for extra_name in additional_v2_names:
        print(f"   Creating v2 collection '{extra_name}'")
        extra_coll = client.create_collection(
            name=extra_name,
            configuration=hnsw_config,
            embedding_function=None,
        )
        assert extra_coll is not None, f"v2 collection '{extra_name}' should be created"
        assert extra_coll.name == extra_name, f"Collection name mismatch: expected {extra_name}, got {extra_coll.name}"
        additional_v2_collections.append(extra_coll)
        print(f"      Created: name={extra_coll.name}, id={extra_coll.id}")
    
    # Verify each v2 collection via get_collection(name) (no raw SQL)
    print(f"   Verifying each v2 collection via get_collection(name)")
    all_v2_names = [V2_COLLECTION_NAME] + additional_v2_names
    for v2_name in all_v2_names:
        assert client.has_collection(v2_name), f"has_collection('{v2_name}') should be True"
        c = client.get_collection(v2_name, embedding_function=None)
        assert c.name == v2_name, f"get_collection('{v2_name}') name mismatch"
        print(f"      ✅ '{v2_name}' -> get_collection OK (name={c.name}, id={c.id})")
    
    # Clean up additional collections
    print(f"   Cleaning up additional v2 collections")
    for extra_coll in additional_v2_collections:
        try:
            client.delete_collection(extra_coll.name)
            print(f"      ✅ Deleted '{extra_coll.name}'")
        except Exception as e:
            print(f"      ⚠️  Failed to delete '{extra_coll.name}': {e}")

    # ---- Verify v2 and v1 via get_collection (v2 uses sdk_collections, v1 does not) ----
    print(f"\n✅ Verifying v2 and v1 via get_collection (sdk_collections behavior reflected in get_collection)")
    print(f"   v2 '{V2_COLLECTION_NAME}': get_collection(name) should return collection (v2 looks up by name via sdk_collections)")
    coll_v2_recheck = client.get_collection(V2_COLLECTION_NAME, embedding_function=None)
    assert coll_v2_recheck.name == V2_COLLECTION_NAME and coll_v2_recheck.dimension == 3
    print(f"   ✅ get_collection('{V2_COLLECTION_NAME}') OK: name={coll_v2_recheck.name}, dimension={coll_v2_recheck.dimension}")
    print(f"   v1 '{v1_name}': get_collection(name) should return collection with id=None (v1 has no row in sdk_collections)")
    coll_v1_recheck = client.get_collection(v1_name, embedding_function=None)
    assert coll_v1_recheck.name == v1_name and coll_v1_recheck.id is None
    print(f"   ✅ get_collection('{v1_name}') OK: name={coll_v1_recheck.name}, id={coll_v1_recheck.id} (v1)")

    # ---- list_collections includes both ----
    print(f"\n✅ Verifying list_collections includes both v1 and v2 collections")
    collections_list = client.list_collections()
    assert isinstance(collections_list, list), "list_collections() should return a list"
    names = [c.name for c in collections_list]
    assert v1_name in names, f"list_collections must include v1 collection '{v1_name}', got {names}"
    assert V2_COLLECTION_NAME in names, f"list_collections must include v2 collection '{V2_COLLECTION_NAME}', got {names}"
    print(f"   ✅ list_collections returned {len(collections_list)} collection(s)")
    print(f"   Collection names: {names}")
    print(f"   ✅ Both '{v1_name}' (v1) and '{V2_COLLECTION_NAME}' (v2) are present")

    # ---- delete v1: v1 gone, v2 unchanged ----
    print(f"\n✅ Deleting v1 collection '{v1_name}'")
    client.delete_collection(v1_name)
    assert not client.has_collection(v1_name), f"v1 collection '{v1_name}' should be gone after delete"
    print(f"   ✅ v1 collection '{v1_name}' successfully deleted")
    
    # Verify v2 collection and data are unchanged
    print(f"   Verifying v2 collection '{V2_COLLECTION_NAME}' is unchanged after v1 delete")
    assert client.has_collection(V2_COLLECTION_NAME), f"v2 collection '{V2_COLLECTION_NAME}' should still exist"
    res_v2_after = coll_v2.get(ids=[v2_test_id])
    assert "ids" in res_v2_after, "get() result must contain 'ids'"
    assert len(res_v2_after["ids"]) == 1, f"v2 data must be unchanged after v1 delete, expected 1 id, got {len(res_v2_after['ids'])}"
    assert res_v2_after["ids"][0] == v2_test_id, f"v2 ID mismatch: expected {v2_test_id}, got {res_v2_after['ids'][0]}"
    assert res_v2_after["documents"][0] == v2_test_document, f"v2 document mismatch: expected {v2_test_document}, got {res_v2_after['documents'][0]}"
    print(f"   ✅ v2 collection data verified unchanged: {res_v2_after['ids'][0]} -> {res_v2_after['documents'][0]}")

    # ---- delete v2: verify post-delete sdk_collections behavior via has_collection / get_collection ----
    print(f"\n✅ Verifying v2 collection '{V2_COLLECTION_NAME}' exists before deletion (via get_collection)")
    assert client.has_collection(V2_COLLECTION_NAME), f"has_collection('{V2_COLLECTION_NAME}') should be True before delete"
    client.get_collection(V2_COLLECTION_NAME, embedding_function=None)  # success implies sdk_collections has the row

    print(f"\n✅ Deleting v2 collection '{V2_COLLECTION_NAME}'")
    client.delete_collection(V2_COLLECTION_NAME)
    assert not client.has_collection(V2_COLLECTION_NAME), f"v2 collection '{V2_COLLECTION_NAME}' should be gone after delete"
    print(f"   ✅ v2 collection '{V2_COLLECTION_NAME}' successfully deleted")

    # After delete, get_collection(name) should raise (sdk_collections row for this name is gone)
    print(f"   Verifying get_collection('{V2_COLLECTION_NAME}') raises after deletion (sdk_collections row removed)")
    print(f"   Verifying get_collection('{V2_COLLECTION_NAME}') raises ValueError after deletion")
    try:
        deleted_coll = client.get_collection(V2_COLLECTION_NAME)
        assert False, f"get_collection('{V2_COLLECTION_NAME}') should raise ValueError after deletion, but returned {deleted_coll}"
    except ValueError as e:
        assert "not found" in str(e).lower() or "does not exist" in str(e).lower(), f"Expected 'not found' or 'does not exist' in error, got: {e}"
        print(f"   ✅ get_collection('{V2_COLLECTION_NAME}') correctly raises ValueError: {e}")

    print(f"\n✅ Phase2 completed successfully: all checks passed")
    print(f"   - v1 collection '{v1_name}' CRUD operations verified")
    print(f"   - v2 collection '{V2_COLLECTION_NAME}' created and verified (post-upgrade)")
    print(f"   - sdk_collections behavior verified via get_collection(name)/has_collection only (same as test_collection_get, no raw SQL)")
    print(f"   - multiple v2 collections verified via get_collection(name)")
    print(f"   - list_collections includes both v1 and v2")
    print(f"   - v1 deleted; after v2 delete: has_collection(False), get_collection raises")


if __name__ == "__main__":
    main()
