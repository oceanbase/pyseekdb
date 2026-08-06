# V1-to-V2 Upgrade Test

Tests that upgrading from pyseekdb 1.0.0b7 (V1) to 1.0.0b8 (V2) keeps v1 collections working and v2 behavior correct.

This test lives under `tests/v1_upgrade_v2/` (not under `integration_tests/`) so that running it via pytest does not load `integration_tests/conftest.py`, avoiding pyseekdb/httpx/idna imports that can fail in some environments (e.g. `ModuleNotFoundError: No module named 'idna.core'`).

## Flow

1. **Phase1** (runs under `pyseekdb==1.0.0b7`): Creates a v1-style collection (table `c$v1$<name>`), inserts data, writes `upgrade_test_state.json`.
2. **Upgrade**: `pip install pyseekdb==1.0.0b8`.
3. **Phase2** (runs under `pyseekdb==1.0.0b8`): Verifies v1 collection CRUD, v2 collection CRUD, `sdk_collections` metadata for v2 only, and isolation between v1 and v2.

## How to run

**Unified pytest (with full suite):**

```bash
python3.11 -m pytest tests/ -v -s
```

**Only this upgrade test via pytest:**

```bash
python3.11 -m pytest tests/v1_upgrade_v2/ -v -s
```

**Standalone script** (from repo root or this directory):

```bash
./tests/v1_upgrade_v2/run_upgrade_test.sh
```

Or with explicit env:

```bash
export SEEKDB_PATH=/path/to/seekdb.db   # default: tests/seekdb.db
export SEEKDB_DATABASE=test
./tests/v1_upgrade_v2/run_upgrade_test.sh
```

By default the script uses `tests/seekdb.db` as the database (same as integration_tests) and creates/uses the virtual env at `/home/chenminsi.cms/.venv_upgrade`. Override with `VENV_UPGRADE_DIR` if needed.

Requires `python3.11` and network (for pip).

## Phase1 and pyseekdb==1.0.0b7

Phase1 is written for `pyseekdb==1.0.0b7`. If that version uses a different `create_collection` signature (e.g. requires `dimension` or `metadata`), edit `phase1_create_v1_data.py` to match that API.

## What is asserted

- V1 collection: `get_collection(name)` returns a collection with `id is None`; add/get/update work.
- V2 collection: `create_collection(name)` returns a collection with `id` set; add/get work.
- `sdk_collections`: one row for the v2 collection with correct `COLLECTION_ID` and table name `c$v2$<collection_id>`; no row for the v1 collection.
- `list_collections` includes both v1 and v2.
- Deleting the v1 collection does not affect the v2 collection; deleting the v2 collection cleans up correctly.
