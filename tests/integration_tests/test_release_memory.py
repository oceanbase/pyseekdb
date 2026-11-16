"""
Integration tests for the release_memory API (embedded mode only, issue #288).

Runs against a real embedded seekdb engine. release_memory() is a
fire-and-forget request: the call submits the engine memory release and
returns immediately, and the engine completes the release in the background.
These tests validate that the request takes effect (observed through the
engine's own views), that data stays queryable and survives reopening, and
that concurrent writes, repeated calls, and open transactions have defined
behavior.

The class name carries "Embedded" so CI's ``-k embedded`` selection runs it.
"""

import threading
import time

import pytest

import pyseekdb

_observability_supported: bool | None = None


def _skip_if_engine_lacks_release_views(client) -> None:
    """Skip when the installed binding hides the views these tests verify with."""
    global _observability_supported
    if _observability_supported is None:
        try:
            client._server._execute("SELECT TABLET_ID, IS_ACTIVE FROM oceanbase.GV$OB_MEMSTORE_INFO")
            _observability_supported = True
        except Exception:
            _observability_supported = False
    if not _observability_supported:
        pytest.skip("the embedded engine/binding does not expose the memory-release verification views")


def _add_batch(collection, start: int, count: int, dimension: int = 4) -> list[str]:
    ids = [f"row-{start + i}" for i in range(count)]
    embeddings = [[round((start + i) * 0.01 + 0.1, 4)] * dimension for i in range(count)]
    documents = [f"document {start + i} " + "x" * 200 for i in range(count)]
    collection.add(ids=ids, embeddings=embeddings, documents=documents)
    return ids


def _frozen_memtables(client) -> int:
    """Count frozen (not yet released) in-memory tables via the engine view."""
    rows = client._server._execute("SELECT TABLET_ID, IS_ACTIVE FROM oceanbase.GV$OB_MEMSTORE_INFO")
    frozen = 0
    for row in rows:
        is_active = str(row.get("IS_ACTIVE") if isinstance(row, dict) else row[1]).upper()
        if is_active != "YES":
            frozen += 1
    return frozen


def _wait_background_release(client, timeout: float = 60.0) -> bool:
    """Wait until the engine's background release finishes (no frozen tables)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _frozen_memtables(client) == 0:
            return True
        time.sleep(0.5)
    return False


def _make_collection(client, name: str):
    return client.get_or_create_collection(
        name=name,
        configuration=pyseekdb.HNSWConfiguration(dimension=4),
        embedding_function=None,
    )


class TestEmbeddedReleaseMemory:
    def test_release_request_completes_in_background_and_keeps_data(self, embedded_client):
        """The request returns immediately; the engine finishes the release in the background."""
        _skip_if_engine_lacks_release_views(embedded_client)
        collection_name = f"test_release_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 500)

            started = time.monotonic()
            result = embedded_client.release_memory()
            elapsed = time.monotonic() - started

            # Fire-and-forget: no result and no waiting for the release itself.
            assert result is None
            assert elapsed < 5.0

            # The engine finishes the release in the background on its own.
            assert _wait_background_release(embedded_client, timeout=60)

            # Data stays queryable with the database open.
            assert collection.count() == 500
        finally:
            embedded_client.delete_collection(collection_name)

    def test_release_after_batch_data_survives_reopen(self, embedded_client):
        """Data written before the release survives closing and reopening the database."""
        _skip_if_engine_lacks_release_views(embedded_client)
        collection_name = f"test_release_reopen_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        ids = _add_batch(collection, 0, 300)

        embedded_client.release_memory()
        assert _wait_background_release(embedded_client, timeout=60)
        assert collection.count() == 300

        # Data survives closing and reopening the database (fresh client).
        path = embedded_client._server.path
        database = embedded_client._server.database
        embedded_client.close()

        reopened = pyseekdb.Client(path=path, database=database)
        try:
            reopened_collection = reopened.get_collection(collection_name)
            assert reopened_collection.count() == 300
            got = reopened_collection.get(ids=ids[:5])
            assert set(got["ids"]) == set(ids[:5])
        finally:
            reopened.delete_collection(collection_name)
            reopened.close()

    def test_repeated_release_calls_have_defined_behavior(self, embedded_client):
        _skip_if_engine_lacks_release_views(embedded_client)
        collection_name = f"test_release_repeat_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 50)

            assert embedded_client.release_memory() is None
            assert _wait_background_release(embedded_client, timeout=60)
            # An immediate second release on an idle instance is a defined no-op.
            assert embedded_client.release_memory() is None
            assert collection.count() == 50

            # A third release after more writes still works.
            _add_batch(collection, 100, 50)
            embedded_client.release_memory()
            assert _wait_background_release(embedded_client, timeout=60)
            assert collection.count() == 100
        finally:
            embedded_client.delete_collection(collection_name)

    def test_concurrent_writes_continue_during_release(self, embedded_client):
        """Writes committed after the request continue into new tables and stay readable."""
        _skip_if_engine_lacks_release_views(embedded_client)
        collection_name = f"test_release_concurrent_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        _add_batch(collection, 0, 200)

        # The concurrent writer uses its own client (own session) so it never
        # shares a connection with the release's dedicated session.
        writer_client = pyseekdb.Client(path=embedded_client._server.path, database=embedded_client._server.database)
        writer_collection = writer_client.get_collection(collection_name)

        def keep_writing() -> None:
            for start in (1000, 2000, 3000):
                _add_batch(writer_collection, start, 100)

        writer = threading.Thread(target=keep_writing)
        writer.start()
        try:
            assert embedded_client.release_memory() is None
        finally:
            writer.join()
            writer_client.close()

        # Every write, before and after the request, is present.
        assert collection.count() == 500
        embedded_client.delete_collection(collection_name)

    def test_release_does_not_commit_open_transaction(self, embedded_client):
        """An open application transaction must not be implicitly committed."""
        _skip_if_engine_lacks_release_views(embedded_client)
        probe_table = f"release_tx_probe_{int(time.time() * 1000)}"
        client_server = embedded_client._server
        client_server._execute(f"CREATE TABLE `{probe_table}` (id INT PRIMARY KEY, msg VARCHAR(256))")
        try:
            client_server._execute(f"INSERT INTO `{probe_table}` VALUES (1, 'committed')")

            raw = client_server.get_raw_connection()
            raw_cursor = raw.cursor()
            try:
                raw_cursor.execute("BEGIN")
                raw_cursor.execute(f"INSERT INTO `{probe_table}` VALUES (999999, 'uncommitted')")

                # Release while the transaction is open; the engine command
                # runs on a dedicated session and must not commit it.
                assert embedded_client.release_memory() is None

                raw_cursor.execute("ROLLBACK")
            finally:
                raw_cursor.close()

            rows = client_server._execute(f"SELECT COUNT(*) AS c FROM `{probe_table}`")
            count = rows[0].get("c") if isinstance(rows[0], dict) else rows[0][0]
            # Only the committed row: the release never committed the transaction.
            assert count == 1
        finally:
            client_server._execute(f"DROP TABLE `{probe_table}`")

    def test_release_memory_is_noop_in_remote_mode(self):
        """Non-embedded modes log a warning and do nothing instead of raising."""
        remote = pyseekdb.Client(host="127.0.0.1", port=2881, user="root", password="")
        try:
            # No exception and no connection attempt: a plain no-op.
            assert remote.release_memory() is None
        finally:
            remote.close()
