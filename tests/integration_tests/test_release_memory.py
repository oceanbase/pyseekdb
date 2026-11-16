"""
Integration tests for the release_memory API (embedded mode only, issue #288).

Runs against a real embedded seekdb engine. Validates that engine memory
held by a batch of committed writes is released while the database stays
open, that data remains queryable and survives reopening, that the release
is measurable on an idle instance, and that concurrent writes, timeouts,
repeated calls, and open transactions have defined behavior.

The class name carries "Embedded" so CI's ``-k embedded`` selection runs it.
"""

import threading
import time

import pytest

import pyseekdb
from pyseekdb.client.memory import MemoryReleaseResult, MemoryReleaseTimeoutError

_observability_supported: bool | None = None


def _skip_if_engine_lacks_release_support(client) -> None:
    """Skip when the installed binding hides the memory-release observability views."""
    global _observability_supported
    if _observability_supported is None:
        try:
            client._server._execute("SELECT FREEZE_CNT FROM oceanbase.GV$OB_MEMSTORE")
            client._server._execute("SELECT TABLET_ID FROM oceanbase.GV$OB_MEMSTORE_INFO")
            _observability_supported = True
        except Exception:
            _observability_supported = False
    if not _observability_supported:
        pytest.skip("the embedded engine/binding does not expose the memory-release observability views")


def _add_batch(collection, start: int, count: int, dimension: int = 4) -> list[str]:
    ids = [f"row-{start + i}" for i in range(count)]
    embeddings = [[round((start + i) * 0.01 + 0.1, 4)] * dimension for i in range(count)]
    documents = [f"document {start + i} " + "x" * 200 for i in range(count)]
    collection.add(ids=ids, embeddings=embeddings, documents=documents)
    return ids


def _engine_memory_used(client) -> int | None:
    rows = client._server._execute("SELECT MEMSTORE_USED FROM oceanbase.GV$OB_MEMSTORE")
    if not rows:
        return None
    row = rows[0]
    value = row.get("MEMSTORE_USED") if isinstance(row, dict) else row[0]
    return int(value)


def _make_collection(client, name: str):
    return client.get_or_create_collection(
        name=name,
        configuration=pyseekdb.HNSWConfiguration(dimension=4),
        embedding_function=None,
    )


class TestEmbeddedReleaseMemory:
    def test_release_after_batch_keeps_data_queryable_and_survives_reopen(self, embedded_client):
        """A release keeps the database open; data stays queryable and survives reopen."""
        _skip_if_engine_lacks_release_support(embedded_client)
        collection_name = f"test_release_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        ids = _add_batch(collection, 0, 500)

        result = embedded_client.release_memory(timeout=60)

        assert isinstance(result, MemoryReleaseResult)
        assert result.memstore_used_before_bytes > 0
        assert result.memstore_used_after_bytes is not None
        assert result.reclaimed_bytes is not None

        # Data stays queryable with the database open.
        got = collection.get(ids=ids[:5])
        assert set(got["ids"]) == set(ids[:5])
        assert collection.count() == 500

        # Data survives closing and reopening the database (fresh client).
        path = embedded_client._server.path
        database = embedded_client._server.database
        embedded_client.close()

        reopened = pyseekdb.Client(path=path, database=database)
        try:
            reopened_collection = reopened.get_collection(collection_name)
            assert reopened_collection.count() == 500
            got = reopened_collection.get(ids=ids[:5])
            assert set(got["ids"]) == set(ids[:5])
        finally:
            reopened.delete_collection(collection_name)
            reopened.close()

    def test_release_measurably_reduces_engine_memory_on_idle_instance(self, embedded_client):
        """The release measurably reduces engine memory accounting on an idle instance."""
        _skip_if_engine_lacks_release_support(embedded_client)
        collection_name = f"test_release_reclaim_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 400)
            _add_batch(collection, 1000, 400)

            used_before = _engine_memory_used(embedded_client)
            assert used_before is not None and used_before > 0

            result = embedded_client.release_memory(timeout=60)

            # The instance-wide release frees every boundary table's memory,
            # so engine accounting drops well below the batch peak.
            assert result.memstore_used_before_bytes == used_before
            assert result.memstore_used_after_bytes < used_before
            assert result.reclaimed_bytes > 0
            assert collection.count() == 800
        finally:
            embedded_client.delete_collection(collection_name)

    def test_release_timeout_reports_still_held_and_keeps_writes(self, embedded_client):
        """A tiny timeout reports what is still held; committed writes are not undone."""
        _skip_if_engine_lacks_release_support(embedded_client)
        collection_name = f"test_release_timeout_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            ids = _add_batch(collection, 0, 100)

            with pytest.raises(MemoryReleaseTimeoutError) as excinfo:
                embedded_client.release_memory(timeout=0.1)

            assert "timed out" in str(excinfo.value)
            assert "Committed writes are unaffected" in str(excinfo.value)
            # Committed writes survive the timeout untouched.
            assert collection.count() == 100
            got = collection.get(ids=ids[:3])
            assert set(got["ids"]) == set(ids[:3])

            # Background release continues; a later call completes.
            followup = embedded_client.release_memory(timeout=60)
            assert followup.reclaimed_bytes is not None
        finally:
            embedded_client.delete_collection(collection_name)

    def test_repeated_release_calls_have_defined_behavior(self, embedded_client):
        _skip_if_engine_lacks_release_support(embedded_client)
        collection_name = f"test_release_repeat_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 50)

            first = embedded_client.release_memory(timeout=60)
            assert first.reclaimed_bytes is not None
            # An immediate second release on an idle instance also completes.
            second = embedded_client.release_memory(timeout=60)
            assert second.memstore_used_after_bytes is not None
            assert collection.count() == 50

            # A third release after more writes still succeeds.
            _add_batch(collection, 100, 50)
            third = embedded_client.release_memory(timeout=60)
            assert third.memstore_used_after_bytes is not None
            assert collection.count() == 100
        finally:
            embedded_client.delete_collection(collection_name)

    def test_concurrent_writes_continue_during_release(self, embedded_client):
        """Writes after the boundary continue into new tables and stay readable."""
        _skip_if_engine_lacks_release_support(embedded_client)
        collection_name = f"test_release_concurrent_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        _add_batch(collection, 0, 200)

        # The concurrent writer uses its own client (own session) so it never
        # shares a connection with the release's polls.
        writer_client = pyseekdb.Client(path=embedded_client._server.path, database=embedded_client._server.database)
        writer_collection = writer_client.get_collection(collection_name)

        def keep_writing() -> None:
            for start in (1000, 2000, 3000):
                _add_batch(writer_collection, start, 100)

        writer = threading.Thread(target=keep_writing)
        writer.start()
        try:
            result = embedded_client.release_memory(timeout=120)
            assert result.reclaimed_bytes is not None
        finally:
            writer.join()
            writer_client.close()

        # Every write, before and after the boundary, is present.
        assert collection.count() == 500
        embedded_client.delete_collection(collection_name)

    def test_release_does_not_commit_open_transaction(self, embedded_client):
        """An open application transaction must not be implicitly committed."""
        _skip_if_engine_lacks_release_support(embedded_client)
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

                # Release while the transaction is open; the freeze command
                # runs on a dedicated session and must not commit it.
                result = embedded_client.release_memory(timeout=60)
                assert result.reclaimed_bytes is not None

                raw_cursor.execute("ROLLBACK")
            finally:
                raw_cursor.close()

            rows = client_server._execute(f"SELECT COUNT(*) AS c FROM `{probe_table}`")
            count = rows[0].get("c") if isinstance(rows[0], dict) else rows[0][0]
            # Only the committed row: the release never committed the transaction.
            assert count == 1
        finally:
            client_server._execute(f"DROP TABLE `{probe_table}`")

    def test_release_rejects_unsupported_timeout_values(self, embedded_client):
        with pytest.raises(ValueError, match="timeout"):
            embedded_client.release_memory(timeout=0)
        with pytest.raises(ValueError, match="timeout"):
            embedded_client.release_memory(timeout=-5)
        with pytest.raises(ValueError, match="timeout"):
            embedded_client.release_memory(timeout="30")

    def test_release_memory_unsupported_in_remote_mode(self):
        """Remote server mode returns a clear error instead of a partial release."""
        remote = pyseekdb.Client(host="127.0.0.1", port=2881, user="root", password="")
        try:
            with pytest.raises(ValueError, match="embedded client"):
                remote.release_memory()
        finally:
            remote.close()
