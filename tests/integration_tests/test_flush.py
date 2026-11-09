"""
Integration tests for the explicit flush API (embedded mode only, issue #288).

Runs against a real embedded seekdb engine. Validates that a batch of
committed writes can be flushed while the database stays open, that data
remains queryable and survives reopening, that MemStore memory is measurably
reclaimed on an idle instance, and that concurrent writes, blocking timeouts,
repeated calls, and open transactions have defined behavior.

The class name carries "Embedded" so CI's ``-k embedded`` selection runs it.
"""

import threading
import time

import pytest

import pyseekdb
from pyseekdb.client.flush import FlushHandle, FlushResult, FlushStage, FlushTimeoutError

_observability_supported: bool | None = None


def _skip_if_engine_lacks_flush_support(client) -> None:
    """Skip when the installed binding hides the flush observability views."""
    global _observability_supported
    if _observability_supported is None:
        try:
            client._server._execute("SELECT FREEZE_CNT FROM oceanbase.GV$OB_MEMSTORE")
            client._server._execute("SELECT TABLET_ID FROM oceanbase.GV$OB_MEMSTORE_INFO")
            _observability_supported = True
        except Exception:
            _observability_supported = False
    if not _observability_supported:
        pytest.skip("the embedded engine/binding does not expose the flush observability views")


def _add_batch(collection, start: int, count: int, dimension: int = 4) -> list[str]:
    ids = [f"row-{start + i}" for i in range(count)]
    embeddings = [[round((start + i) * 0.01 + 0.1, 4)] * dimension for i in range(count)]
    documents = [f"document {start + i} " + "x" * 200 for i in range(count)]
    collection.add(ids=ids, embeddings=embeddings, documents=documents)
    return ids


def _memstore_used(client) -> int | None:
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


class TestEmbeddedFlush:
    def test_flush_after_batch_keeps_data_queryable_and_survives_reopen(self, embedded_client):
        """A batch flush keeps the database open; data stays queryable and survives reopen."""
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        ids = _add_batch(collection, 0, 500)

        result = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)

        assert isinstance(result, FlushResult)
        assert result.completed
        assert result.stage == FlushStage.RECLAIMED
        assert result.frozen_memtable_count > 0
        assert result.released_memtable_count == result.frozen_memtable_count
        assert result.memstore_used_before_bytes > 0
        assert result.memstore_used_after_bytes is not None

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

    def test_flush_reclaims_memstore_memory_on_idle_instance(self, embedded_client):
        """Reclaiming flush measurably reduces MemStore usage on an idle instance."""
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_reclaim_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 400)
            _add_batch(collection, 1000, 400)

            used_before = _memstore_used(embedded_client)
            assert used_before is not None and used_before > 0

            result = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)

            assert result.completed
            assert result.memstore_used_before_bytes == used_before
            # The instance-level freeze releases every boundary MemTable's
            # arena chunks, so MemStore usage drops well below the batch peak.
            assert result.memstore_used_after_bytes < used_before
            assert result.memstore_reclaimed_bytes > 0
            assert collection.count() == 800
        finally:
            embedded_client.delete_collection(collection_name)

    def test_flush_without_reclaim_stops_at_flushed_stage(self, embedded_client):
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_noreclaim_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 100)

            result = embedded_client.flush(wait=True, reclaim_memory=False, timeout=60)

            assert result.completed
            assert result.stage == FlushStage.FLUSHED
            # Data remains readable after the MemTables were dumped to SSTables.
            assert collection.count() == 100
        finally:
            embedded_client.delete_collection(collection_name)

    def test_wait_false_returns_completion_handle(self, embedded_client):
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_handle_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 100)

            handle = embedded_client.flush(wait=False, reclaim_memory=True, timeout=60)

            assert isinstance(handle, FlushHandle)
            assert handle.target_stage == FlushStage.RECLAIMED
            assert handle.boundary_freeze_scn is not None

            result = handle.wait(timeout=60)
            assert result.completed
            assert result.stage == FlushStage.RECLAIMED
            assert handle.is_completed()
        finally:
            embedded_client.delete_collection(collection_name)

    def test_flush_timeout_reports_incomplete_stage_and_keeps_writes(self, embedded_client):
        """A tiny timeout reports the incomplete stage; committed writes are not undone."""
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_timeout_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            ids = _add_batch(collection, 0, 100)

            with pytest.raises(FlushTimeoutError) as excinfo:
                embedded_client.flush(wait=True, reclaim_memory=True, timeout=0.1)

            error = excinfo.value
            assert error.timeout_seconds == 0.1
            assert error.stage in (FlushStage.SUBMITTED, FlushStage.FLUSHED)
            # Committed writes survive the timeout untouched.
            assert collection.count() == 100
            got = collection.get(ids=ids[:3])
            assert set(got["ids"]) == set(ids[:3])

            # Background flushing continues; a later flush reaches the target.
            followup = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)
            assert followup.completed
        finally:
            embedded_client.delete_collection(collection_name)

    def test_repeated_flush_calls_have_defined_behavior(self, embedded_client):
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_repeat_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        try:
            _add_batch(collection, 0, 50)

            first = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)
            assert first.completed
            # An immediate second flush on an idle instance also completes.
            second = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)
            assert second.completed
            assert second.boundary_freeze_scn > first.boundary_freeze_scn
            assert collection.count() == 50

            # A third flush after more writes still succeeds.
            _add_batch(collection, 100, 50)
            third = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)
            assert third.completed
            assert collection.count() == 100
        finally:
            embedded_client.delete_collection(collection_name)

    def test_concurrent_writes_continue_during_flush(self, embedded_client):
        """Writes after the boundary continue into new MemTables and stay readable."""
        _skip_if_engine_lacks_flush_support(embedded_client)
        collection_name = f"test_flush_concurrent_{int(time.time() * 1000)}"
        collection = _make_collection(embedded_client, collection_name)
        _add_batch(collection, 0, 200)

        # The concurrent writer uses its own client (own session) so it never
        # shares a connection with the flush's polls.
        writer_client = pyseekdb.Client(path=embedded_client._server.path, database=embedded_client._server.database)
        writer_collection = writer_client.get_collection(collection_name)

        def keep_writing() -> None:
            for start in (1000, 2000, 3000):
                _add_batch(writer_collection, start, 100)

        writer = threading.Thread(target=keep_writing)
        writer.start()
        try:
            result = embedded_client.flush(wait=True, reclaim_memory=True, timeout=120)
            assert result.completed
        finally:
            writer.join()
            writer_client.close()

        # Every write, before and after the boundary, is present.
        assert collection.count() == 500
        embedded_client.delete_collection(collection_name)

    def test_flush_does_not_commit_open_transaction(self, embedded_client):
        """An open application transaction must not be implicitly committed by flush."""
        _skip_if_engine_lacks_flush_support(embedded_client)
        probe_table = f"flush_tx_probe_{int(time.time() * 1000)}"
        client_server = embedded_client._server
        client_server._execute(f"CREATE TABLE `{probe_table}` (id INT PRIMARY KEY, msg VARCHAR(256))")
        try:
            client_server._execute(f"INSERT INTO `{probe_table}` VALUES (1, 'committed')")

            raw = embedded_client._server.get_raw_connection()
            raw_cursor = raw.cursor()
            try:
                raw_cursor.execute("BEGIN")
                raw_cursor.execute(f"INSERT INTO `{probe_table}` VALUES (999999, 'uncommitted')")

                # Flush while the transaction is open; the freeze command runs
                # on a dedicated session and must not commit it.
                result = embedded_client.flush(wait=True, reclaim_memory=True, timeout=60)
                assert result.completed

                raw_cursor.execute("ROLLBACK")
            finally:
                raw_cursor.close()

            rows = client_server._execute(f"SELECT COUNT(*) AS c FROM `{probe_table}`")
            count = rows[0].get("c") if isinstance(rows[0], dict) else rows[0][0]
            # Only the committed row: the flush never committed the transaction.
            assert count == 1
        finally:
            client_server._execute(f"DROP TABLE `{probe_table}`")

    def test_flush_rejects_unsupported_options(self, embedded_client):
        _skip_if_engine_lacks_flush_support(embedded_client)
        with pytest.raises(TypeError, match="wait"):
            embedded_client.flush(wait="yes")
        with pytest.raises(TypeError, match="reclaim_memory"):
            embedded_client.flush(reclaim_memory=1)
        with pytest.raises(ValueError, match="timeout"):
            embedded_client.flush(timeout=0)
        with pytest.raises(ValueError, match="timeout"):
            embedded_client.flush(timeout=-5)

    def test_flush_unsupported_in_remote_mode(self):
        """Remote server mode returns a clear error instead of a partial flush."""
        remote = pyseekdb.Client(host="127.0.0.1", port=2881, user="root", password="")
        try:
            with pytest.raises(ValueError, match="embedded client"):
                remote.flush()
        finally:
            remote.close()
