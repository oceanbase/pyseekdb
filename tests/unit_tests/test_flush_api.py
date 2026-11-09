"""Unit tests for the explicit flush API (embedded MemStore flush, issue #288)."""

from __future__ import annotations

import importlib
import sys
import time
from types import ModuleType
from typing import Any

import pytest

from pyseekdb.client.admin_client import _ClientProxy
from pyseekdb.client.client_seekdb_server import RemoteServerClient
from pyseekdb.client.flush import (
    FlushError,
    FlushHandle,
    FlushResult,
    FlushStage,
    FlushTimeoutError,
    FlushTracker,
    FlushUnsupportedError,
    make_connection_sql_runner,
    validate_flush_options,
)

_MAX_SCN = 4611686018427387903
_AGGREGATE_SQL = "SELECT FREEZE_CNT, MEMSTORE_USED FROM oceanbase.GV$OB_MEMSTORE"
_MEMTABLE_INFO_SQL = "SELECT TABLET_ID, IS_ACTIVE, START_SCN, END_SCN FROM oceanbase.GV$OB_MEMSTORE_INFO"
_HISTORY_SQL = (
    "SELECT TABLET_ID, COMPACTION_SCN FROM oceanbase.GV$OB_TABLET_COMPACTION_HISTORY WHERE TYPE = 'MINI_MERGE'"
)
_MINOR_FREEZE_SQL = "ALTER SYSTEM MINOR FREEZE"


class FakeFlushEngine:
    """Scripted embedded engine serving the flush observability views.

    ``dump_after_polls`` / ``release_after_polls`` count MemTable-info reads;
    setting them to ``None`` makes the corresponding step never happen.
    """

    def __init__(
        self,
        *,
        active: list[tuple[int, int]] | None = None,
        used_before: int = 1_000_000,
        used_after: int = 300_000,
        dump_after_polls: int = 1,
        release_after_polls: int = 2,
        freeze_noop: bool = False,
        history_fails: bool = False,
        dict_rows: bool = False,
    ) -> None:
        self.active = active if active is not None else [(1, 100), (2, 100)]
        self.used_before = used_before
        self.used_after = used_after
        self.dump_after_polls = dump_after_polls
        self.release_after_polls = release_after_polls
        self.freeze_noop = freeze_noop
        self.history_fails = history_fails
        self.dict_rows = dict_rows
        self.freeze_cnt = 0
        self.memstore_used = used_before
        self.frozen: list[tuple[int, int, int]] = []
        self.dumped: set[tuple[int, int]] = set()
        self.info_polls = 0
        self.next_scn = 1000
        self.sql_log: list[tuple[str, str]] = []  # (connection_id, sql)

    # ---- engine behavior -------------------------------------------------

    def apply_freeze(self) -> None:
        if self.freeze_noop:
            return
        self.freeze_cnt += 1
        boundary = self.next_scn
        self.next_scn += 1000
        # Earlier frozen MemTables stay pending; a new freeze only adds to them.
        self.frozen = self.frozen + [(tablet, start, boundary) for tablet, start in self.active]
        self.active = []

    def _tick(self) -> None:
        if not self.frozen:
            # Nothing frozen yet (e.g. the pre-submission boundary read).
            return
        self.info_polls += 1
        if self.dump_after_polls is not None and self.info_polls >= self.dump_after_polls:
            self.dumped = {(tablet, end) for tablet, _start, end in self.frozen}
        release_due = self.release_after_polls is not None and self.info_polls >= self.release_after_polls
        dump_allows_release = self.dump_after_polls is None or (
            self.release_after_polls is not None and self.release_after_polls >= self.dump_after_polls
        )
        if release_due and dump_allows_release:
            self.frozen = []
            self.memstore_used = self.used_after

    # ---- SQL surface -----------------------------------------------------

    def run_sql(self, connection_id: str, sql: str) -> list[Any]:
        self.sql_log.append((connection_id, sql))
        if sql == _MINOR_FREEZE_SQL:
            self.apply_freeze()
            return []
        if sql == _AGGREGATE_SQL:
            row = (self.freeze_cnt, self.memstore_used)
            return [{"FREEZE_CNT": row[0], "MEMSTORE_USED": row[1]}] if self.dict_rows else [row]
        if sql == _MEMTABLE_INFO_SQL:
            self._tick()
            rows = [(tablet, "YES", start, _MAX_SCN) for tablet, start in self.active]
            rows += [(tablet, "NO", start, end) for tablet, start, end in self.frozen]
            if self.dict_rows:
                return [
                    {
                        "TABLET_ID": r[1],
                        "IS_ACTIVE": "YES" if r[0] == "active" else "NO",
                        "START_SCN": r[2],
                        "END_SCN": r[3],
                    }
                    for r in rows
                ]
            return rows
        if sql == _HISTORY_SQL:
            if self.history_fails:
                raise RuntimeError("history view unavailable")
            rows = [(tablet, scn) for tablet, scn in self.dumped]
            return [{"TABLET_ID": r[0], "COMPACTION_SCN": r[1]} for r in rows] if self.dict_rows else rows
        raise AssertionError(f"unexpected SQL: {sql}")


# ==================== validation ====================


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wait": "yes"},
        {"reclaim_memory": 1},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": "30"},
        {"timeout": True},
    ],
)
def test_flush_option_validation_rejects_unsupported_values(kwargs) -> None:
    defaults: dict[str, Any] = {"wait": True, "reclaim_memory": True, "timeout": 30.0}
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        validate_flush_options(**defaults)


def test_flush_option_validation_accepts_valid_values() -> None:
    validate_flush_options(wait=True, reclaim_memory=True, timeout=30)
    validate_flush_options(wait=False, reclaim_memory=False, timeout=0.5)


# ==================== FlushTracker state machine ====================


def test_tracker_reclaim_happy_path_reports_measurements() -> None:
    engine = FakeFlushEngine(used_before=1_000_000, used_after=300_000)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=True)

    tracker.capture_boundary()
    assert tracker.run_sql is not None
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    stage = FlushStage.SUBMITTED
    for _ in range(10):
        stage = tracker.poll()
        if stage == FlushStage.RECLAIMED:
            break
    assert stage == FlushStage.RECLAIMED

    result = tracker.build_result(stage)
    assert result.completed
    assert result.stage == FlushStage.RECLAIMED
    assert result.frozen_memtable_count == 2
    assert result.released_memtable_count == 2
    assert result.memstore_used_before_bytes == 1_000_000
    assert result.memstore_used_after_bytes == 300_000
    assert result.memstore_reclaimed_bytes == 700_000
    assert result.boundary_freeze_scn == 1000
    assert result.elapsed_seconds >= 0


def test_tracker_without_reclaim_completes_at_flushed_while_memtables_held() -> None:
    # Release never happens (a long-lived reference keeps the MemTable), but
    # the dumps are visible in compaction history, so the flush stage completes.
    engine = FakeFlushEngine(release_after_polls=None, dump_after_polls=1)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=False)

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    stages = [tracker.poll() for _ in range(5)]
    assert stages[-1] == FlushStage.FLUSHED
    result = tracker.build_result(stages[-1])
    assert result.completed
    assert result.stage == FlushStage.FLUSHED
    assert result.frozen_memtable_count == 2
    assert result.released_memtable_count == 0


def test_tracker_timeout_during_release_reports_incomplete_stage() -> None:
    engine = FakeFlushEngine(release_after_polls=None, dump_after_polls=1)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=True)

    tracker.capture_boundary()
    tracker.submit()
    deadline = time.monotonic() + 0.5
    tracker.confirm_freeze(deadline=deadline, timeout_seconds=30)

    with pytest.raises(FlushTimeoutError) as excinfo:
        while True:
            stage = tracker.poll()
            if stage == tracker.target_stage:
                break
            if time.monotonic() >= deadline + 0.3:
                raise tracker.timeout_error(stage, 0.5)
            time.sleep(0.05)

    error = excinfo.value
    assert error.stage == FlushStage.FLUSHED
    assert error.pending_memtable_count == 2
    assert error.frozen_memtable_count == 2
    assert "not undone" in str(error)
    assert "released" in str(error)


def test_tracker_timeout_during_freeze_confirmation() -> None:
    engine = FakeFlushEngine(freeze_noop=True)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=True)

    tracker.capture_boundary()
    tracker.submit()
    with pytest.raises(FlushTimeoutError) as excinfo:
        tracker.confirm_freeze(deadline=time.monotonic() + 0.1, timeout_seconds=0.1)

    assert excinfo.value.stage == FlushStage.SUBMITTED
    assert "confirm the freeze" in str(excinfo.value)


def test_tracker_timeout_during_dump_reports_flush_stage() -> None:
    engine = FakeFlushEngine(dump_after_polls=None, release_after_polls=None)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=False)

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    with pytest.raises(FlushTimeoutError) as excinfo:
        while True:
            stage = tracker.poll()
            if stage == tracker.target_stage:
                break
            raise tracker.timeout_error(stage, 0.5)

    assert excinfo.value.stage == FlushStage.SUBMITTED
    assert "flushed to SSTables" in str(excinfo.value)


def test_tracker_falls_back_to_release_only_completion_without_history() -> None:
    # Compaction history is unavailable: a MemTable counts as flushed only once
    # it is released, so the flush stage cannot complete while it is held.
    engine = FakeFlushEngine(history_fails=True, release_after_polls=None, dump_after_polls=1)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=False)

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    assert tracker.poll() == FlushStage.SUBMITTED

    engine.release_after_polls = 3
    engine.dump_after_polls = 3
    stages = [tracker.poll() for _ in range(4)]
    assert stages[-1] == FlushStage.FLUSHED


def test_tracker_accepts_dict_cursor_rows() -> None:
    engine = FakeFlushEngine(dict_rows=True, used_before=500, used_after=100)
    tracker = FlushTracker(lambda sql: engine.run_sql("admin", sql), reclaim_memory=True)

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    for _ in range(5):
        if tracker.poll() == FlushStage.RECLAIMED:
            break
    result = tracker.build_result(FlushStage.RECLAIMED)
    assert result.memstore_used_before_bytes == 500
    assert result.memstore_used_after_bytes == 100
    assert result.memstore_reclaimed_bytes == 400


def test_tracker_waits_for_leftovers_from_earlier_flush() -> None:
    # MemTables frozen by an earlier flush still hold writes committed before
    # this flush's boundary, so a new flush also waits for them.
    engine = FakeFlushEngine(active=[(1, 100), (2, 100)], release_after_polls=None)
    first = FlushTracker(lambda sql: engine.run_sql("a", sql), reclaim_memory=True)
    first.capture_boundary()
    first.submit()
    first.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    engine.active = [(2, 2000)]
    second = FlushTracker(lambda sql: engine.run_sql("a", sql), reclaim_memory=True)
    second.capture_boundary()
    second.submit()
    second.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)
    # Two MemTables left over from the first flush plus one newly frozen one.
    assert second._boundary_frozen_count == 3

    engine.release_after_polls = engine.info_polls + 2
    for _ in range(10):
        if second.poll() == FlushStage.RECLAIMED:
            break
    assert second.poll() == FlushStage.RECLAIMED


# ==================== SQL runner ====================


class _FakeNativeCursor:
    description = None

    def __init__(self, engine: FakeFlushEngine, connection_id: str) -> None:
        self._engine = engine
        self._connection_id = connection_id

    def execute(self, sql: str) -> None:
        self._pending = self._engine.run_sql(self._connection_id, sql)

    def fetchall(self) -> list[Any]:
        return self._pending

    def close(self) -> None:
        pass


class _FakeNativeConnection:
    def __init__(self, engine: FakeFlushEngine, connection_id: str) -> None:
        self._engine = engine
        self._connection_id = connection_id
        self.closed = False

    def cursor(self) -> _FakeNativeCursor:
        return _FakeNativeCursor(self._engine, self._connection_id)

    def close(self) -> None:
        self.closed = True


def test_connection_sql_runner_never_fetches_non_query_results() -> None:
    engine = FakeFlushEngine()
    connection = _FakeNativeConnection(engine, "admin")
    runner = make_connection_sql_runner(connection, use_context_manager=False)

    runner(_MINOR_FREEZE_SQL)
    rows = runner(_AGGREGATE_SQL)

    assert rows == [(1, 1_000_000)]
    assert _flush_sql_statements(engine) == [_MINOR_FREEZE_SQL, _AGGREGATE_SQL]


# ==================== SeekdbEmbeddedClient.flush ====================


class _FakeSeekdbModuleApi:
    """Legacy module-level binding whose connections share one scripted engine."""

    def __init__(self, engine: FakeFlushEngine) -> None:
        self.engine = engine
        self.connections: list[_FakeNativeConnection] = []

    def open(self, *, db_dir: str) -> None:
        return None

    def connect(self, *, database: str, autocommit: bool) -> _FakeNativeConnection:
        connection = _FakeNativeConnection(self.engine, f"conn{len(self.connections)}")
        self.connections.append(connection)
        return connection


@pytest.fixture
def embedded_module(monkeypatch: pytest.MonkeyPatch):
    """Import the embedded module without loading the native pylibseekdb extension."""
    module_name = "pyseekdb.client.client_seekdb_embedded"
    monkeypatch.setitem(sys.modules, "pylibseekdb", ModuleType("pylibseekdb"))
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def _make_client(embedded_module, monkeypatch, engine, path):
    fake_seekdb = _FakeSeekdbModuleApi(engine)
    monkeypatch.setattr(embedded_module, "seekdb", fake_seekdb)
    monkeypatch.setattr(embedded_module, "_PYLIBSEEKDB_AVAILABLE", True)
    client = embedded_module.SeekdbEmbeddedClient(path=str(path), database="test")
    return client, fake_seekdb


def _flush_sql_statements(engine: FakeFlushEngine) -> list[str]:
    return [sql for _conn, sql in engine.sql_log]


def test_embedded_flush_happy_path_uses_dedicated_session_and_main_polls(
    tmp_path, monkeypatch, embedded_module
) -> None:
    engine = FakeFlushEngine(used_before=1_000_000, used_after=300_000, release_after_polls=2)
    client, fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    result = client.flush(wait=True, reclaim_memory=True, timeout=10)

    assert isinstance(result, FlushResult)
    assert result.completed
    assert result.stage == FlushStage.RECLAIMED
    assert result.memstore_reclaimed_bytes == 700_000

    statements = _flush_sql_statements(engine)
    # The freeze command runs exactly once and never commits a transaction.
    assert statements.count(_MINOR_FREEZE_SQL) == 1
    assert not any("COMMIT" in sql.upper() for sql in statements)
    # The freeze was submitted on a dedicated session, not the client's main one.
    alter_connection = next(conn for conn, sql in engine.sql_log if sql == _MINOR_FREEZE_SQL)
    assert alter_connection != "conn0"
    # The main session stays open; the dedicated session is closed after submit.
    assert fake_seekdb.connections[0].closed is False
    admin_connection = next(c for c in fake_seekdb.connections if c._connection_id == alter_connection)
    assert admin_connection.closed is True


def test_embedded_flush_with_wait_false_returns_pollable_handle(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeFlushEngine(release_after_polls=3)
    client, _fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    handle = client.flush(wait=False, reclaim_memory=True, timeout=10)

    assert isinstance(handle, FlushHandle)
    assert handle.target_stage == FlushStage.RECLAIMED
    assert handle.boundary_freeze_scn is not None

    result = handle.wait(timeout=10)
    assert isinstance(result, FlushResult)
    assert result.completed
    assert result.stage == FlushStage.RECLAIMED
    # Completion was polled through the main session, never a new connection.
    assert _flush_sql_statements(engine).count(_MINOR_FREEZE_SQL) == 1


def test_embedded_flush_handle_wait_validates_timeout(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeFlushEngine(release_after_polls=3)
    client, _fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")
    handle = client.flush(wait=False, timeout=10)

    with pytest.raises(ValueError, match="timeout"):
        handle.wait(timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        handle.wait(timeout="soon")


def test_embedded_flush_timeout_reports_pending_release(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeFlushEngine(release_after_polls=None, dump_after_polls=1)
    client, _fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    with pytest.raises(FlushTimeoutError) as excinfo:
        client.flush(wait=True, reclaim_memory=True, timeout=0.5)

    error = excinfo.value
    assert error.stage == FlushStage.FLUSHED
    assert error.pending_memtable_count == 2
    # Committed writes are untouched: no rollback/commit statements were issued.
    statements = _flush_sql_statements(engine)
    assert not any("COMMIT" in sql.upper() or "ROLLBACK" in sql.upper() for sql in statements)


def test_embedded_flush_unsupported_observability_raises_clear_error(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeFlushEngine()

    class _FailingCursor(_FakeNativeCursor):
        def execute(self, sql: str) -> None:
            if sql == _AGGREGATE_SQL:
                raise RuntimeError("Table 'oceanbase.gv$ob_memstore' doesn't exist")
            self._pending = engine.run_sql(self._connection_id, sql)

    fake_seekdb = _FakeSeekdbModuleApi(engine)
    monkeypatch.setattr(embedded_module, "seekdb", fake_seekdb)
    monkeypatch.setattr(embedded_module, "_PYLIBSEEKDB_AVAILABLE", True)

    original_connect = fake_seekdb.connect

    def connect_with_failing_views(*, database: str, autocommit: bool) -> _FakeNativeConnection:
        connection = original_connect(database=database, autocommit=autocommit)
        connection.cursor = lambda: _FailingCursor(engine, connection._connection_id)  # type: ignore[method-assign]
        return connection

    fake_seekdb.connect = connect_with_failing_views  # type: ignore[method-assign]
    client = embedded_module.SeekdbEmbeddedClient(path=str(tmp_path / "db"), database="test")

    with pytest.raises(FlushUnsupportedError, match="does not expose the MemStore observability"):
        client.flush(wait=True, timeout=5)


def test_embedded_flush_freeze_rejection_raises_flush_error(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeFlushEngine()

    class _RejectingCursor(_FakeNativeCursor):
        def execute(self, sql: str) -> None:
            if sql == _MINOR_FREEZE_SQL:
                raise RuntimeError("admin command rejected")
            self._pending = engine.run_sql(self._connection_id, sql)

    fake_seekdb = _FakeSeekdbModuleApi(engine)
    monkeypatch.setattr(embedded_module, "seekdb", fake_seekdb)
    monkeypatch.setattr(embedded_module, "_PYLIBSEEKDB_AVAILABLE", True)

    original_connect = fake_seekdb.connect

    def connect_with_rejecting_cursor(*, database: str, autocommit: bool) -> _FakeNativeConnection:
        connection = original_connect(database=database, autocommit=autocommit)
        connection.cursor = lambda: _RejectingCursor(engine, connection._connection_id)  # type: ignore[method-assign]
        return connection

    fake_seekdb.connect = connect_with_rejecting_cursor  # type: ignore[method-assign]
    client = embedded_module.SeekdbEmbeddedClient(path=str(tmp_path / "db"), database="test")

    with pytest.raises(FlushError, match="ALTER SYSTEM MINOR FREEZE"):
        client.flush(wait=True, timeout=5)


def test_embedded_flush_validates_options_before_touching_the_engine(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeFlushEngine()
    client, _fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    with pytest.raises(TypeError, match="wait"):
        client.flush(wait="please")
    with pytest.raises(ValueError, match="timeout"):
        client.flush(timeout=0)

    assert engine.sql_log == []


# ==================== unsupported modes and proxy delegation ====================


def test_remote_server_client_flush_raises_clear_error() -> None:
    client = RemoteServerClient(host="127.0.0.1", port=2881)
    with pytest.raises(ValueError, match="embedded client"):
        client.flush()


def test_client_proxy_flush_delegates_to_server() -> None:
    calls: list[dict[str, Any]] = []

    class _Server:
        def flush(self, wait=True, reclaim_memory=True, timeout=30.0) -> str:
            calls.append({"wait": wait, "reclaim_memory": reclaim_memory, "timeout": timeout})
            return "flushed"

    proxy = _ClientProxy(_Server())
    assert proxy.flush(wait=False, reclaim_memory=False, timeout=5) == "flushed"
    assert calls == [{"wait": False, "reclaim_memory": False, "timeout": 5}]

    # The default proxy signature forwards flush() to the underlying server.
    assert proxy.flush() == "flushed"
    assert calls[-1] == {"wait": True, "reclaim_memory": True, "timeout": 30.0}


def test_flush_result_repr_is_informative() -> None:
    result = FlushResult(
        stage=FlushStage.RECLAIMED,
        boundary_freeze_scn=1000,
        frozen_memtable_count=2,
        released_memtable_count=2,
        memstore_used_before_bytes=100,
        memstore_used_after_bytes=40,
        memstore_reclaimed_bytes=60,
        elapsed_seconds=0.25,
    )
    text = repr(result)
    assert "reclaimed" in text
    assert "60" in text
