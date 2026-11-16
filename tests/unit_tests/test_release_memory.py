"""Unit tests for the release_memory API (embedded memory release, issue #288)."""

from __future__ import annotations

import importlib
import sys
import time
from types import ModuleType
from typing import Any

import pytest

from pyseekdb.client.admin_client import _ClientProxy
from pyseekdb.client.client_seekdb_server import RemoteServerClient
from pyseekdb.client.memory import (
    MemoryReleaseResult,
    MemoryReleaseTimeoutError,
    MemoryReleaseTracker,
    make_connection_sql_runner,
    validate_release_memory_options,
)

_MAX_SCN = 4611686018427387903
_AGGREGATE_SQL = "SELECT FREEZE_CNT, MEMSTORE_USED FROM oceanbase.GV$OB_MEMSTORE"
_MEMTABLE_INFO_SQL = "SELECT TABLET_ID, IS_ACTIVE, START_SCN, END_SCN FROM oceanbase.GV$OB_MEMSTORE_INFO"
_MINOR_FREEZE_SQL = "ALTER SYSTEM MINOR FREEZE"


class FakeReleaseEngine:
    """Scripted embedded engine serving the memory-release observability views.

    ``release_after_polls`` counts MemTable-info reads; ``None`` makes the
    release never happen. ``dict_rows`` serves dict-cursor rows instead of
    positional tuples.
    """

    def __init__(
        self,
        *,
        active: list[tuple[int, int]] | None = None,
        used_before: int = 1_000_000,
        used_after: int = 300_000,
        release_after_polls: int = 2,
        freeze_noop: bool = False,
        dict_rows: bool = False,
    ) -> None:
        self.active = active if active is not None else [(1, 100), (2, 100)]
        self.used_before = used_before
        self.used_after = used_after
        self.release_after_polls = release_after_polls
        self.freeze_noop = freeze_noop
        self.dict_rows = dict_rows
        self.freeze_cnt = 0
        self.memstore_used = used_before
        self.frozen: list[tuple[int, int, int]] = []
        self.info_polls = 0
        self.next_scn = 1000
        self.sql_log: list[tuple[str, str]] = []  # (connection_id, sql)

    def apply_freeze(self) -> None:
        if self.freeze_noop:
            return
        self.freeze_cnt += 1
        boundary = self.next_scn
        self.next_scn += 1000
        # Earlier frozen tables stay pending; a new freeze only adds to them.
        self.frozen = self.frozen + [(tablet, start, boundary) for tablet, start in self.active]
        self.active = []

    def _tick(self) -> None:
        if not self.frozen:
            # Nothing frozen yet (e.g. the pre-submission boundary read).
            return
        self.info_polls += 1
        if self.release_after_polls is not None and self.info_polls >= self.release_after_polls:
            self.frozen = []
            self.memstore_used = self.used_after

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
                        "TABLET_ID": r[0],
                        "IS_ACTIVE": r[1],
                        "START_SCN": r[2],
                        "END_SCN": r[3],
                    }
                    for r in rows
                ]
            return rows
        raise AssertionError(f"unexpected SQL: {sql}")


# ==================== validation ====================


@pytest.mark.parametrize("timeout", [0, -1, "30", True, None])
def test_release_memory_option_validation_rejects_unsupported_values(timeout) -> None:
    with pytest.raises(ValueError, match="timeout"):
        validate_release_memory_options(timeout)


def test_release_memory_option_validation_accepts_valid_values() -> None:
    validate_release_memory_options(30)
    validate_release_memory_options(0.5)


# ==================== MemoryReleaseTracker ====================


def test_tracker_happy_path_reports_before_after_bytes() -> None:
    engine = FakeReleaseEngine(used_before=1_000_000, used_after=300_000)
    tracker = MemoryReleaseTracker(lambda sql: engine.run_sql("admin", sql))

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    for _ in range(10):
        if tracker.released():
            break
    assert tracker.released()

    result = tracker.build_result()
    assert result.memstore_used_before_bytes == 1_000_000
    assert result.memstore_used_after_bytes == 300_000
    assert result.reclaimed_bytes == 700_000


def test_tracker_timeout_reports_still_held_tables() -> None:
    engine = FakeReleaseEngine(release_after_polls=None)
    tracker = MemoryReleaseTracker(lambda sql: engine.run_sql("admin", sql))

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    assert tracker.released() is False
    error = tracker.timeout_error(0.5)
    assert isinstance(error, MemoryReleaseTimeoutError)
    assert "still held" in str(error)
    assert "2 in-memory table(s)" in str(error)
    assert "Committed writes are unaffected" in str(error)


def test_tracker_timeout_when_freeze_never_confirmed() -> None:
    engine = FakeReleaseEngine(freeze_noop=True)
    tracker = MemoryReleaseTracker(lambda sql: engine.run_sql("admin", sql))

    tracker.capture_boundary()
    tracker.submit()
    with pytest.raises(MemoryReleaseTimeoutError, match="not confirmed"):
        tracker.confirm_freeze(deadline=time.monotonic() + 0.1, timeout_seconds=0.1)


def test_tracker_accepts_dict_cursor_rows() -> None:
    engine = FakeReleaseEngine(dict_rows=True, used_before=500, used_after=100)
    tracker = MemoryReleaseTracker(lambda sql: engine.run_sql("admin", sql))

    tracker.capture_boundary()
    tracker.submit()
    tracker.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)

    for _ in range(10):
        if tracker.released():
            break
    result = tracker.build_result()
    assert result.memstore_used_before_bytes == 500
    assert result.memstore_used_after_bytes == 100
    assert result.reclaimed_bytes == 400


def test_tracker_waits_for_leftovers_from_earlier_release() -> None:
    # Tables frozen by an earlier call still hold writes committed before
    # this release, so the new call also waits for them.
    engine = FakeReleaseEngine(active=[(1, 100), (2, 100)], release_after_polls=None)
    first = MemoryReleaseTracker(lambda sql: engine.run_sql("a", sql))
    first.capture_boundary()
    first.submit()
    first.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)
    assert not first.released()

    engine.active = [(2, 2000)]
    second = MemoryReleaseTracker(lambda sql: engine.run_sql("a", sql))
    second.capture_boundary()
    second.submit()
    second.confirm_freeze(deadline=time.monotonic() + 5, timeout_seconds=5)
    assert not second.released()  # both the leftovers and the new table

    engine.release_after_polls = engine.info_polls + 2
    for _ in range(10):
        if second.released():
            break
    assert second.released()


# ==================== SQL runner ====================


class _FakeNativeCursor:
    description = None

    def __init__(self, engine: FakeReleaseEngine, connection_id: str) -> None:
        self._engine = engine
        self._connection_id = connection_id
        self._pending: list[Any] = []

    def execute(self, sql: str) -> None:
        self._pending = self._engine.run_sql(self._connection_id, sql)

    def fetchall(self) -> list[Any]:
        return self._pending

    def close(self) -> None:
        pass


class _FakeNativeConnection:
    def __init__(self, engine: FakeReleaseEngine, connection_id: str) -> None:
        self._engine = engine
        self._connection_id = connection_id
        self.closed = False

    def cursor(self) -> _FakeNativeCursor:
        return _FakeNativeCursor(self._engine, self._connection_id)

    def close(self) -> None:
        self.closed = True


def test_connection_sql_runner_never_fetches_non_query_results() -> None:
    engine = FakeReleaseEngine()
    connection = _FakeNativeConnection(engine, "admin")
    runner = make_connection_sql_runner(connection, use_context_manager=False)

    runner(_MINOR_FREEZE_SQL)
    rows = runner(_AGGREGATE_SQL)

    assert rows == [(1, 1_000_000)]
    assert [sql for _conn, sql in engine.sql_log] == [_MINOR_FREEZE_SQL, _AGGREGATE_SQL]


# ==================== SeekdbEmbeddedClient.release_memory ====================


class _FakeSeekdbModuleApi:
    """Legacy module-level binding whose connections share one scripted engine."""

    def __init__(self, engine: FakeReleaseEngine) -> None:
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


def _sql_statements(engine: FakeReleaseEngine) -> list[str]:
    return [sql for _conn, sql in engine.sql_log]


def test_embedded_release_memory_uses_dedicated_session_and_main_polls(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeReleaseEngine(used_before=1_000_000, used_after=300_000, release_after_polls=2)
    client, fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    result = client.release_memory(timeout=10)

    assert isinstance(result, MemoryReleaseResult)
    assert result.memstore_used_before_bytes == 1_000_000
    assert result.memstore_used_after_bytes == 300_000
    assert result.reclaimed_bytes == 700_000

    statements = _sql_statements(engine)
    # The freeze command runs exactly once and never commits a transaction.
    assert statements.count(_MINOR_FREEZE_SQL) == 1
    assert not any("COMMIT" in sql.upper() or "ROLLBACK" in sql.upper() for sql in statements)
    # The freeze was submitted on a dedicated session, not the client's main one.
    alter_connection = next(conn for conn, sql in engine.sql_log if sql == _MINOR_FREEZE_SQL)
    assert alter_connection != "conn0"
    # The main session stays open; the dedicated session is closed after submit.
    assert fake_seekdb.connections[0].closed is False
    admin_connection = next(c for c in fake_seekdb.connections if c._connection_id == alter_connection)
    assert admin_connection.closed is True


def test_embedded_release_memory_timeout_reports_pending(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeReleaseEngine(release_after_polls=None)
    client, _fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    with pytest.raises(MemoryReleaseTimeoutError) as excinfo:
        client.release_memory(timeout=0.5)

    assert "still held" in str(excinfo.value)
    # Committed writes are untouched: no rollback/commit statements were issued.
    assert not any("COMMIT" in sql.upper() or "ROLLBACK" in sql.upper() for sql in _sql_statements(engine))


def test_embedded_release_memory_unsupported_views_raise_clear_error(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeReleaseEngine()

    class _FailingCursor(_FakeNativeCursor):
        def execute(self, sql: str) -> None:
            if sql == _AGGREGATE_SQL:
                raise RuntimeError("Table 'oceanbase.gv$ob_memstore' doesn't exist")
            self._pending = self._engine.run_sql(self._connection_id, sql)

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

    with pytest.raises(ValueError, match="not supported by the installed seekdb binding"):
        client.release_memory(timeout=5)


def test_embedded_release_memory_freeze_rejection_raises_value_error(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeReleaseEngine()

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

    with pytest.raises(ValueError, match="failed to submit the engine minor freeze"):
        client.release_memory(timeout=5)


def test_embedded_release_memory_validates_timeout_before_touching_the_engine(
    tmp_path, monkeypatch, embedded_module
) -> None:
    engine = FakeReleaseEngine()
    client, _fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    with pytest.raises(ValueError, match="timeout"):
        client.release_memory(timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        client.release_memory(timeout="soon")

    assert engine.sql_log == []


# ==================== unsupported modes and proxy delegation ====================


def test_remote_server_client_release_memory_raises_clear_error() -> None:
    client = RemoteServerClient(host="127.0.0.1", port=2881)
    with pytest.raises(ValueError, match="embedded client"):
        client.release_memory()


def test_client_proxy_release_memory_delegates_to_server() -> None:
    calls: list[float] = []

    class _Server:
        def release_memory(self, timeout: float = 30.0) -> str:
            calls.append(timeout)
            return "released"

    proxy = _ClientProxy(_Server())
    assert proxy.release_memory(timeout=5) == "released"
    assert calls == [5]
    # The default signature forwards release_memory() with its default timeout.
    assert proxy.release_memory() == "released"
    assert calls == [5, 30.0]


def test_release_memory_result_repr_is_informative() -> None:
    result = MemoryReleaseResult(memstore_used_before_bytes=100, memstore_used_after_bytes=40)
    text = repr(result)
    assert "reclaimed_bytes=60" in text
