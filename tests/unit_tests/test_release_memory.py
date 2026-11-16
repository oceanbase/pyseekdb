"""Unit tests for the release_memory API (embedded memory release request, issue #288)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from pyseekdb.client.admin_client import _ClientProxy
from pyseekdb.client.client_seekdb_server import RemoteServerClient
from pyseekdb.client.memory import (
    _MINOR_FREEZE_SQL,
    make_connection_sql_runner,
    submit_memory_release,
)


class FakeReleaseEngine:
    """Scripted embedded engine recording every statement it receives."""

    def __init__(self) -> None:
        self.sql_log: list[tuple[str, str]] = []  # (connection_id, sql)

    def run_sql(self, connection_id: str, sql: str) -> list[Any]:
        self.sql_log.append((connection_id, sql))
        if sql == _MINOR_FREEZE_SQL:
            return []
        raise AssertionError(f"unexpected SQL: {sql}")


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


# ==================== submit helper and SQL runner ====================


def test_submit_memory_release_executes_the_engine_command() -> None:
    engine = FakeReleaseEngine()

    submit_memory_release(lambda sql: engine.run_sql("admin", sql))

    assert engine.sql_log == [("admin", _MINOR_FREEZE_SQL)]


def test_submit_memory_release_raises_value_error_when_rejected() -> None:
    def reject(sql: str) -> list[Any]:
        raise RuntimeError("admin command rejected")

    with pytest.raises(ValueError, match="failed to submit the engine memory release"):
        submit_memory_release(reject)


def test_connection_sql_runner_never_fetches_non_query_results() -> None:
    engine = FakeReleaseEngine()
    connection = _FakeNativeConnection(engine, "admin")
    runner = make_connection_sql_runner(connection, use_context_manager=False)

    assert runner(_MINOR_FREEZE_SQL) == []
    assert engine.sql_log == [("admin", _MINOR_FREEZE_SQL)]


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


def test_embedded_release_memory_submits_on_dedicated_session_and_returns(
    tmp_path, monkeypatch, embedded_module
) -> None:
    engine = FakeReleaseEngine()
    client, fake_seekdb = _make_client(embedded_module, monkeypatch, engine, tmp_path / "db")

    result = client.release_memory()

    # Fire-and-forget: no result, exactly one engine statement, nothing polled.
    assert result is None
    assert [sql for _conn, sql in engine.sql_log] == [_MINOR_FREEZE_SQL]
    # Never commits or rolls back a transaction.
    statements = [sql for _conn, sql in engine.sql_log]
    assert not any("COMMIT" in sql.upper() or "ROLLBACK" in sql.upper() for sql in statements)
    # The release command ran on a dedicated session, not the client's main one.
    alter_connection = next(conn for conn, sql in engine.sql_log if sql == _MINOR_FREEZE_SQL)
    assert alter_connection != "conn0"
    # The main session stays open; the dedicated session is closed immediately.
    assert fake_seekdb.connections[0].closed is False
    admin_connection = next(c for c in fake_seekdb.connections if c._connection_id == alter_connection)
    assert admin_connection.closed is True


def test_embedded_release_memory_rejection_raises_value_error(tmp_path, monkeypatch, embedded_module) -> None:
    engine = FakeReleaseEngine()

    class _RejectingCursor(_FakeNativeCursor):
        def execute(self, sql: str) -> None:
            raise RuntimeError("admin command rejected")

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

    with pytest.raises(ValueError, match="failed to submit the engine memory release"):
        client.release_memory()


# ==================== unsupported modes and proxy delegation ====================


def test_remote_server_client_release_memory_raises_clear_error() -> None:
    client = RemoteServerClient(host="127.0.0.1", port=2881)
    with pytest.raises(ValueError, match="embedded client"):
        client.release_memory()


def test_client_proxy_release_memory_delegates_to_server() -> None:
    calls: list[int] = []

    class _Server:
        def release_memory(self) -> str:
            calls.append(1)
            return "released"

    proxy = _ClientProxy(_Server())
    assert proxy.release_memory() == "released"
    assert calls == [1]
