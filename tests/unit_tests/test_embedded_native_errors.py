"""Unit tests for mapping native embedded binding errors to PyMySQL errors (issue #242)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pymysql
import pytest


class SeekdbError(RuntimeError):
    """Stand-in for pylibseekdb.SeekdbError."""


class _FailingNativeCursor:
    def __init__(self, error: BaseException, events: list[str]) -> None:
        self._error = error
        self._events = events

    def execute(self, sql: str) -> int:
        self._events.append(f"execute:{sql}")
        raise self._error

    def close(self) -> None:
        self._events.append("cursor.close")


class _FailingNativeConnection:
    def __init__(self, error: BaseException, events: list[str]) -> None:
        self._error = error
        self._events = events

    def cursor(self) -> _FailingNativeCursor:
        return _FailingNativeCursor(self._error, self._events)


@pytest.fixture
def embedded_module(monkeypatch: pytest.MonkeyPatch):
    """Import the embedded module without loading the native pylibseekdb extension."""
    module_name = "pyseekdb.client.client_seekdb_embedded"
    monkeypatch.setitem(sys.modules, "pylibseekdb", ModuleType("pylibseekdb"))
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


@pytest.fixture
def native_client(tmp_path, monkeypatch: pytest.MonkeyPatch, embedded_module: Any):
    # A legacy binding without SeekdbInstance.connection_options selects the native cursor path.
    monkeypatch.setattr(embedded_module, "seekdb", ModuleType("pylibseekdb"))
    monkeypatch.setattr(embedded_module, "_PYLIBSEEKDB_AVAILABLE", True)
    client = embedded_module.SeekdbEmbeddedClient(path=str(tmp_path / "db"))
    assert not client._backend.supports_dbapi_cursor
    return client


def _run(client: Any, error: BaseException) -> tuple[BaseException, list[str]]:
    events: list[str] = []
    conn = _FailingNativeConnection(error, events)
    with pytest.raises(Exception) as exc_info:
        client._execute_query_with_cursor(conn, "DESCRIBE `test`.`some_table`", [], use_context_manager=False)
    return exc_info.value, events


def test_missing_table_from_released_binding_becomes_programming_error(native_client) -> None:
    original = SeekdbError("Table `test.some_table` does not exist failed: code=1146")

    raised, events = _run(native_client, original)

    assert type(raised) is pymysql.err.ProgrammingError
    assert raised.args == (1146, "Table `test.some_table` does not exist")
    assert raised.__cause__ is original
    assert events == ["execute:DESCRIBE `test`.`some_table`", "cursor.close"]


def test_structured_error_code_is_preferred(native_client) -> None:
    original = SeekdbError(1146, "Table 'test.some_table' doesn't exist")
    original.code = 1146
    original.errno = 1146
    original.msg = "Table 'test.some_table' doesn't exist"

    raised, _ = _run(native_client, original)

    assert type(raised) is pymysql.err.ProgrammingError
    assert raised.args == (1146, "Table 'test.some_table' doesn't exist")
    assert raised.__cause__ is original


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (1062, pymysql.err.IntegrityError),
        (1064, pymysql.err.ProgrammingError),
        (4012, pymysql.err.OperationalError),
        (600, pymysql.err.InternalError),
    ],
)
def test_error_class_matches_pymysql(native_client, code: int, expected: type) -> None:
    raised, _ = _run(native_client, SeekdbError(f"some server error failed: code={code}"))

    assert type(raised) is expected
    assert raised.args[0] == code


@pytest.mark.parametrize(
    "original",
    [
        RuntimeError("Cursor.execute: no connection"),
        SeekdbError("seekdb_query failed: code=-1"),
        SeekdbError("syntax error near 'code=1146' in the middle"),
        pymysql.err.ProgrammingError(1146, "already a PyMySQL error"),
    ],
)
def test_errors_without_server_code_propagate_unchanged(native_client, original: BaseException) -> None:
    raised, events = _run(native_client, original)

    assert raised is original
    assert events[-1] == "cursor.close"
