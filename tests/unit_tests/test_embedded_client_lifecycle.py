"""Unit tests for embedded pylibseekdb instance ownership."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from pyseekdb.client.admin_client import _AdminClientProxy, _ClientProxy


class _FakeConnection:
    def __init__(self, path: str, events: list[tuple[Any, ...]]) -> None:
        self.path = path
        self._events = events

    def close(self) -> None:
        self._events.append(("connection.close", self.path))


class _FakeInstance:
    def __init__(self, path: str, events: list[tuple[Any, ...]], fail_connect: bool = False) -> None:
        self.path = path
        self._events = events
        self._fail_connect = fail_connect

    def connect(self, *, database: str, autocommit: bool) -> _FakeConnection:
        self._events.append(("instance.connect", self.path, database, autocommit))
        if self._fail_connect:
            raise RuntimeError("connect failed")
        return _FakeConnection(self.path, self._events)

    def close(self) -> None:
        self._events.append(("instance.close", self.path))


class _FakeInstanceApi:
    SeekdbInstance = _FakeInstance

    def __init__(self, *, fail_first_connect: bool = False) -> None:
        self.events: list[tuple[Any, ...]] = []
        self._fail_first_connect = fail_first_connect

    def open(self, *, db_dir: str) -> _FakeInstance:
        self.events.append(("open", db_dir))
        fail_connect = self._fail_first_connect
        self._fail_first_connect = False
        return _FakeInstance(db_dir, self.events, fail_connect=fail_connect)

    def connect(self, *, database: str, autocommit: bool) -> _FakeConnection:
        raise AssertionError("the module-level connection must not be used by the instance API")


class _FakeLegacyApi:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def open(self, *, db_dir: str) -> None:
        self.events.append(("open", db_dir))

    def connect(self, *, database: str, autocommit: bool) -> _FakeConnection:
        self.events.append(("module.connect", database, autocommit))
        return _FakeConnection("legacy", self.events)


@pytest.fixture
def embedded_module(monkeypatch: pytest.MonkeyPatch):
    """Import the embedded module without loading the native pylibseekdb extension."""
    module_name = "pyseekdb.client.client_seekdb_embedded"
    monkeypatch.setitem(sys.modules, "pylibseekdb", ModuleType("pylibseekdb"))
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def _install_fake_seekdb(embedded_module: Any, monkeypatch: pytest.MonkeyPatch, fake_seekdb: Any) -> None:
    monkeypatch.setattr(embedded_module, "seekdb", fake_seekdb)
    monkeypatch.setattr(embedded_module, "_PYLIBSEEKDB_AVAILABLE", True)


def test_clients_use_and_close_their_own_seekdb_instances(tmp_path, monkeypatch, embedded_module) -> None:
    fake_seekdb = _FakeInstanceApi()
    _install_fake_seekdb(embedded_module, monkeypatch, fake_seekdb)
    path_a = str((tmp_path / "a").resolve())
    path_b = str((tmp_path / "b").resolve())

    client_a = embedded_module.SeekdbEmbeddedClient(path=path_a, database="db_a")
    client_b = embedded_module.SeekdbEmbeddedClient(path=path_b, database="db_b")

    assert client_a.get_raw_connection().path == path_a
    assert client_b.get_raw_connection().path == path_b

    client_a.close()
    assert client_b.is_connected()
    client_b.close()
    client_b.close()

    assert fake_seekdb.events == [
        ("open", path_a),
        ("instance.connect", path_a, "db_a", True),
        ("open", path_b),
        ("instance.connect", path_b, "db_b", True),
        ("connection.close", path_a),
        ("instance.close", path_a),
        ("connection.close", path_b),
        ("instance.close", path_b),
    ]


def test_connect_failure_releases_new_instance_and_allows_retry(tmp_path, monkeypatch, embedded_module) -> None:
    fake_seekdb = _FakeInstanceApi(fail_first_connect=True)
    _install_fake_seekdb(embedded_module, monkeypatch, fake_seekdb)
    path = str((tmp_path / "db").resolve())
    client = embedded_module.SeekdbEmbeddedClient(path=path)

    with pytest.raises(RuntimeError, match="connect failed"):
        client.get_raw_connection()

    assert client._instance is None
    assert not client._initialized
    assert not client.is_connected()
    assert fake_seekdb.events[-1] == ("instance.close", path)

    assert client.get_raw_connection().path == path
    client.close()
    assert [event for event in fake_seekdb.events if event[0] == "open"] == [("open", path), ("open", path)]


def test_legacy_module_api_remains_compatible(tmp_path, monkeypatch, embedded_module) -> None:
    fake_seekdb = _FakeLegacyApi()
    _install_fake_seekdb(embedded_module, monkeypatch, fake_seekdb)
    path = str((tmp_path / "legacy").resolve())
    client = embedded_module.SeekdbEmbeddedClient(path=path)

    client.get_raw_connection()
    client.close()
    client.get_raw_connection()
    client.close()

    assert [event for event in fake_seekdb.events if event[0] == "open"] == [("open", path)]
    assert [event for event in fake_seekdb.events if event[0] == "module.connect"] == [
        ("module.connect", "test", True),
        ("module.connect", "test", True),
    ]


@pytest.mark.parametrize("proxy_type", [_ClientProxy, _AdminClientProxy])
def test_public_proxy_close_delegates_to_server(proxy_type) -> None:
    class _Server:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    server = _Server()
    proxy = proxy_type(server)

    proxy.close()

    assert server.close_count == 1
