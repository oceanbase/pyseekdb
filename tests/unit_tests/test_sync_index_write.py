"""Write-time sync option for asynchronous index refresh."""

import inspect
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pymysql.err import OperationalError

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pyseekdb.client.client_base import (  # noqa: E402
    INDEX_REFRESH_MIN_VERSION,
    BaseClient,
    _index_refresh_supported,
)
from pyseekdb.client.collection import Collection  # noqa: E402
from pyseekdb.client.namespace import Namespace  # noqa: E402
from pyseekdb.client.version import Version  # noqa: E402

REFRESH_SQL = "CALL dbms_index_manager.refresh();"


class RecordingCursor:
    """Cursor that records SQL and returns no rows."""

    def __init__(self, client: "RecordingClient"):
        self.client = client
        self.description = None

    def execute(self, sql, params=None):
        self.client.sqls.append(" ".join(sql.split()))

    def fetchall(self):
        return []

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RecordingConnection:
    """Connection whose cursors record SQL."""

    def __init__(self, client: "RecordingClient"):
        self.client = client

    def cursor(self):
        return RecordingCursor(self.client)


class RecordingClient(BaseClient):
    """BaseClient that records SQL and reports a fixed database version."""

    def __init__(self, db_type: str, version: Version):
        self.database = "test"
        self.db_type = db_type
        self.version = version
        self.sqls: list[str] = []
        self.fail_insert = False
        self.fail_refresh = False
        self.namespace_alive = True

    def detect_db_type_and_version(self):
        return self.db_type, self.version

    def _execute(self, sql, params=None):
        normalized = " ".join(str(sql).split())
        upper = normalized.upper()
        if self.fail_insert and upper.startswith("INSERT"):
            raise RuntimeError("insert failed")
        if self.fail_refresh and "dbms_index_manager.refresh" in normalized.lower():
            self.sqls.append(normalized)
            raise OperationalError(4012, "Timeout, query has reached the maximum query timeout")
        self.sqls.append(normalized)
        return None

    def _ensure_connection(self):
        return RecordingConnection(self)

    def _ns_collection_exists_by_id(self, collection_id: str) -> bool:
        return self.namespace_alive

    def _ns_namespace_exists_by_id(self, collection_id: str, namespace_id: str) -> bool:
        return self.namespace_alive

    def is_connected(self):
        return True

    def _cleanup(self):
        return None

    def get_raw_connection(self):
        return None

    @property
    def mode(self):
        return "RecordingClient"

    def create_collection(self, *args, **kwargs):
        return None

    def get_collection(self, *args, **kwargs):
        return None

    def delete_collection(self, *args, **kwargs):
        return None

    def list_collections(self):
        return []

    def has_collection(self, name):
        return False

    def create_database(self, *args, **kwargs):
        return None

    def get_database(self, *args, **kwargs):
        return None

    def delete_database(self, *args, **kwargs):
        return None

    def list_databases(self, *args, **kwargs):
        return []

    def fork_database(self, *args, **kwargs):
        return None


def _collection(client: RecordingClient, *, use_namespace: bool = False) -> Collection:
    return Collection(
        client=client,
        name="docs",
        collection_id="42",
        dimension=3,
        use_namespace=use_namespace,
        has_vector_index=True,
    )


def _namespace(client: RecordingClient) -> Namespace:
    client._cache_namespace_ltable_id("42", 7, 1)
    collection = _collection(client, use_namespace=True)
    return Namespace(client=client, collection=collection, name="ns", namespace_id="7")


def _refresh_count(sqls: list[str]) -> int:
    return sum(1 for sql in sqls if "dbms_index_manager.refresh" in sql.lower())


def _supported_client() -> RecordingClient:
    return RecordingClient("seekdb", Version("1.3.0.0"))


class TestIndexRefreshSupport:
    """Version gate shared by refresh_index() and sync=True."""

    def test_minimum_version_constant(self):
        assert str(INDEX_REFRESH_MIN_VERSION) == "1.3.0.0"

    @pytest.mark.parametrize(
        ("db_type", "version", "supported"),
        [
            ("seekdb", "1.3.0.0", True),
            ("SeekDB", "1.3.1.0", True),
            ("seekdb", "1.2.0.0", False),
            ("seekdb", "1.2.9.9", False),
            ("oceanbase", "4.6.1.0", False),
            ("mysql", "8.0.0", False),
        ],
    )
    def test_index_refresh_supported(self, db_type, version, supported):
        assert _index_refresh_supported(db_type, Version(version)) is supported

    def test_refresh_index_is_noop_when_unsupported(self):
        client = RecordingClient("oceanbase", Version("4.6.1.0"))
        client.refresh_index()
        assert client.sqls == []

    def test_refresh_index_calls_procedure_when_supported(self):
        client = _supported_client()
        client.refresh_index()
        assert client.sqls == [REFRESH_SQL]

    def test_assert_rejects_unsupported_backend(self):
        client = RecordingClient("seekdb", Version("1.2.0.0"))
        with pytest.raises(ValueError, match="sync=True"):
            client._assert_index_refresh_supported()

    def test_assert_accepts_minimum_seekdb(self):
        _supported_client()._assert_index_refresh_supported()


class TestCollectionSyncWrite:
    """Collection add/update/upsert/delete honor sync."""

    def test_sync_is_keyword_only_and_defaults_false(self):
        for method_name in ("add", "update", "upsert", "delete"):
            parameter = inspect.signature(getattr(Collection, method_name)).parameters["sync"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is False

    def test_default_add_does_not_refresh(self):
        client = _supported_client()
        _collection(client).add(ids="row1", embeddings=[0.1, 0.2, 0.3], documents="hello")
        assert any(sql.upper().startswith("INSERT") for sql in client.sqls)
        assert _refresh_count(client.sqls) == 0

    def test_explicit_sync_false_does_not_refresh(self):
        client = _supported_client()
        _collection(client).update(ids="row1", metadatas={"tag": "a"}, sync=False)
        assert any(sql.upper().startswith("UPDATE") for sql in client.sqls)
        assert _refresh_count(client.sqls) == 0

    def test_sync_add_refreshes_after_insert(self):
        client = _supported_client()
        _collection(client).add(ids="row1", embeddings=[0.1, 0.2, 0.3], sync=True)
        assert any(sql.upper().startswith("INSERT") for sql in client.sqls)
        assert client.sqls[-1] == REFRESH_SQL
        assert _refresh_count(client.sqls) == 1

    def test_sync_update_upsert_and_delete_refresh_once(self):
        client = _supported_client()
        collection = _collection(client)
        collection.update(ids="row1", embeddings=[0.2, 0.3, 0.4], sync=True)
        collection.upsert(ids="row2", embeddings=[0.3, 0.4, 0.5], documents="new", sync=True)
        collection.delete(ids="row1", sync=True)
        assert _refresh_count(client.sqls) == 3
        assert [sql for sql in client.sqls if "dbms_index_manager.refresh" in sql] == [REFRESH_SQL] * 3
        assert any(sql.upper().startswith("UPDATE") for sql in client.sqls)
        assert any(sql.upper().startswith("INSERT") for sql in client.sqls)
        assert any(sql.upper().startswith("DELETE") for sql in client.sqls)

    def test_sync_true_on_old_seekdb_does_not_write(self):
        client = RecordingClient("seekdb", Version("1.2.0.0"))
        with pytest.raises(ValueError, match="was not performed"):
            _collection(client).add(ids="row1", embeddings=[0.1, 0.2, 0.3], sync=True)
        assert client.sqls == []

    def test_sync_true_on_oceanbase_does_not_write(self):
        client = RecordingClient("oceanbase", Version("4.6.1.0"))
        with pytest.raises(ValueError, match=r"oceanbase 4\.6\.1\.0"):
            _collection(client).upsert(ids="row1", embeddings=[0.1, 0.2, 0.3], sync=True)
        assert client.sqls == []

    def test_invalid_sync_type_rejected_before_write(self):
        client = _supported_client()
        with pytest.raises(TypeError, match="sync must be a bool"):
            _collection(client).add(ids="row1", embeddings=[0.1, 0.2, 0.3], sync=1)
        assert client.sqls == []

    def test_refresh_timeout_propagates_after_committed_write(self, caplog):
        client = _supported_client()
        client.fail_refresh = True
        with caplog.at_level(logging.ERROR), pytest.raises(OperationalError) as exc_info:
            _collection(client).add(ids="row1", embeddings=[0.1, 0.2, 0.3], documents="kept", sync=True)
        assert exc_info.value.args[0] == 4012
        assert any(sql.upper().startswith("INSERT") for sql in client.sqls)
        assert client.sqls[-1] == REFRESH_SQL
        assert "Write committed" in caplog.text

    def test_write_failure_does_not_refresh(self):
        client = _supported_client()
        client.fail_insert = True
        with pytest.raises(RuntimeError, match="insert failed"):
            _collection(client).add(ids="row1", embeddings=[0.1, 0.2, 0.3], sync=True)
        assert _refresh_count(client.sqls) == 0

    def test_extra_kwargs_still_forwarded_without_refresh(self):
        client = MagicMock()
        collection = Collection(
            client=client,
            name="docs",
            collection_id="42",
            sparse_vector_index_config="cfg",
        )
        collection.add(ids="row1", embeddings=[0.1], trace_id="abc")
        forwarded = client._collection_add.call_args.kwargs
        assert forwarded["sparse_vector_index_config"] == "cfg"
        assert forwarded["trace_id"] == "abc"
        client.refresh_index.assert_not_called()
        client._assert_index_refresh_supported.assert_not_called()

    def test_namespace_collection_guard_runs_before_sync(self):
        client = _supported_client()
        with pytest.raises(ValueError, match="namespace enabled"):
            _collection(client, use_namespace=True).delete(ids="row1", sync=True)
        assert client.sqls == []


class TestNamespaceSyncWrite:
    """Namespace writes use the same sync option and refresh once."""

    def test_sync_is_keyword_only_and_defaults_false(self):
        for method_name in ("add", "update", "upsert", "delete"):
            parameter = inspect.signature(getattr(Namespace, method_name)).parameters["sync"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is False

    def test_default_add_does_not_refresh(self):
        client = _supported_client()
        _namespace(client).add(ids="doc_1", embeddings=[0.1, 0.2, 0.3])
        assert any(sql.upper().startswith("INSERT") for sql in client.sqls)
        assert _refresh_count(client.sqls) == 0

    def test_sync_add_update_delete_refresh_after_write(self):
        client = _supported_client()
        namespace = _namespace(client)
        namespace.add(ids="doc_1", embeddings=[0.1, 0.2, 0.3], documents="one", sync=True)
        namespace.update(ids="doc_1", metadatas={"tag": "b"}, sync=True)
        namespace.delete(ids="doc_1", sync=True)
        assert _refresh_count(client.sqls) == 3
        assert client.sqls[-1] == REFRESH_SQL

    def test_sync_upsert_refreshes_once(self):
        client = _supported_client()
        _namespace(client).upsert(
            ids=["doc_1", "doc_2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["one", "two"],
            sync=True,
        )
        assert any(sql.upper().startswith("INSERT") for sql in client.sqls)
        assert _refresh_count(client.sqls) == 1
        assert client.sqls[-1] == REFRESH_SQL

    def test_unsupported_backend_does_not_write(self):
        client = RecordingClient("oceanbase", Version("4.6.1.0"))
        with pytest.raises(ValueError, match="sync=True"):
            _namespace(client).add(ids="doc_1", embeddings=[0.1, 0.2, 0.3], sync=True)
        assert client.sqls == []

    def test_refresh_failure_propagates(self):
        client = _supported_client()
        client.fail_refresh = True
        with pytest.raises(OperationalError):
            _namespace(client).update(ids="doc_1", embeddings=[0.2, 0.2, 0.2], sync=True)
        assert _refresh_count(client.sqls) == 1

    def test_deleted_namespace_rejects_sync_before_write(self):
        client = _supported_client()
        namespace = _namespace(client)
        client.namespace_alive = False
        with pytest.raises(ValueError, match="no longer exists"):
            namespace.add(ids="doc_1", embeddings=[0.1, 0.2, 0.3], sync=True)
        assert client.sqls == []
        assert _refresh_count(client.sqls) == 0
