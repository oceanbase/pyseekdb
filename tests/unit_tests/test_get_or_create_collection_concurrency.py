"""
Unit tests for concurrent-safe get_or_create_collection helpers.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import (  # noqa: E402
    BaseClient,
    _is_catalog_bootstrap_transient_error,
    _is_catalog_table_missing_error,
    _is_collection_conflict_error,
    _is_sdk_collection_catalog_conflict_error,
)
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402


class TestCollectionCatalogConflictDetection:
    """TestCollectionCatalogConflictDetection class."""

    def test_detects_integrity_error_on_sdk_collections(self):
        """Test detects integrity error on sdk collections."""

        class IntegrityError(Exception):
            """IntegrityError class."""

            pass

        exc = IntegrityError("(1062, \"Duplicate entry 'my_coll' for key 'uk_sdk_coll_name'\")")
        assert _is_sdk_collection_catalog_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        """Test ignores unrelated errors."""
        assert not _is_sdk_collection_catalog_conflict_error(ValueError("invalid dimension"))


class TestCollectionCatalogInsertRecovery:
    """TestCollectionCatalogInsertRecovery class."""

    def test_insert_conflict_reuses_existing_collection_id(self):
        """Test insert conflict reuses existing collection id."""
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.side_effect = [ValueError("not found"), "existing_id"]
        conn = MagicMock()
        client._ensure_connection.return_value = conn

        class IntegrityError(Exception):
            """IntegrityError class."""

            pass

        def execute_side_effect(sql):
            """Execute side effect."""
            if "INSERT INTO" in sql:
                raise IntegrityError("(1062, \"Duplicate entry 'items' for key 'uk_sdk_coll_name'\")")
            return []

        client._execute.side_effect = execute_side_effect

        result = BaseClient._create_collection_meta_v2(client, "items", None)

        assert result["collection_id"] == "existing_id"
        conn.rollback.assert_called_once()
        assert client._get_collection_id.call_count == 2

    def test_existing_catalog_row_is_reused_without_insert(self):
        """Test existing catalog row is reused without insert."""
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.return_value = "existing_id"

        result = BaseClient._create_collection_meta_v2(client, "items", None)

        assert result["collection_id"] == "existing_id"
        insert_calls = [call for call in client._execute.call_args_list if "INSERT INTO" in str(call)]
        assert not insert_calls


class TestCollectionCatalogBootstrap:
    """Tests for state-driven, retry-bounded catalog initialization."""

    @staticmethod
    def _client(execute_side_effect):
        """Build a client mock with the real catalog state helpers bound."""
        client = MagicMock(spec=BaseClient)
        client._qtable.side_effect = lambda table: f"`test_db`.`{table}`"
        client._use_catalog_database = MagicMock()
        client._rollback_connection_if_supported = MagicMock()
        client._execute.side_effect = execute_side_effect
        client._catalog_table_is_visible = BaseClient._catalog_table_is_visible.__get__(client, BaseClient)
        client._catalog_unique_index_is_ready = BaseClient._catalog_unique_index_is_ready.__get__(client, BaseClient)
        client._index_row_value = BaseClient._index_row_value
        return client

    @staticmethod
    def _unique_index_row(column="collection_name", *, non_unique=0):
        """Build one tuple-shaped SHOW INDEX row."""
        return ("sdk_collections", non_unique, "uk_sdk_coll_name", 1, column)

    def test_fast_path_uses_no_ddl_when_catalog_is_ready(self):
        """A ready catalog is only probed and never receives redundant DDL."""

        def execute(sql):
            if sql.startswith("SHOW INDEX"):
                return [self._unique_index_row()]
            return []

        client = self._client(execute)

        BaseClient._ensure_catalog_table(
            client,
            "sdk_collections",
            "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
            {"uk_sdk_coll_name": ("collection_name",)},
        )

        sql_calls = [call.args[0] for call in client._execute.call_args_list]
        assert not any(sql.startswith("CREATE") for sql in sql_calls)
        assert sum(sql.startswith("SHOW INDEX") for sql in sql_calls) == 1

    def test_missing_table_is_created_and_inline_unique_index_is_verified(self):
        """A missing table is created once and its inline unique key avoids extra index DDL."""
        state = {"table": False}

        def execute(sql):
            if sql.startswith("SELECT 1") and not state["table"]:
                raise RuntimeError('(1146, "Table test_db.sdk_collections doesn\'t exist")')
            if sql.startswith("CREATE TABLE"):
                state["table"] = True
                return None
            if sql.startswith("SHOW INDEX"):
                return [self._unique_index_row()]
            return []

        client = self._client(execute)

        BaseClient._ensure_catalog_table(
            client,
            "sdk_collections",
            "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
            {"uk_sdk_coll_name": ("collection_name",)},
        )

        sql_calls = [call.args[0] for call in client._execute.call_args_list]
        assert sum(sql.startswith("CREATE TABLE") for sql in sql_calls) == 1
        assert not any(sql.startswith("CREATE UNIQUE INDEX") for sql in sql_calls)

    def test_existing_legacy_table_adds_only_the_missing_unique_index(self):
        """An old table without the required index receives exactly one index DDL."""
        state = {"index": False}

        def execute(sql):
            if sql.startswith("SHOW INDEX"):
                return [self._unique_index_row()] if state["index"] else []
            if sql.startswith("CREATE UNIQUE INDEX"):
                state["index"] = True
            return []

        client = self._client(execute)

        BaseClient._ensure_catalog_table(
            client,
            "sdk_collections",
            "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
            {"uk_sdk_coll_name": ("collection_name",)},
        )

        sql_calls = [call.args[0] for call in client._execute.call_args_list]
        assert sum(sql.startswith("CREATE UNIQUE INDEX") for sql in sql_calls) == 1
        assert not any(sql.startswith("CREATE TABLE") for sql in sql_calls)

    def test_duplicate_index_race_rechecks_state_before_succeeding(self):
        """A concurrent 1061 is retried and accepted only after the index is visible."""
        state = {"create_attempts": 0}

        def execute(sql):
            if sql.startswith("SHOW INDEX"):
                return [self._unique_index_row()] if state["create_attempts"] else []
            if sql.startswith("CREATE UNIQUE INDEX"):
                state["create_attempts"] += 1
                raise RuntimeError('(1061, "Duplicate key name uk_sdk_coll_name")')
            return []

        client = self._client(execute)

        with patch("pyseekdb.client.client_base.time.sleep") as sleep:
            BaseClient._ensure_catalog_table(
                client,
                "sdk_collections",
                "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
                {"uk_sdk_coll_name": ("collection_name",)},
            )

        assert state["create_attempts"] == 1
        client._rollback_connection_if_supported.assert_called_once()
        sleep.assert_called_once()

    def test_persistent_schema_visibility_race_exhausts_bounded_retries(self):
        """Persistent 1146 errors are not swallowed as successful initialization."""

        def execute(sql):
            if sql.startswith("SELECT 1"):
                raise RuntimeError('(1146, "Table test_db.sdk_collections doesn\'t exist")')
            return []

        client = self._client(execute)

        with (
            patch("pyseekdb.client.client_base.time.sleep") as sleep,
            pytest.raises(RuntimeError, match="not visible after CREATE TABLE"),
        ):
            BaseClient._ensure_catalog_table(
                client,
                "sdk_collections",
                "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
                {"uk_sdk_coll_name": ("collection_name",)},
            )

        assert sleep.call_count == 5
        assert client._rollback_connection_if_supported.call_count == 6

    def test_non_transient_errors_are_not_retried(self):
        """Permission and other unrelated failures remain visible to callers."""

        def execute(_sql):
            raise RuntimeError('(1142, "SELECT command denied")')

        client = self._client(execute)

        with (
            patch("pyseekdb.client.client_base.time.sleep") as sleep,
            pytest.raises(RuntimeError, match="1142"),
        ):
            BaseClient._ensure_catalog_table(
                client,
                "sdk_collections",
                "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
                {"uk_sdk_coll_name": ("collection_name",)},
            )

        sleep.assert_not_called()
        client._rollback_connection_if_supported.assert_not_called()

    def test_wrong_existing_index_definition_is_rejected(self):
        """A same-named non-unique or wrong-column index is never treated as ready."""

        def execute(sql):
            if sql.startswith("SHOW INDEX"):
                return [self._unique_index_row(non_unique=1)]
            return []

        client = self._client(execute)

        with pytest.raises(ValueError, match="not UNIQUE"):
            BaseClient._ensure_catalog_table(
                client,
                "sdk_collections",
                "CREATE TABLE IF NOT EXISTS sdk_collections (...) ",
                {"uk_sdk_coll_name": ("collection_name",)},
            )

    def test_transient_error_detection_walks_cause_chain(self):
        """Wrapped database error codes remain classifiable."""
        inner = RuntimeError('(1146, "Table test_db.sdk_collections doesn\'t exist")')
        outer = ValueError("catalog bootstrap failed")
        outer.__cause__ = inner

        assert _is_catalog_table_missing_error(outer)
        assert _is_catalog_bootstrap_transient_error(outer)
        assert not _is_catalog_bootstrap_transient_error(RuntimeError('(1142, "SELECT command denied")'))

    def test_embedded_table_missing_error_format_is_transient(self):
        """Embedded reports symbolic names followed by a parenthesized numeric code."""
        exc = RuntimeError("execute sql failed OB_TABLE_NOT_EXIST(1146): Table '%s.%s' doesn't exist")

        assert _is_catalog_table_missing_error(exc)
        assert _is_catalog_bootstrap_transient_error(exc)


class TestCollectionConflictDetection:
    """TestCollectionConflictDetection class."""

    def test_detects_value_error_for_existing_collection(self):
        """Test detects value error for existing collection."""
        assert _is_collection_conflict_error(ValueError("Collection 'items' already exists"))

    def test_detects_seekdb_table_exists_error(self):
        """Test detects seekdb table exists error."""

        class SeekdbError(Exception):
            """SeekdbError class."""

            pass

        exc = SeekdbError("Table 'c$v2$abc' already exists failed: code=1050")
        assert _is_collection_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        """Test ignores unrelated errors."""
        assert not _is_collection_conflict_error(ValueError("invalid dimension"))

    def test_ignores_metadata_failure_without_conflict_cause(self):
        """Test ignores metadata failure without conflict cause."""
        assert not _is_collection_conflict_error(
            ValueError("Failed to create collection metadata: Collection not found: 'items'")
        )

    def test_detects_conflict_in_cause_chain(self):
        """Test detects conflict in cause chain."""
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        assert _is_collection_conflict_error(outer)


class TestGetOrCreateCollectionRecovery:
    """TestGetOrCreateCollectionRecovery class."""

    @staticmethod
    def _bind_resume_helper(client):
        """Bind resume helper."""
        client._get_or_resume_existing_collection = BaseClient._get_or_resume_existing_collection.__get__(
            client, BaseClient
        )

    def test_returns_existing_collection_after_create_conflict(self):
        """Test returns existing collection after create conflict."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        client.has_collection.return_value = False
        existing = object()
        client.get_collection.return_value = existing
        client.create_collection.side_effect = ValueError("Collection 'items' already exists")

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_retries_get_after_wrapped_table_conflict(self):
        """Test retries get after wrapped table conflict."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        client.has_collection.return_value = False
        existing = object()
        client.get_collection.return_value = existing
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        client.create_collection.side_effect = outer

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_conflict_on_namespace_collection_resumes_incomplete_handle(self):
        """Test conflict on namespace collection resumes incomplete handle."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        resumed = object()
        client.has_collection.return_value = False
        client._get_ns_collection_meta.return_value = {
            "collection_id": "abc",
            "collection_name": "items",
            "settings": {"use_namespace": True},
        }
        client._is_incomplete_ns_collection.return_value = True
        client.create_collection.side_effect = [
            ValueError("Collection 'items' already exists"),
            resumed,
        ]

        result = BaseClient.get_or_create_collection(client, "items", use_namespace=True)

        assert result is resumed
        assert client.create_collection.call_count == 2
        client.get_collection.assert_not_called()

    def test_conflict_on_namespace_collection_without_metadata_reraises(self):
        """Test conflict on namespace collection without metadata reraises."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        client.has_collection.return_value = False
        client._get_ns_collection_meta.return_value = None
        client.create_collection.side_effect = ValueError("Collection 'items' already exists")

        with pytest.raises(ValueError, match="namespace metadata is missing"):
            BaseClient.get_or_create_collection(client, "items", use_namespace=True)

    def test_resumes_incomplete_namespace_collection_when_present(self):
        """Test resumes incomplete namespace collection when present."""
        client = MagicMock(spec=BaseClient)
        resumed = object()
        client.has_collection.return_value = True
        client._is_incomplete_ns_collection.return_value = True
        client.create_collection.return_value = resumed

        result = BaseClient.get_or_create_collection(client, "items", use_namespace=True)

        assert result is resumed
        client.create_collection.assert_called_once()
        client.get_collection.assert_not_called()

    def test_does_not_mask_unrelated_create_errors(self):
        """Test does not mask unrelated create errors."""
        client = MagicMock(spec=BaseClient)
        client.has_collection.return_value = False
        client.create_collection.side_effect = ValueError("invalid dimension")

        with pytest.raises(ValueError, match="invalid dimension"):
            BaseClient.get_or_create_collection(client, "items")


class TestListNsNamespacesRecyclebinFilter:
    """TestListNsNamespacesRecyclebinFilter class."""

    def test_sql_excludes_recyclebin_rows(self):
        """Test sql excludes recyclebin rows."""
        client = MagicMock(spec=BaseClient)
        client._qtable.return_value = "`sdk_namespaces`"
        client._execute.return_value = [
            ("1", "active_ns"),
        ]

        result = BaseClient._list_ns_namespaces(client, "coll_1")

        assert result == [{"namespace_id": "1", "namespace_name": "active_ns"}]
        sql = client._execute.call_args[0][0]
        assert "__recyclebin_" in sql
        assert "LEFT(namespace_name, 13) <> '__recyclebin_'" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
