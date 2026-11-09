"""
Embedded mode client - based on seekdb
Available when the current ``seekdb`` binding or legacy ``pylibseekdb`` is installed.
"""

import importlib
import logging
import os
import threading
import time
from collections.abc import Sequence
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from .admin_client import DEFAULT_TENANT
from .client_base import BaseClient
from .database import Database
from .flush import (
    _POLL_INTERVAL_SECONDS,
    FlushError,
    FlushHandle,
    FlushResult,
    FlushTracker,
    FlushUnsupportedError,
    make_connection_sql_runner,
    validate_flush_options,
)
from .sql_utils import render_sql_with_params


def _load_embedded_runtime() -> tuple[Any | None, str | None]:
    """Prefer ``seekdb``; fall back only when the top-level binding is absent."""
    for distribution in ("seekdb", "pylibseekdb"):
        try:
            return importlib.import_module(distribution), distribution
        except ModuleNotFoundError as exc:
            if exc.name != distribution:
                raise
    return None, None


seekdb, _EMBEDDED_RUNTIME_DISTRIBUTION = _load_embedded_runtime()
_PYLIBSEEKDB_AVAILABLE = seekdb is not None

logger = logging.getLogger(__name__)


class _NativeEmbeddedBackend:
    """Connect through the embedded binding's native connection wrapper."""

    supports_dbapi_cursor = False

    @staticmethod
    def connect(instance: Any, database: str, connection_kwargs: dict[str, Any]) -> Any:
        del connection_kwargs
        if instance is not None:
            return instance.connect(database=database, autocommit=True)
        return seekdb.connect(database=database, autocommit=True)  # type: ignore[union-attr]

    @staticmethod
    def is_connection_open(connection: Any) -> bool:
        return connection is not None


class _PyMySQLEmbeddedBackend:
    """Connect to an owned SeekDB instance through its MySQL endpoint."""

    supports_dbapi_cursor = True

    @staticmethod
    def connect(instance: Any, database: str, connection_kwargs: dict[str, Any]) -> pymysql.Connection:
        if instance is None:
            raise RuntimeError("the embedded binding returned no SeekdbInstance for a PyMySQL connection")

        options = dict(instance.connection_options())
        kwargs = dict(connection_kwargs)

        # The endpoint and user belong to the lifecycle handle. Never let
        # caller-provided values redirect this connection to another instance.
        for key in ("host", "port", "unix_socket", "user"):
            kwargs.pop(key, None)
        kwargs.update(options)

        # Database selection remains caller-owned. BaseClient relies on a
        # dictionary cursor and autocommit behavior matching RemoteServerClient.
        kwargs.pop("db", None)
        kwargs["database"] = database
        kwargs.setdefault("charset", "utf8mb4")
        kwargs["cursorclass"] = DictCursor
        kwargs["autocommit"] = True
        return pymysql.connect(**kwargs)

    @staticmethod
    def is_connection_open(connection: Any) -> bool:
        return connection is not None and bool(getattr(connection, "open", False))


def _create_embedded_backend() -> _NativeEmbeddedBackend | _PyMySQLEmbeddedBackend:
    """Select the safest connection backend exposed by the installed binding."""
    instance_type = getattr(seekdb, "SeekdbInstance", None)
    if instance_type is not None and callable(getattr(instance_type, "connection_options", None)):
        return _PyMySQLEmbeddedBackend()
    return _NativeEmbeddedBackend()


class SeekdbEmbeddedClient(BaseClient):
    """Embedded seekdb client (lazy connection)

    Availability follows the installed embedded binding and its supported platform.
    """

    def __init__(self, path: str = "./seekdb.db", database: str = "test", **kwargs):
        """
        Initialize embedded client (no immediate connection)

        Args:
            path: seekdb data directory path
            database: database name

        Raises:
            RuntimeError: If no embedded binding is available
        """
        # Check if an embedded binding is available.
        if not _PYLIBSEEKDB_AVAILABLE or seekdb is None:
            raise RuntimeError(
                "Embedded Client is not available because no SeekDB Python binding was found. "
                "Please install seekdb (or legacy pylibseekdb), or use RemoteServerClient (host/port) instead."
            )

        self.path = os.path.abspath(path)
        # Create directory if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            logger.info(f"Created directory: {self.path}")
        elif not os.path.isdir(self.path):
            raise ValueError(f"Path exists but is not a directory: {self.path}")

        self.database = database
        self._connection_kwargs = dict(kwargs)
        self._backend = _create_embedded_backend()
        self._connection_lock = threading.RLock()
        self._instance = None
        self._connection = None
        self._initialized = False

        logger.info(f"Initialize SeekdbEmbeddedClient: path={self.path}, database={self.database}")

    # ==================== Connection Management ====================

    def _ensure_connection(self) -> Any:
        """Ensure connection is established (internal method)"""
        with self._connection_lock:
            if self._backend.is_connection_open(self._connection):
                return self._connection
            self._connection = None

            if not self._initialized:
                try:
                    self._instance = seekdb.open(db_dir=self.path)  # type: ignore[attr-defined]
                    logger.info(f"✅ seekdb opened: {self.path}")
                except Exception as exc:
                    # pylibseekdb < 1.4 exposes only a process-wide module API.
                    # Keep that legacy API working, but never hide this error for
                    # the instance API where each db_dir has independent ownership.
                    if hasattr(seekdb, "SeekdbInstance") or "initialized twice" not in str(exc):
                        raise
                    logger.debug(f"seekdb already opened through the legacy module API: {exc}")
                self._initialized = True

            try:
                connection = self._backend.connect(self._instance, self.database, self._connection_kwargs)
            except Exception:
                if self._instance is not None:
                    try:
                        self._instance.close()
                    except Exception as close_exc:
                        logger.warning("Failed to close seekdb instance after connection error: %s", close_exc)
                    finally:
                        self._instance = None
                        self._initialized = False
                raise

            self._connection = connection
            logger.info(f"✅ Connected to database: {self.database}")
            return self._connection

    def _cleanup(self) -> None:
        """Close the connection and its owning embedded runtime instance."""
        with self._connection_lock:
            connection = self._connection
            instance = self._instance
            self._connection = None
            self._instance = None

            # Legacy pylibseekdb has only a process-wide module instance. Leave
            # its initialization state intact because another pyseekdb client may
            # still be using it. The object API provides safe per-instance close.
            if instance is not None:
                self._initialized = False

            try:
                if connection is not None:
                    connection.close()
            finally:
                if instance is not None:
                    instance.close()

            if connection is not None or instance is not None:
                logger.info(f"Connection closed: path={self.path}, database={self.database}")

    def is_connected(self) -> bool:
        """Check connection status"""
        with self._connection_lock:
            return self._initialized and self._backend.is_connection_open(self._connection)

    def get_raw_connection(self) -> Any:
        """Get raw connection object"""
        return self._ensure_connection()

    @property
    def mode(self) -> str:
        """Return the client mode identifier."""
        return "SeekdbEmbeddedClient"

    def _use_context_manager_for_cursor(self) -> bool:
        """Use DB-API cursor contexts for PyMySQL, not for the legacy native cursor."""
        return self._backend.supports_dbapi_cursor

    def _execute_query_with_cursor(  # noqa: C901
        self, conn: Any, sql: str, params: list[Any], use_context_manager: bool = True
    ) -> list[dict[str, Any]]:
        """
        Execute SQL through DB-API for PyMySQL or adapt the legacy native cursor.

        Args:
            conn: Database connection
            sql: SQL query string with %s placeholders
            params: Query parameters
            use_context_manager: Whether the selected cursor supports a context manager

        Returns:
            List of normalized row dictionaries
        """
        if self._backend.supports_dbapi_cursor:
            return super()._execute_query_with_cursor(conn, sql, params, use_context_manager)

        # pyseekdb.Cursor.execute() only accepts SQL string, not parameters
        # Embed parameters directly into SQL
        embedded_sql = render_sql_with_params(sql, params)

        cursor = conn.cursor()
        try:
            cursor.execute(embedded_sql)

            # Check if this is a query statement (SELECT, SHOW, DESCRIBE, DESC)
            # Only query statements return result sets that need fetchall()
            is_query = self._should_fetch_results(cursor, embedded_sql)

            if not is_query:
                # For non-query statements (DELETE, UPDATE, INSERT, etc.), return empty list
                return []

            # For query statements, fetch results
            rows = cursor.fetchall()

            # pyseekdb.Cursor doesn't have description, extract column names from SQL
            cursor_description = getattr(cursor, "description", None)
            if cursor_description is None and rows:
                import re

                # Extract column names from SELECT clause using simple regex
                select_match = re.search(r"SELECT\s+(.+?)\s+FROM", embedded_sql, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1).strip()
                    # Split by comma, but skip commas inside parentheses (for function calls)
                    parts = []
                    depth = 0
                    current = ""
                    for char in select_clause:
                        if char == "(":
                            depth += 1
                        elif char == ")":
                            depth -= 1
                        elif char == "," and depth == 0:
                            parts.append(current.strip())
                            current = ""
                            continue
                        current += char
                    if current:
                        parts.append(current.strip())

                    # Extract column names: look for AS alias, otherwise use column name
                    column_names = []
                    for part in parts:
                        # Match "AS alias" pattern
                        as_match = re.search(r"\s+AS\s+(\w+)", part, re.IGNORECASE)
                        if as_match:
                            column_names.append(as_match.group(1))
                        else:
                            # No alias, extract column name (remove backticks, get identifier)
                            col = part.replace("`", "").strip().split()[-1]
                            column_names.append(col)

                    cursor_description = [(name,) for name in column_names]

            normalized_rows = []
            for row in rows:
                normalized_rows.append(self._normalize_row(row, cursor_description))
            return normalized_rows
        finally:
            cursor.close()

    # ==================== Explicit Flush (embedded only) ====================

    def flush(self, wait: bool = True, reclaim_memory: bool = True, timeout: float = 30.0) -> FlushResult | FlushHandle:
        """Explicitly flush committed writes and reclaim eligible MemStore memory.

        Intended for the end of a write-heavy batch or business phase: the
        instance-wide freeze covers every MemTable that was active when the
        boundary was submitted (data tables and their associated index
        MemTables), and concurrent writes continue into newly created active
        MemTables. The flush never implicitly commits an application
        transaction; the freeze command runs on a dedicated session.

        Usage: call this once per batch/phase, not as an automatic
        memory-pressure handler. Frequent small flushes create many small
        SSTables and increase read amplification; automatic threshold-based
        freezing still covers applications that do not call flush().

        Args:
            wait: With ``True`` (default), block until the target stage is
                reached or ``timeout`` expires. With ``False``, submit the
                freeze and return a :class:`FlushHandle` immediately.
            reclaim_memory: With ``True`` (default), also wait for the
                boundary MemTables to be released, reclaiming their MemStore
                memory, and report before/after usage. With ``False``, stop
                once every boundary MemTable is flushed to SSTables.
            timeout: Seconds to wait (default 30). Applies to freeze
                confirmation and stage completion.

        Returns:
            FlushResult when ``wait=True``; FlushHandle when ``wait=False``.

        Raises:
            FlushTimeoutError: The target stage was not reached in time. The
                error reports the incomplete stage and pending MemTables.
                Committed writes are never undone and background flushing
                continues.
            FlushUnsupportedError: The installed engine/binding does not
                expose the MemStore observability required by the flush
                completion contract.
            TypeError / ValueError: Unsupported argument values.

        Note:
            Reclaiming MemStore memory is a logical/held-memory guarantee:
            the engine's MemStore accounting drops once MemTables are
            released, but process RSS may stay higher because allocators
            legitimately retain cached pages.
        """
        validate_flush_options(wait, reclaim_memory, timeout)
        self._ensure_connection()

        admin_connection = self._open_flush_admin_connection()
        try:
            admin_runner = make_connection_sql_runner(
                admin_connection, use_context_manager=self._backend.supports_dbapi_cursor
            )
            tracker = FlushTracker(admin_runner, reclaim_memory=reclaim_memory)
            try:
                tracker.capture_boundary()
            except FlushError:
                raise
            except Exception as exc:
                raise FlushUnsupportedError(
                    "the embedded engine does not expose the MemStore observability views required by "
                    "flush() (oceanbase.GV$OB_MEMSTORE / GV$OB_MEMSTORE_INFO); the installed seekdb "
                    f"binding or engine version does not support the flush completion contract: {exc}"
                ) from exc
            try:
                tracker.submit()
            except FlushError:
                raise
            except Exception as exc:
                raise FlushError(f"flush failed to submit the minor freeze (ALTER SYSTEM MINOR FREEZE): {exc}") from exc
            tracker.confirm_freeze(time.monotonic() + timeout, timeout)
        finally:
            self._quietly_close(admin_connection)

        # Read-only completion polls can share the client's main session;
        # they cannot commit an application transaction.
        tracker.run_sql = make_connection_sql_runner(
            self._ensure_connection(), use_context_manager=self._use_context_manager_for_cursor()
        )
        if not wait:
            return FlushHandle(self._flush_wait, tracker, timeout_seconds=timeout)
        return self._flush_wait(tracker, timeout)

    def _flush_wait(self, tracker: FlushTracker, timeout: float) -> FlushResult:
        """Poll until the tracker's target stage is reached or the timeout expires."""
        deadline = time.monotonic() + timeout
        while True:
            stage = tracker.poll()
            if stage == tracker.target_stage:
                return tracker.build_result(stage)
            if time.monotonic() >= deadline:
                raise tracker.timeout_error(stage, timeout)
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _open_flush_admin_connection(self) -> Any:
        """Open a dedicated session for freeze submission.

        The flush command must never run on the client's main session: if the
        application holds an open transaction there (via get_raw_connection()),
        an ALTER on that session could implicitly commit it.
        """
        try:
            return self._backend.connect(self._instance, self.database, self._connection_kwargs)
        except Exception as exc:
            raise FlushUnsupportedError(f"flush could not open a dedicated engine session: {exc}") from exc

    @staticmethod
    def _quietly_close(connection: Any) -> None:
        try:
            connection.close()
        except Exception as exc:
            logger.debug("flush: failed to close dedicated session: %s", exc)

    # ==================== Collection Management (framework) ====================

    # create_collection is inherited from BaseClient - no override needed
    # get_collection is inherited from BaseClient - no override needed
    # delete_collection is inherited from BaseClient - no override needed
    # list_collections is inherited from BaseClient - no override needed
    # has_collection is inherited from BaseClient - no override needed

    # ==================== Collection Internal Operations ====================
    # These methods are called by Collection objects

    # -------------------- DML Operations --------------------
    # _collection_add is inherited from BaseClient
    # _collection_update is inherited from BaseClient
    # _collection_upsert is inherited from BaseClient
    # _collection_delete is inherited from BaseClient

    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() use base class implementation

    # _collection_hybrid_search is inherited from BaseClient

    # -------------------- Collection Info --------------------

    # _collection_count is inherited from BaseClient - no override needed

    # ==================== Database Management ====================

    def create_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Create database (tenant parameter ignored for embedded mode)

        Args:
            name: database name
            tenant: ignored for embedded mode (no tenant concept)
        """
        return super().create_database(name=name, tenant=tenant)

    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Get database object (tenant parameter ignored for embedded mode)

        Args:
            name: database name
            tenant: ignored for embedded mode (no tenant concept)
        """
        return super().get_database(name=name, tenant=tenant)

    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Delete database (tenant parameter ignored for embedded mode)

        Args:
            name: database name
            tenant: ignored for embedded mode (no tenant concept)
        """
        return super().delete_database(name=name, tenant=tenant)

    def list_databases(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tenant: str = DEFAULT_TENANT,
    ) -> Sequence[Database]:
        """
        List all databases (tenant parameter ignored for embedded mode)

        Args:
            limit: maximum number of results to return
            offset: number of results to skip
            tenant: ignored for embedded mode (no tenant concept)
        """
        return super().list_databases(limit=limit, offset=offset, tenant=tenant)

    def _namespace_prewarm(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        **kwargs,
    ) -> None:
        """Prewarm the namespace logical table to reduce first-query latency."""
        raise ValueError("prewarm is only supported in shared-storage remote deployment")

    def __repr__(self):
        """Return the developer-readable representation."""
        status = "connected" if self.is_connected() else "disconnected"
        return f"<SeekdbEmbeddedClient path={self.path} database={self.database} status={status}>"
