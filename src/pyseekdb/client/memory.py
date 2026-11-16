"""
Explicit memory release request for embedded seekdb.

Implements the client-level memory release requested in pyseekdb issue #288:
after a batch of committed writes, an embedded application can ask the engine
to release the memory those writes still occupy, without closing and reopening
the database.

The contract is deliberately minimal — a request, not a measurement:

- The call submits the request and returns immediately; the engine completes
  the release in the background. There is no result and nothing to wait for.
- Today the request releases the memory held by recent committed writes
  (the engine's in-memory tables). Future versions may extend the request to
  other reclaimable engine memory (block caches and similar pools), so
  callers must not rely on any measured reclaim amount.
- This is not a process-RSS guarantee: allocators legitimately retain freed
  pages in their caches, so RSS may stay higher than the released amount.

Engine integration (seekdb embedded, OceanBase-derived engine):

- ``ALTER SYSTEM MINOR FREEZE`` freezes every currently active MemTable
  instance-wide; data tables and their associated index MemTables share the
  freeze boundary. Writes committed before the freeze SCN belong to the
  frozen set; subsequent concurrent writes continue into newly created
  active MemTables. The command never commits an application transaction.
"""

from collections.abc import Callable
from typing import Any

from .sql_utils import is_query_sql

# SQL executed against the embedded engine.
_MINOR_FREEZE_SQL = "ALTER SYSTEM MINOR FREEZE"

SqlRunner = Callable[[str], list[Any]]


def submit_memory_release(run_sql: SqlRunner) -> None:
    """Submit the engine memory release request.

    Raises:
        ValueError: The engine rejected the release request.
    """
    try:
        run_sql(_MINOR_FREEZE_SQL)
    except Exception as exc:
        raise ValueError(f"release_memory failed to submit the engine memory release: {exc}") from exc


def make_connection_sql_runner(connection: Any, *, use_context_manager: bool) -> SqlRunner:
    """Build a SqlRunner over an arbitrary DB-API connection.

    Non-query statements (the minor-freeze command) never fetch results, which
    keeps the legacy native cursor happy.
    """

    def run_sql(sql: str) -> list[Any]:
        if use_context_manager:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                if not is_query_sql(sql):
                    return []
                return list(cursor.fetchall())
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            if not is_query_sql(sql):
                return []
            return list(cursor.fetchall())
        finally:
            cursor.close()

    return run_sql
