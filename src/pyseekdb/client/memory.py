"""
Explicit memory release for embedded seekdb.

Implements the client-level memory release requested in pyseekdb issue #288:
after a batch of committed writes, an embedded application can ask the engine
to release the memory those writes still occupy in its MemStore, without
closing and reopening the database.

Engine integration (seekdb embedded, OceanBase-derived engine):

- ``ALTER SYSTEM MINOR FREEZE`` freezes every currently active MemTable
  instance-wide; data tables and their associated index MemTables share the
  freeze boundary. Writes committed before the freeze SCN belong to the
  frozen set; subsequent concurrent writes continue into newly created
  active MemTables. The command never commits an application transaction.
- ``oceanbase.GV$OB_MEMSTORE_INFO`` lists the engine's MemTables. A frozen
  MemTable disappears from this view only after it has been written to disk
  and released, which is how completion is detected.
- ``oceanbase.GV$OB_MEMSTORE`` provides the aggregate memory accounting used
  for the before/after measurements.

This releases memory held by recent writes in the engine. It is not a
process-RSS guarantee: allocators legitimately retain freed pages in their
caches, so RSS may stay higher than the released amount.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .sql_utils import is_query_sql

logger = logging.getLogger(__name__)

# SQL executed against the embedded engine. The oceanbase-qualified names make
# the virtual tables visible regardless of the session's current database.
_AGGREGATE_SQL = "SELECT FREEZE_CNT, MEMSTORE_USED FROM oceanbase.GV$OB_MEMSTORE"
_MEMTABLE_INFO_SQL = "SELECT TABLET_ID, IS_ACTIVE, START_SCN, END_SCN FROM oceanbase.GV$OB_MEMSTORE_INFO"
_MINOR_FREEZE_SQL = "ALTER SYSTEM MINOR FREEZE"

# The engine applies a submitted freeze within roughly one scheduler tick;
# polling faster than this only adds query latency.
_POLL_INTERVAL_SECONDS = 0.2

SqlRunner = Callable[[str], list[Any]]


class MemoryReleaseTimeoutError(RuntimeError):
    """Raised when a memory release did not complete within the timeout.

    A timeout never undoes committed writes and never cancels the engine's
    background release; the operation can simply be called again.
    """


@dataclass
class MemoryReleaseResult:
    """Outcome of a completed memory release.

    Attributes:
        memstore_used_before_bytes: Engine memory accounting before the
            release was submitted.
        memstore_used_after_bytes: Engine memory accounting after the
            release completed. On an idle instance this is lower than
            ``before``; writes committed during the release can offset it.
    """

    memstore_used_before_bytes: int | None
    memstore_used_after_bytes: int | None

    @property
    def reclaimed_bytes(self) -> int | None:
        """Bytes released: ``before - after``, or None if not measurable."""
        if self.memstore_used_before_bytes is None or self.memstore_used_after_bytes is None:
            return None
        return self.memstore_used_before_bytes - self.memstore_used_after_bytes

    def __repr__(self) -> str:
        return (
            f"MemoryReleaseResult(memstore_used_before_bytes={self.memstore_used_before_bytes}, "
            f"memstore_used_after_bytes={self.memstore_used_after_bytes}, "
            f"reclaimed_bytes={self.reclaimed_bytes})"
        )


def validate_release_memory_options(timeout: float) -> None:
    """Validate release_memory() arguments and raise on unsupported values."""
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Normalize a row from either a dict cursor or a positional tuple cursor."""
    if isinstance(row, dict):
        return {str(k).upper(): v for k, v in row.items()}
    if isinstance(row, (tuple, list)) and len(row) == 1:
        return {"VALUE": row[0]}
    return {"VALUE": tuple(row)}


def _col(row: dict[str, Any], name: str, index: int) -> Any:
    """Read one column by name, falling back to its known position."""
    if name in row:
        return row[name]
    values = row.get("VALUE")
    if isinstance(values, tuple):
        return values[index]
    return values


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


class MemoryReleaseTracker:
    """State machine for one memory release, decoupled from connection details.

    SQL execution is injected through ``run_sql`` so the embedded client can
    run freeze submission on a dedicated session — a memory release must never
    implicitly commit an application transaction on the client's main session
    — while read-only completion polls can reuse the main session.

    Completion waits for *every* MemTable that is frozen at poll time, not
    only the ones from this call's own freeze: MemTables left over from an
    earlier call still hold writes committed before this release, so this
    call also waits for them. Late automatic freezes can only widen the
    awaited set, which is the safe direction.
    """

    def __init__(self, run_sql: SqlRunner, start_time: float | None = None) -> None:
        self.run_sql = run_sql
        self._start_time = start_time if start_time is not None else time.monotonic()
        self._freeze_cnt_before: int | None = None
        self._used_before: int | None = None
        self._max_end_scn_before: int = 0

    # ==================== submission ====================

    def capture_boundary(self) -> None:
        """Snapshot memory accounting before submitting the freeze."""
        freeze_cnt, used, frozen = self._read_state()
        self._freeze_cnt_before = freeze_cnt
        self._used_before = used
        self._max_end_scn_before = max((row[2] for row in frozen), default=0)

    def submit(self) -> None:
        """Submit the instance-wide minor freeze that establishes the boundary."""
        self.run_sql(_MINOR_FREEZE_SQL)

    def confirm_freeze(self, deadline: float, timeout_seconds: float) -> None:
        """Wait until the engine confirms a new freeze."""
        while True:
            _freeze_cnt, _used, frozen = self._read_state()
            confirmed = (
                self._freeze_cnt_before is not None
                and _freeze_cnt is not None
                and _freeze_cnt > self._freeze_cnt_before
            ) or any(row[2] > self._max_end_scn_before for row in frozen)
            if confirmed:
                return
            if time.monotonic() >= deadline:
                raise self.timeout_error(timeout_seconds)
            time.sleep(_POLL_INTERVAL_SECONDS)

    # ==================== completion ====================

    def released(self) -> bool:
        """Read the MemTable view once and return whether everything is released.

        A frozen MemTable disappears from the view only once it has been
        written to disk and released.
        """
        _freeze_cnt, _used, frozen = self._read_state()
        return not frozen

    def build_result(self) -> MemoryReleaseResult:
        """Assemble the final result from a fresh read at completion time."""
        _freeze_cnt, used, _frozen = self._read_state()
        return MemoryReleaseResult(
            memstore_used_before_bytes=self._used_before,
            memstore_used_after_bytes=used,
        )

    def timeout_error(self, timeout_seconds: float) -> MemoryReleaseTimeoutError:
        """Build the timeout error reporting what is still held."""
        _freeze_cnt, used_after, frozen = self._read_state()
        if frozen:
            waiting_for = (
                f"{len(frozen)} in-memory table(s) are still held; live references "
                "(open snapshots or result handles) can delay their release"
            )
        else:
            waiting_for = "the engine has not confirmed the requested release yet"
        return MemoryReleaseTimeoutError(
            f"release_memory timed out after {timeout_seconds}s: {waiting_for}; "
            f"engine memory accounting: {self._used_before} -> {used_after} bytes. Committed writes are "
            "unaffected, background release continues, and release_memory() can be called again to keep waiting."
        )

    # ==================== internals ====================

    def _read_state(self) -> tuple[int | None, int | None, list[tuple[int, int, int]]]:
        """Read the aggregate accounting and the frozen MemTable list."""
        freeze_cnt: int | None = None
        memstore_used: int | None = None
        for row in self.run_sql(_AGGREGATE_SQL):
            normalized = _row_to_dict(row)
            row_cnt = _to_int(_col(normalized, "FREEZE_CNT", 0))
            row_used = _to_int(_col(normalized, "MEMSTORE_USED", 1))
            if row_cnt is not None:
                freeze_cnt = row_cnt if freeze_cnt is None else max(freeze_cnt, row_cnt)
            if row_used is not None:
                memstore_used = row_used if memstore_used is None else memstore_used + row_used

        frozen: list[tuple[int, int, int]] = []
        for row in self.run_sql(_MEMTABLE_INFO_SQL):
            normalized = _row_to_dict(row)
            tablet_id = _to_int(_col(normalized, "TABLET_ID", 0))
            is_active = str(_col(normalized, "IS_ACTIVE", 1)).upper()
            start_scn = _to_int(_col(normalized, "START_SCN", 2))
            end_scn = _to_int(_col(normalized, "END_SCN", 3))
            if tablet_id is None or start_scn is None or end_scn is None:
                continue
            if is_active != "YES":
                frozen.append((tablet_id, start_scn, end_scn))

        return freeze_cnt, memstore_used, frozen


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
