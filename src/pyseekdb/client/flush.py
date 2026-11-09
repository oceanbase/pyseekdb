"""
Explicit flush API for embedded seekdb.

Implements the client-level flush operation requested in pyseekdb issue #288:
after a batch of committed writes, an embedded application can flush the
MemStore and reclaim eligible memory without closing and reopening the
database.

Engine integration (seekdb embedded, OceanBase-derived engine):

- ``ALTER SYSTEM MINOR FREEZE`` freezes every currently active MemTable
  instance-wide; data-table and associated index MemTables share the freeze
  boundary. Writes committed before the freeze SCN belong to the flushed set;
  subsequent concurrent writes continue into newly created active MemTables.
  The command never commits an application transaction.
- ``oceanbase.GV$OB_MEMSTORE_INFO`` lists the engine's MemTables. A frozen
  MemTable disappears from this view only after it has been dumped to an
  SSTable and released, which is how reclamation completion is detected.
- ``oceanbase.GV$OB_TABLET_COMPACTION_HISTORY`` records one ``MINI_MERGE``
  entry per finished MemTable dump; it evidences "flushed to SSTable"
  completion independently of release.
- ``oceanbase.GV$OB_MEMSTORE`` provides aggregate MemStore accounting
  (``FREEZE_CNT``, ``MEMSTORE_USED``) for before/after measurements.

Reclaiming MemStore memory is a logical/held-memory guarantee: the engine's
MemStore accounting drops once MemTables are released, but process RSS may
stay higher because allocators legitimately retain cached pages.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .sql_utils import is_query_sql

logger = logging.getLogger(__name__)

# SQL executed against the embedded engine. The oceanbase-qualified names make
# the virtual tables visible regardless of the session's current database.
_AGGREGATE_SQL = "SELECT FREEZE_CNT, MEMSTORE_USED FROM oceanbase.GV$OB_MEMSTORE"
_MEMTABLE_INFO_SQL = "SELECT TABLET_ID, IS_ACTIVE, START_SCN, END_SCN FROM oceanbase.GV$OB_MEMSTORE_INFO"
_COMPACTION_HISTORY_SQL = (
    "SELECT TABLET_ID, COMPACTION_SCN FROM oceanbase.GV$OB_TABLET_COMPACTION_HISTORY WHERE TYPE = 'MINI_MERGE'"
)
_MINOR_FREEZE_SQL = "ALTER SYSTEM MINOR FREEZE"

# The engine applies a submitted freeze within roughly one scheduler tick;
# polling faster than this only adds query latency.
_POLL_INTERVAL_SECONDS = 0.2

SqlRunner = Callable[[str], list[Any]]


class FlushStage(str, Enum):
    """Stages of an explicit flush, in completion order."""

    # Freeze submitted; the boundary MemTables are not all dumped to SSTables yet.
    SUBMITTED = "submitted"
    # Every boundary MemTable has been dumped to an SSTable.
    FLUSHED = "flushed"
    # Every boundary MemTable has additionally been released, reclaiming its
    # MemStore memory.
    RECLAIMED = "reclaimed"


class FlushError(RuntimeError):
    """Base class for explicit-flush failures."""


class FlushUnsupportedError(FlushError):
    """Raised when the installed engine/binding cannot honor the flush contract."""


class FlushTimeoutError(FlushError):
    """Raised when a flush did not reach its target stage within the timeout.

    A timeout never undoes committed writes and never cancels background
    flushing; the engine keeps flushing after this error is raised, and a
    later ``flush()`` call can be used to keep waiting.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: FlushStage,
        timeout_seconds: float,
        elapsed_seconds: float,
        frozen_memtable_count: int,
        pending_memtable_count: int,
        memstore_used_before_bytes: int | None,
        memstore_used_after_bytes: int | None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.frozen_memtable_count = frozen_memtable_count
        self.pending_memtable_count = pending_memtable_count
        self.memstore_used_before_bytes = memstore_used_before_bytes
        self.memstore_used_after_bytes = memstore_used_after_bytes


@dataclass
class FlushResult:
    """Outcome of a completed explicit flush.

    Attributes:
        stage: Final stage reached: ``FLUSHED`` when ``reclaim_memory`` was
            False, ``RECLAIMED`` when it was True.
        boundary_freeze_scn: Freeze SCN the engine assigned to this flush; all
            writes committed before it belong to the flushed set.
        frozen_memtable_count: MemTables frozen by this flush boundary (an
            instance-level flush covers data tables and their index MemTables).
        released_memtable_count: Boundary MemTables already released (their
            MemStore memory reclaimed) when the flush returned.
        memstore_used_before_bytes: Aggregate MemStore usage captured before
            the freeze was submitted.
        memstore_used_after_bytes: Aggregate MemStore usage captured after the
            target stage completed.
        memstore_reclaimed_bytes: ``before - after``. Positive on an idle
            instance; concurrent writes during the flush can reduce it or make
            it negative.
        elapsed_seconds: Wall-clock time the flush took.
    """

    stage: FlushStage
    boundary_freeze_scn: int | None
    frozen_memtable_count: int
    released_memtable_count: int
    memstore_used_before_bytes: int | None
    memstore_used_after_bytes: int | None
    memstore_reclaimed_bytes: int | None
    elapsed_seconds: float

    @property
    def completed(self) -> bool:
        """Return True when the requested stage was reached."""
        return self.stage in (FlushStage.FLUSHED, FlushStage.RECLAIMED)

    def __repr__(self) -> str:
        return (
            f"FlushResult(stage={self.stage.value}, boundary_freeze_scn={self.boundary_freeze_scn}, "
            f"frozen_memtable_count={self.frozen_memtable_count}, "
            f"released_memtable_count={self.released_memtable_count}, "
            f"memstore_used_before_bytes={self.memstore_used_before_bytes}, "
            f"memstore_used_after_bytes={self.memstore_used_after_bytes}, "
            f"memstore_reclaimed_bytes={self.memstore_reclaimed_bytes}, "
            f"elapsed_seconds={self.elapsed_seconds})"
        )


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


@dataclass
class _MemtableState:
    """One engine-visible snapshot of the flush observability views."""

    freeze_cnt: int | None = None
    memstore_used: int | None = None
    # (tablet_id, start_scn, end_scn) for every frozen (IS_ACTIVE='NO') MemTable.
    frozen: list[tuple[int, int, int]] = field(default_factory=list)
    # (tablet_id, compaction_scn) for every finished MemTable dump.
    dumped: set[tuple[int, int]] = field(default_factory=set)
    history_available: bool = True

    @property
    def frozen_count(self) -> int:
        return len(self.frozen)


class FlushTracker:
    """State machine for one explicit flush, decoupled from connection details.

    The tracker only orchestrates; SQL execution is injected through
    ``run_sql`` (swappable) so the embedded client can run freeze submission
    on a dedicated session — an explicit flush must never implicitly commit an
    application transaction on the client's main session — while read-only
    completion polls can reuse the main session.

    The completion set is *every* MemTable frozen at poll time, not only the
    ones from this call's own freeze SCN: MemTables left over from an earlier
    flush still hold writes committed before this flush's boundary, so a new
    flush also waits for them. Late automatic freezes can only widen the set,
    which is the safe direction.
    """

    def __init__(self, run_sql: SqlRunner, *, reclaim_memory: bool, start_time: float | None = None) -> None:
        self.run_sql = run_sql
        self.reclaim_memory = reclaim_memory
        self._start_time = start_time if start_time is not None else time.monotonic()
        self._freeze_cnt_before: int | None = None
        self._used_before: int | None = None
        self._max_end_scn_before: int = 0
        self.boundary_freeze_scn: int | None = None
        self._boundary_frozen_count: int = 0
        self._last_state: _MemtableState | None = None

    @property
    def target_stage(self) -> FlushStage:
        return FlushStage.RECLAIMED if self.reclaim_memory else FlushStage.FLUSHED

    # ==================== boundary + submission ====================

    def capture_boundary(self) -> None:
        """Snapshot MemStore accounting before submitting the freeze."""
        state = self._read_state()
        self._freeze_cnt_before = state.freeze_cnt
        self._used_before = state.memstore_used
        self._max_end_scn_before = max((row[2] for row in state.frozen), default=0)
        self._last_state = state

    def submit(self) -> None:
        """Submit the instance-wide minor freeze that establishes the boundary."""
        self.run_sql(_MINOR_FREEZE_SQL)

    def confirm_freeze(self, deadline: float, timeout_seconds: float) -> None:
        """Wait until the engine confirms a new freeze and record its SCN."""
        while True:
            state = self._read_state()
            self._last_state = state
            confirmed = (
                self._freeze_cnt_before is not None
                and state.freeze_cnt is not None
                and state.freeze_cnt > self._freeze_cnt_before
            ) or any(row[2] > self._max_end_scn_before for row in state.frozen)
            if confirmed:
                break
            if time.monotonic() >= deadline:
                raise self.timeout_error(FlushStage.SUBMITTED, timeout_seconds)
            time.sleep(_POLL_INTERVAL_SECONDS)

        # The freeze SCN is the boundary every frozen MemTable now shares. If
        # the flush was so small that frozen rows were already released before
        # the first poll, fall back to a SCN past every previously seen one so
        # the completion checks start from a clean slate.
        self.boundary_freeze_scn = max((row[2] for row in state.frozen), default=self._max_end_scn_before + 1)
        self._boundary_frozen_count = state.frozen_count

    # ==================== completion polling ====================

    def poll(self) -> FlushStage:
        """Read the observability views once and return the stage reached."""
        state = self._read_state()
        self._last_state = state
        if not self._dump_completed(state):
            return FlushStage.SUBMITTED
        if self.reclaim_memory and state.frozen:
            return FlushStage.FLUSHED
        return FlushStage.RECLAIMED if self.reclaim_memory else FlushStage.FLUSHED

    def build_result(self, stage: FlushStage) -> FlushResult:
        """Assemble the final result from a fresh state read at completion time.

        The completion-detecting poll may have read the MemStore aggregate just
        before the engine released memory, so the after-measurement is taken
        again here, at the moment the flush returns.
        """
        state = self._read_state()
        self._last_state = state
        released = self._boundary_frozen_count - state.frozen_count
        used_after = state.memstore_used
        reclaimed = self._used_before - used_after if self._used_before is not None and used_after is not None else None
        return FlushResult(
            stage=stage,
            boundary_freeze_scn=self.boundary_freeze_scn,
            frozen_memtable_count=self._boundary_frozen_count,
            released_memtable_count=max(released, 0),
            memstore_used_before_bytes=self._used_before,
            memstore_used_after_bytes=used_after,
            memstore_reclaimed_bytes=reclaimed,
            elapsed_seconds=time.monotonic() - self._start_time,
        )

    def timeout_error(self, stage: FlushStage, timeout_seconds: float) -> FlushTimeoutError:
        """Build the timeout error reporting the incomplete stage."""
        state = self._last_state or _MemtableState()
        elapsed = time.monotonic() - self._start_time
        if self.boundary_freeze_scn is None:
            pending = state.frozen_count
            waiting_for = "the engine to confirm the freeze"
        elif not self._dump_completed(state):
            pending = len(self._undumped_frozen(state))
            waiting_for = (
                f"{pending} boundary MemTable(s) to be flushed to SSTables"
                if state.history_available
                else f"{pending} boundary MemTable(s) to be flushed and released"
            )
        else:
            pending = state.frozen_count
            waiting_for = (
                f"{pending} flushed MemTable(s) to be released; live references "
                "(open snapshots or result handles) can delay release"
            )
        message = (
            f"flush timed out after {timeout_seconds}s at stage '{stage.value}', waiting for {waiting_for}; "
            f"target stage was '{self.target_stage.value}'. Committed writes are not undone, background "
            "flushing continues, and flush() can be called again to keep waiting."
        )
        return FlushTimeoutError(
            message,
            stage=stage,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=elapsed,
            frozen_memtable_count=self._boundary_frozen_count,
            pending_memtable_count=pending,
            memstore_used_before_bytes=self._used_before,
            memstore_used_after_bytes=state.memstore_used,
        )

    # ==================== internals ====================

    @staticmethod
    def _undumped_frozen(state: _MemtableState) -> list[tuple[int, int, int]]:
        """Frozen MemTables whose SSTable dump has not finished yet.

        Without compaction history, release is the only observable completion
        signal; a released MemTable was necessarily dumped first.
        """
        if not state.history_available:
            return list(state.frozen)
        return [row for row in state.frozen if (row[0], row[2]) not in state.dumped]

    def _dump_completed(self, state: _MemtableState) -> bool:
        return not self._undumped_frozen(state)

    def _read_state(self) -> _MemtableState:
        state = _MemtableState()

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
        state.freeze_cnt = freeze_cnt
        state.memstore_used = memstore_used

        for row in self.run_sql(_MEMTABLE_INFO_SQL):
            normalized = _row_to_dict(row)
            tablet_id = _to_int(_col(normalized, "TABLET_ID", 0))
            is_active = str(_col(normalized, "IS_ACTIVE", 1)).upper()
            start_scn = _to_int(_col(normalized, "START_SCN", 2))
            end_scn = _to_int(_col(normalized, "END_SCN", 3))
            if tablet_id is None or start_scn is None or end_scn is None:
                continue
            if is_active != "YES":
                state.frozen.append((tablet_id, start_scn, end_scn))

        try:
            for row in self.run_sql(_COMPACTION_HISTORY_SQL):
                normalized = _row_to_dict(row)
                tablet_id = _to_int(_col(normalized, "TABLET_ID", 0))
                compaction_scn = _to_int(_col(normalized, "COMPACTION_SCN", 1))
                if tablet_id is not None and compaction_scn is not None:
                    state.dumped.add((tablet_id, compaction_scn))
        except Exception as exc:
            # Compaction history is an optional flush-evidence view. Without it,
            # reclamation completion (MemTable release) is still observable.
            state.history_available = False
            logger.debug("flush: compaction history unavailable, falling back to release-only completion: %s", exc)

        return state


class FlushHandle:
    """Completion handle for a flush submitted with ``wait=False``.

    The freeze has already been submitted and confirmed when this handle is
    returned. Use :meth:`wait` to block for completion or :meth:`is_completed`
    to poll. The handle runs no background work by itself: every check
    re-queries the engine on demand, so submission is never silently reported
    as completion.
    """

    def __init__(
        self, wait_fn: Callable[[FlushTracker, float], FlushResult], tracker: FlushTracker, *, timeout_seconds: float
    ) -> None:
        self._wait_fn = wait_fn
        self._tracker = tracker
        self._timeout_seconds = timeout_seconds

    @property
    def target_stage(self) -> FlushStage:
        """Stage this flush is waiting to reach."""
        return self._tracker.target_stage

    @property
    def boundary_freeze_scn(self) -> int | None:
        """Freeze SCN of this flush's boundary, or None if not yet confirmed."""
        return self._tracker.boundary_freeze_scn

    def is_completed(self) -> bool:
        """Check once whether the flush reached its target stage."""
        return self._tracker.poll() == self._tracker.target_stage

    def wait(self, timeout: float | None = None) -> FlushResult:
        """Block until the flush reaches its target stage.

        Args:
            timeout: Seconds to wait; defaults to the timeout given to
                ``flush()``.

        Returns:
            FlushResult for the completed flush.

        Raises:
            FlushTimeoutError: The target stage was not reached in time.
            ValueError: If ``timeout`` is not a positive number.
        """
        effective_timeout = self._timeout_seconds if timeout is None else timeout
        if (
            not isinstance(effective_timeout, (int, float))
            or isinstance(effective_timeout, bool)
            or effective_timeout <= 0
        ):
            raise ValueError(f"timeout must be a positive number of seconds, got {effective_timeout!r}")
        return self._wait_fn(self._tracker, effective_timeout)

    def __repr__(self) -> str:
        scn = self.boundary_freeze_scn
        boundary = f"freeze_scn={scn}" if scn is not None else "freeze not confirmed yet"
        return f"<FlushHandle {boundary}, target_stage={self.target_stage.value}>"


def validate_flush_options(wait: bool, reclaim_memory: bool, timeout: float) -> None:
    """Validate flush() arguments and raise on unsupported values."""
    if not isinstance(wait, bool):
        raise TypeError(f"wait must be a bool, got {type(wait).__name__}")
    if not isinstance(reclaim_memory, bool):
        raise TypeError(f"reclaim_memory must be a bool, got {type(reclaim_memory).__name__}")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")


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
