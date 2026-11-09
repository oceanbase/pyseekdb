# 7. Explicit Flush (Embedded Mode)

After a batch of inserts or updates, the embedded engine keeps the written data in MemStore memory until it is flushed to SSTables and the MemTables are released. Applications know when a write-heavy phase has finished; `flush()` turns that knowledge into an explicit, instance-wide flush that reclaims eligible MemStore memory **without closing and reopening the database** — following the explicit flush APIs of RocksDB and Pebble.

```python
import pyseekdb

client = pyseekdb.Client(path="./seekdb.db", database="demo")
collection = client.get_or_create_collection(
    name="docs",
    configuration=pyseekdb.HNSWConfiguration(dimension=4),
    embedding_function=None,
)

# ... a write-heavy batch phase ...
collection.add(ids=["d1", "d2"], embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

# Flush committed writes and reclaim eligible MemStore memory.
result = client.flush(wait=True, reclaim_memory=True, timeout=30)
```

`flush()` is an **instance-level, embedded-mode** operation. Server-mode clients raise a clear `ValueError`; database/collection scope and server-mode support are evaluated separately ([issue #288](https://github.com/oceanbase/pyseekdb/issues/288)).

## 7.1 API

```python
client.flush(wait=True, reclaim_memory=True, timeout=30.0)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `wait` | `True` | Block until the target stage completes or `timeout` expires. With `False`, submit the flush and return a `FlushHandle` immediately. |
| `reclaim_memory` | `True` | Wait until the boundary MemTables are *released* (their MemStore memory reclaimed) and report before/after usage. With `False`, stop once every boundary MemTable is flushed to SSTables. |
| `timeout` | `30.0` | Seconds to wait. Applies to freeze confirmation and stage completion. Must be a positive number. |

Returns a `FlushResult` (blocking call) or a `FlushHandle` (`wait=False`).

`FlushResult` fields:

| Field | Meaning |
| --- | --- |
| `stage` | Final stage: `FlushStage.FLUSHED` (MemTables dumped to SSTables) or `FlushStage.RECLAIMED` (additionally released from memory). |
| `boundary_freeze_scn` | The freeze SCN the engine assigned; all writes committed before it are covered by this flush. |
| `frozen_memtable_count` | MemTables frozen at this flush's boundary — data tables and their associated index MemTables, instance-wide. |
| `released_memtable_count` | Boundary MemTables already released when the flush returned. |
| `memstore_used_before_bytes` / `memstore_used_after_bytes` | Aggregate MemStore usage before submission and after completion. |
| `memstore_reclaimed_bytes` | `before - after`. On an idle instance this is positive; writes committed *during* the flush can reduce it or make it negative. |
| `elapsed_seconds` | Wall-clock time the flush took. |

## 7.2 Flush semantics

- **Boundary.** The flush submits an instance-wide minor freeze. Every MemTable that was active at submission — including associated index MemTables — is frozen and flushed; the engine assigns the boundary a freeze SCN. Writes committed *after* the boundary continue into newly created active MemTables and are not waited for (late automatic freezes can widen the awaited set, never narrow it).
- **No implicit commits.** The freeze command runs on a dedicated session and never commits an application transaction. An open transaction stays open; its uncommitted rows are not made visible by a flush.
- **Two completion stages.** `FLUSHED` means every boundary MemTable has been dumped to an SSTable (evidenced by the engine's compaction history). `RECLAIMED` additionally means those MemTables have been released and their MemStore accounting has dropped. The engine's threshold-based automatic freeze continues to work for applications that never call `flush()`.
- **Timeouts do not undo work.** On timeout, `FlushTimeoutError` reports the incomplete stage (`submitted` / `flushed`), the pending MemTable count, and the last-known MemStore usage. Committed writes are never rolled back and background flushing continues; call `flush()` again to keep waiting.
- **Concurrent writes are safe.** A flush while other sessions are writing completes for its own boundary; the concurrent writes land in new active MemTables and are flushed by a later flush (or automatically).

## 7.3 Asynchronous flush

`wait=False` returns a `FlushHandle` after the freeze is submitted and confirmed. The handle runs no background thread: every check re-queries the engine on demand, so submission is never silently treated as completion.

```python
handle = client.flush(wait=False, reclaim_memory=True, timeout=30)
# ... continue application work ...
if handle.is_completed():
    ...

result = handle.wait(timeout=30)   # blocks until RECLAIMED; raises FlushTimeoutError on timeout
```

## 7.4 Errors

| Error | Raised when |
| --- | --- |
| `FlushTimeoutError` | The target stage was not reached within `timeout`. Carries `stage`, `pending_memtable_count`, `frozen_memtable_count`, and before/after MemStore usage. |
| `FlushUnsupportedError` | The installed engine/binding does not expose the MemStore observability views required by the flush completion contract (for example user-scoped bindings that hide `oceanbase.GV$OB_MEMSTORE`). |
| `FlushError` | The engine rejected the freeze command itself. |
| `TypeError` / `ValueError` | Unsupported argument values (`wait`/`reclaim_memory` must be bools; `timeout` must be a positive number). Non-embedded clients raise `ValueError` from `flush()`. |

## 7.5 Memory reclamation and RSS

Reclaiming MemStore memory is a **logical/held-memory guarantee**: after a reclaiming flush, the engine no longer holds the boundary MemTables and the engine's MemStore accounting (`GV$OB_MEMSTORE`) drops accordingly. Process RSS may remain higher than that accounting suggests, because allocators legitimately retain freed pages in their caches for reuse, and other components (block caches, vector indexes) stay resident. Do not use `flush()` to assert on process RSS.

## 7.6 Cost of frequent small flushes

`flush()` is intended **once per batch or business phase**, not as an automatic memory-pressure handler (the engine's `freeze_trigger_percentage` already covers pressure-driven freezing). Each flush creates a new SSTable per tablet; flushing very small batches frequently produces many small SSTables, increases read amplification, and makes each subsequent read merge more SSTable levels. Prefer batching writes and flushing once per phase.
