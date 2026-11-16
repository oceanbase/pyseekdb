# 7. Release Memory (Embedded Mode)

After a batch of inserts or updates, the embedded engine keeps those writes in memory until they are written to disk and their in-memory tables are released. Applications know when a write-heavy phase has finished; `release_memory()` turns that knowledge into an explicit request for the engine to hand that memory back — **without closing and reopening the database**.

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

# Ask the engine to release the memory those writes occupy.
client.release_memory()
```

`release_memory()` is an **instance-level, embedded-mode** operation. Other connection modes (remote server, OceanBase) log a warning and do nothing, so applications can call it unconditionally after a write phase; server-mode support is evaluated separately ([issue #288](https://github.com/oceanbase/pyseekdb/issues/288)).

## 7.1 API

```python
client.release_memory() -> None
```

The call just submits the request and returns immediately. The engine completes the release in the background; there is nothing to wait for and no result to inspect. If the engine rejects the request, a `ValueError` is raised. In non-embedded modes the call is a logged no-op — it returns `None` without raising, so the same application code works in every connection mode.

## 7.2 Semantics

- **What it requests.** Release of reclaimable engine memory. Today that is the memory held by your recent committed writes (the engine writes those tables to disk and drops them from memory, instance-wide, including associated index tables). Future versions may extend the request to other reclaimable engine memory (block caches and similar pools), so **do not rely on measured reclaim amounts**.
- **Fire-and-forget.** The call does not wait for the release to finish and does not report how much memory was released. If you need a later memory state, observe it separately; the release itself is complete when the engine finishes it in the background.
- **Boundary.** Writes committed after the call continue into new in-memory tables and are untouched by it. Concurrent writes are safe: they land in new tables and are released by a later `release_memory()` (or automatically).
- **No implicit commits.** The underlying engine command runs on a dedicated session and never commits an application transaction. An open transaction stays open; its uncommitted rows are not made visible by a release.
- **Automatic release still works.** The engine's own threshold-based release covers applications that never call `release_memory()`; this call is for the end of a known write-heavy phase.

## 7.3 Errors

| Error | Raised when |
| --- | --- |
| `ValueError` | The engine rejected the release request, or a dedicated engine session could not be opened for it (embedded mode). |

Non-embedded connection modes never raise: `release_memory()` logs a warning and does nothing, so callers do not need to branch on the connection mode.

## 7.4 What it releases — and what it does not

`release_memory()` requests release of **memory held by recent writes in the engine**. It is not a process-RSS guarantee: allocators legitimately retain freed pages in their caches for reuse, and other engine components (block caches, vector indexes) stay resident. Process RSS may therefore stay higher than the released amount, and callers should not rely on measured reclaim amounts.

## 7.5 Cost of frequent small releases

`release_memory()` is intended **once per batch or business phase**, not as an automatic memory-pressure handler (the engine's own thresholds already cover pressure-driven release). Each call makes the engine write new on-disk tables; requesting release for very small batches frequently produces many small tables and increases read amplification. Prefer batching writes and releasing once per phase.
