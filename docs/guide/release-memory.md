# 7. Release Memory (Embedded Mode)

After a batch of inserts or updates, the embedded engine keeps those writes in memory until they are written to disk and their in-memory tables are released. Applications know when a write-heavy phase has finished; `release_memory()` turns that knowledge into an explicit, instance-wide release that hands memory back **without closing and reopening the database**.

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

# Hand the memory those writes occupy back to the engine.
result = client.release_memory(timeout=30)
print(result.reclaimed_bytes)
```

`release_memory()` is an **instance-level, embedded-mode** operation. Server-mode clients raise a clear `ValueError`; server-mode support is evaluated separately ([issue #288](https://github.com/oceanbase/pyseekdb/issues/288)).

## 7.1 API

```python
client.release_memory(timeout: float = 30.0) -> MemoryReleaseResult
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `timeout` | `30.0` | Seconds to wait. Must be a positive number. |

The call blocks until the release completes or the timeout expires. There is one guarantee and no knobs: memory held by writes committed before the call is released.

`MemoryReleaseResult` fields:

| Field | Meaning |
| --- | --- |
| `memstore_used_before_bytes` | Engine memory accounting before the release was submitted. |
| `memstore_used_after_bytes` | Engine memory accounting after the release completed. On an idle instance this is lower than `before`; writes committed during the release can offset it. |
| `reclaimed_bytes` | Derived: `before - after`. |

## 7.2 Semantics

- **What it does.** The release asks the engine to finish writing the tables that hold your recent committed writes to disk and then drop them from memory — data tables and their associated index tables, instance-wide. Data stays fully queryable the whole time and is not modified.
- **Boundary.** Writes committed after the call starts continue into new in-memory tables and are not waited for. Concurrent writes are safe: they land in new tables and are released by a later `release_memory()` (or automatically).
- **No implicit commits.** The underlying engine command runs on a dedicated session and never commits an application transaction. An open transaction stays open; its uncommitted rows are not made visible by a release.
- **Timeouts do not undo work.** On timeout, `MemoryReleaseTimeoutError` reports what is still held, the last-known memory accounting, and that committed writes are unaffected. Background release continues; calling `release_memory()` again keeps waiting.
- **Automatic release still works.** The engine's own threshold-based release covers applications that never call `release_memory()`; this call is for the end of a known write-heavy phase.

## 7.3 Errors

| Error | Raised when |
| --- | --- |
| `MemoryReleaseTimeoutError` | The release did not complete within `timeout`. The message reports what is still held; call again to keep waiting. |
| `ValueError` | The connected mode does not support this operation (remote server mode), the installed engine/binding does not expose the required engine views, the engine rejected the release command, or `timeout` is not a positive number. |

## 7.4 What it releases — and what it does not

`release_memory()` releases **memory held by recent writes in the engine**. It is not a process-RSS guarantee: allocators legitimately retain freed pages in their caches for reuse, and other engine components (block caches, vector indexes) stay resident. Process RSS may therefore stay higher than the released amount. Use the returned accounting numbers — not RSS — to reason about what the engine holds.

## 7.5 Cost of frequent small releases

`release_memory()` is intended **once per batch or business phase**, not as an automatic memory-pressure handler (the engine's own thresholds already cover pressure-driven release). Each call makes the engine write new on-disk tables; releasing very small batches frequently produces many small tables and increases read amplification. Prefer batching writes and releasing once per phase.
