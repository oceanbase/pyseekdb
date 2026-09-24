"""Isolated, dependency-free harness; pyseekdb is imported only in the worker.

Synthetic deterministic documents (not the original SciFact benchmark). The
scan/freezer uses the same SQL shape as the Linux issue #1382 reproducer, without
a vector index. Public hybrid_search coverage lives in the separate first pass.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import importlib.util
import json
import platform
import re
import threading
import time
import traceback
from pathlib import Path

ROWS = 24515
WORKERS = 8
SECONDS = 75
INTERVAL = 0.120
WRITE_BATCH = 32
QUERIES = 300
PASSES = 4
TEXT = "the database retrieval benchmark"
TABLE = "`c$v2$16d82ab3afe311f186fe000000000000`"
SEARCH = (
    f"SELECT `_id`, MATCH(`document`) AGAINST('{TEXT}') AS relevance FROM {TABLE} "
    f"WHERE MATCH(`document`) AGAINST('{TEXT}') ORDER BY relevance DESC LIMIT 10"
)
TRACE = re.compile(r"\bY[A-Za-z0-9]+-[A-Za-z0-9]+-\d+-\d+\b")


def document(ordinal):
    """Every row matches the broad query while cohorts supply varied queries."""
    return (
        f"{TEXT} exercises full text postings during concurrent memtable release "
        f"cohort{ordinal % QUERIES:03d} document ordinal{ordinal:08d}"
    )


def error_details(error):
    """Retain wrapper/cause messages and only report explicit numeric codes."""
    chain, seen, codes = [], set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(f"{type(error).__name__}: {error}")
        if error.args and isinstance(error.args[0], int):
            codes.append(error.args[0])
        codes.extend(int(value) for value in re.findall(r"\bcode[=:]\s*(-?\d+)\b", str(error)))
        # Released bindings may expose a RuntimeError rather than numeric args,
        # e.g. "execute sql failed OB_ERR_UNEXPECTED(4016): %s".
        codes.extend(int(value) for value in re.findall(r"\bOB_[A-Z0-9_]+\((-?\d+)\)", str(error)))
        error = error.__cause__ or error.__context__
    text = "\n".join(chain)
    return {"exception_chain": chain, "error_codes": sorted(set(codes)), "trace_ids": sorted(set(TRACE.findall(text)))}


def capture_logs(root, evidence, label, traces=(), budget=128 * 1024 * 1024, output_limit=2 * 1024 * 1024):
    """Scan once under a byte budget; prioritize errors over routine context.

    Each priority buffer is bounded by output_limit (at most 4x that limit in
    memory). A full low-priority buffer must not stop the scan: the failing
    trace can occur later in a large, noisy rotated log.
    """
    if budget < 0 or output_limit <= 0:
        raise ValueError("invalid diagnostic byte budget")
    trace_patterns = [value.encode() for value in traces if value]
    error_patterns = [
        b"-4016",
        b"data/schema type does not match",
        b"failed to get next row from memtable scanner",
        b"ob_memtable_key.h",
    ]
    files = sorted(
        (p for p in root.rglob("seekdb.log*") if p.is_file() and not p.is_symlink()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    inventory = []
    priorities = ("trace-error", "trace", "error", "memtable-context")
    buffers = [bytearray() for _ in priorities]
    matched = [0] * len(priorities)
    capped = [False] * len(priorities)

    def append(priority, content):
        available = output_limit - len(buffers[priority])
        if len(content) > available:
            capped[priority] = True
        buffers[priority].extend(content[:available])

    for path in files:
        size = path.stat().st_size
        count = min(size, budget)
        inventory.append({"path": str(path), "size": size, "read_bytes": count})
        if not count:
            continue
        with path.open("rb") as stream:
            stream.seek(size - count)
            previous, following, remaining = collections.deque(maxlen=5), 0, count
            context_priority = 3
            while remaining > 0:
                line = stream.readline(min(65536, remaining))
                if not line:
                    break
                remaining -= len(line)
                is_trace = any(pattern in line for pattern in trace_patterns)
                is_error = any(pattern in line for pattern in error_patterns)
                priority = (
                    0
                    if is_trace and is_error
                    else 1
                    if is_trace
                    else 2
                    if is_error
                    else 3
                    if b"read_barrier_:true" in line
                    else None
                )
                if priority is not None:
                    # Preserve the matching line before optional context so
                    # a huge preceding SQL line cannot consume the budget.
                    context = f"\nFILE={path}\n".encode() + line + b"[preceding context]\n" + b"".join(previous)
                    matched[priority] += 1
                    append(priority, context)
                    following, context_priority = 5, priority
                elif following:
                    append(context_priority, line)
                    following -= 1
                previous.append(line)
        budget -= count
    written = 0
    with (evidence / f"{label}-db-context.log").open("wb") as output:
        for content in buffers:
            part = content[: max(0, output_limit - written)]
            output.write(part)
            written += len(part)
    (evidence / f"{label}-log-inventory.json").write_text(
        json.dumps(
            {
                "files": inventory,
                "context_bytes": written,
                "context_capped": written >= output_limit,
                "priority_matches": dict(zip(priorities, matched, strict=True)),
                "priority_capped": dict(zip(priorities, capped, strict=True)),
                "scan_capped": any(item["read_bytes"] < item["size"] for item in inventory),
            },
            indent=2,
        )
    )


class Recorder:
    """Flush errors immediately and snapshot the first failure before recycling."""

    def __init__(self, root, evidence):
        self.root, self.evidence = root, evidence
        self.lock = threading.Lock()
        self.errors = 0
        self.traces = set()
        self.first = None

    def error(self, error, phase, client=None, **details):
        event = {"time": time.time(), "phase": phase, **details, **error_details(error)}
        if not event["trace_ids"] and client is not None:
            try:
                # Immediately on the failing worker's own connection; mark it
                # as best-effort, not proof of a particular internal SDK SQL.
                rows = client._server._execute("SELECT last_trace_id() FROM DUAL")
                event["trace_ids"] = TRACE.findall(str(rows))
                event["trace_source"] = "same_connection_last_trace_id_best_effort"
            except Exception as lookup_error:
                event["trace_lookup_error"] = str(lookup_error)
        with self.lock:
            self.errors += 1
            self.traces.update(event["trace_ids"])
            first = self.first is None
            if first:
                self.first = event
            # Keep a bounded sample but never lose the total failure count.
            if self.errors <= 200:
                with (self.evidence / "errors.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
            if first:
                (self.evidence / "first-error.json").write_text(json.dumps(event, indent=2))
        if first:
            print(f"[FTS][FIRST_ERROR] {json.dumps(event)}", flush=True)
            capture_logs(self.root, self.evidence, "early", event["trace_ids"])


def validate_report(report, scenario):
    """A zero worker exit alone is insufficient with native binding teardown."""
    assert report.get("scenario") == scenario, report
    assert report.get("completed") is True and report.get("errors") == 0, report
    assert report.get("rows") == ROWS, report
    if scenario == "first-pass":
        assert report.get("first_pass_successes") == QUERIES, report
        assert report.get("fts_successes") == QUERIES * PASSES, report
    else:
        counts = report.get("worker_successes", [])
        assert len(counts) == WORKERS and all(value > 0 for value in counts), report
        assert report.get("freeze_successes", 0) > 0, report
        assert report.get("elapsed_seconds", 0) >= SECONDS, report


def first_pass(client, recorder, report):
    import pyseekdb

    collection = client.create_collection(
        "fts_first_pass",
        configuration=pyseekdb.HNSWConfiguration(dimension=8, distance="cosine"),
        embedding_function=None,
    )
    vector = [float(i + 1) / 8 for i in range(8)]
    for start in range(0, ROWS, 100):
        indexes = range(start, min(start + 100, ROWS))
        collection.upsert(
            ids=[f"doc-{i}" for i in indexes],
            documents=[document(i) for i in indexes],
            embeddings=[vector for _ in indexes],
            metadatas=[{"ordinal": i} for i in indexes],
        )
        if start % 1000 == 0:
            print(f"[FTS] ingest={min(start + 100, ROWS)}/{ROWS}", flush=True)
    assert collection.count() == ROWS
    collection.refresh_index()
    report.update(rows=ROWS, first_pass_successes=0, fts_successes=0)
    # No FTS warm-up, sleep, retry or reopen before pass zero.
    for pass_id in range(PASSES):
        for query_id in range(QUERIES):
            try:
                result = collection.hybrid_search(
                    query={"where_document": {"$contains": f"{TEXT} cohort{query_id:03d}"}, "n_results": 10},
                    knn=None,
                    n_results=10,
                    include=["documents", "metadatas"],
                )
                assert result.get("ids") and result["ids"][0], result
                report["fts_successes"] += 1
                if pass_id == 0:
                    report["first_pass_successes"] += 1
            except Exception as error:
                recorder.error(
                    error,
                    "hybrid_search",
                    client,
                    pass_id=pass_id,
                    query_id=query_id,
                    query_text=f"{TEXT} cohort{query_id:03d}",
                )
        print(f"[FTS] pass={pass_id} successes={report['fts_successes']} errors={recorder.errors}", flush=True)


def insert_batch(client, start, size):
    # IDs/documents are generated here, not user-controlled SQL inputs.
    values = ",".join(f"('doc-{i}', '{document(i)}', '{{\"ordinal\":{i}}}')" for i in range(start, start + size))
    client._server._execute(f"INSERT INTO {TABLE} (`_id`, `document`, `metadata`) VALUES {values}")


def scan_freeze(client, make_client, recorder, report):
    client._server._execute(
        f"CREATE TABLE {TABLE} (`__pk_increment` BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY, "
        "`_id` VARCHAR(128) NOT NULL, `document` LONGTEXT NOT NULL, `metadata` JSON NOT NULL, "
        "UNIQUE KEY uk_id (`_id`), FULLTEXT KEY idx_document (`document`) WITH PARSER space)"
    )
    for start in range(0, ROWS, 100):
        insert_batch(client, start, min(100, ROWS - start))
        if start % 1000 == 0:
            print(f"[FTS] ingest={min(start + 100, ROWS)}/{ROWS}", flush=True)
    counts = client._server._execute(f"SELECT COUNT(*) AS row_count FROM {TABLE}")
    count = counts[0]["row_count"] if isinstance(counts[0], dict) else counts[0][0]
    assert count == ROWS, counts
    schema = client._server._execute(f"SHOW CREATE TABLE {TABLE}")
    assert "FULLTEXT" in str(schema).upper() and "VECTOR" not in str(schema).upper(), schema
    report.update(
        rows=ROWS,
        schema=str(schema),
        search_sql=SEARCH,
        write_batch_size=WRITE_BATCH,
        worker_successes=[0] * WORKERS,
        freeze_successes=0,
    )
    stop = threading.Event()
    deadline = [0.0]
    started = [0.0]

    def start_window():
        started[0] = time.monotonic()
        deadline[0] = started[0] + SECONDS

    barrier = threading.Barrier(WORKERS + 2, action=start_window)

    def worker(number):
        connection = None
        try:
            connection = make_client()
            connection._server._execute("SELECT 1")
            barrier.wait(timeout=60)
            while time.monotonic() < deadline[0] and not stop.is_set():
                started = time.monotonic()
                try:
                    if number == WORKERS:
                        start = ROWS + report["freeze_successes"] * WRITE_BATCH
                        insert_batch(connection, start, WRITE_BATCH)
                        connection._server._execute("ALTER SYSTEM MINOR FREEZE")
                        report["freeze_successes"] += 1
                    else:
                        rows = connection._server._execute(SEARCH)
                        assert rows, "full-text scan returned no rows"
                        report["worker_successes"][number] += 1
                except Exception as error:
                    recorder.error(error, "minor-freeze" if number == WORKERS else "scan", connection, worker=number)
                    if number == WORKERS:
                        stop.set()  # No valid pressure if writes/freezes cannot execute.
                if number == WORKERS:
                    stop.wait(max(0, INTERVAL - (time.monotonic() - started)))
        except Exception as error:
            recorder.error(error, "worker-setup", connection, worker=number)
            barrier.abort()
            stop.set()
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception as error:
                    recorder.error(error, "worker-close", worker=number)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS + 1)]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=60)
        while not stop.wait(min(5, max(0, deadline[0] - time.monotonic()))) and time.monotonic() < deadline[0]:
            print(
                f"[FTS] scans={sum(report['worker_successes'])} freezes={report['freeze_successes']} "
                f"errors={recorder.errors}",
                flush=True,
            )
    finally:
        for thread in threads:
            thread.join(timeout=65)
        assert not any(thread.is_alive() for thread in threads), "FTS worker did not drain"
        report["elapsed_seconds"] = time.monotonic() - started[0]


def run(scenario, root, evidence):
    import pyseekdb

    runtime_distribution = "seekdb" if importlib.util.find_spec("seekdb") else "pylibseekdb"

    report = {
        "scenario": scenario,
        "completed": False,
        "platform": platform.platform(),
        "pyseekdb": importlib.metadata.version("pyseekdb"),
        "embedded_runtime": runtime_distribution,
        "embedded_runtime_version": importlib.metadata.version(runtime_distribution),
        "corpus": "synthetic deterministic / 24515 documents",
        "workers": WORKERS,
        "duration_seconds": SECONDS,
        "freeze_interval_seconds": INTERVAL,
    }
    recorder = Recorder(root, evidence)
    client = None
    try:
        db_dir = root / "database"
        assert not db_dir.exists(), "fresh database required"

        def make_client():
            return pyseekdb.Client(path=str(db_dir), database="test")

        client = make_client()
        report["database_version"] = str(client._server._execute("SELECT VERSION()"))
        client._server._execute("ALTER SYSTEM SET max_syslog_file_count = 50")
        print(f"[FTS] start {json.dumps(report)}", flush=True)
        if scenario == "first-pass":
            first_pass(client, recorder, report)
        else:
            scan_freeze(client, make_client, recorder, report)
        report["completed"] = True
    except Exception as error:
        recorder.error(error, "scenario", client)
        traceback.print_exc()
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as error:
                recorder.error(error, "close")
        try:
            capture_logs(root, evidence, "final", recorder.traces)
        except Exception as error:
            recorder.error(error, "diagnostics")
        report.update(errors=recorder.errors, first_error=recorder.first, trace_ids=sorted(recorder.traces))
        (evidence / "summary.json").write_text(json.dumps(report, indent=2))
    validate_report(report, scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["first-pass", "scan-freeze"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    options = parser.parse_args()
    run(options.scenario, options.root, options.evidence)
