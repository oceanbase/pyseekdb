"""
Integration tests for concurrent collection operations in embedded mode.

Covers multiprocess + multithread workloads. Each thread must use its own
``pyseekdb.Client`` instance because ``Client`` is not thread-safe.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import importlib.metadata
import multiprocessing as mp
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty
from typing import Any

import pytest
from packaging.version import Version

repo_root = Path(__file__).resolve().parents[2]
src_root = repo_root / "src"
sys.path.insert(0, str(src_root))

EMBED_DIM = 8
NUM_PROCESSES = 2
THREADS_PER_PROCESS = 3
ITEMS_PER_THREAD = 5
WORKER_TIMEOUT_SECONDS = 60
MIN_PYLIBSEEKDB_VERSION = Version("1.3.0.post1")


def _purge_pyseekdb_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "pyseekdb" or module_name.startswith("pyseekdb."):
            del sys.modules[module_name]


def _build_embedding(seed: int, embed_dim: int = EMBED_DIM) -> list[float]:
    base = float(seed + 1)
    return [((base + i) % 13) / 13.0 for i in range(embed_dim)]


def _record_id(process_id: int, thread_id: int, seq: int) -> str:
    return f"p{process_id:02d}_t{thread_id:02d}_{seq:04d}"


def _import_pyseekdb():
    return importlib.import_module("pyseekdb")


def _make_client(db_path: str, database: str):
    pyseekdb = _import_pyseekdb()
    return pyseekdb.Client(path=db_path, database=database)


def _get_collection(client, collection_name: str):
    return client.get_collection(collection_name, embedding_function=None)


def _refresh_collection(db_path: str, database: str, collection_name: str) -> None:
    client = _make_client(db_path, database)
    collection = _get_collection(client, collection_name)
    collection.refresh_index()


def _require_embedded_pylibseekdb() -> None:
    try:
        import pylibseekdb  # noqa: F401
    except ImportError:
        pytest.skip("seekdb embedded package is not installed")

    try:
        installed_version = Version(importlib.metadata.version("pylibseekdb"))
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("pylibseekdb is not installed")

    if installed_version <= MIN_PYLIBSEEKDB_VERSION:
        pytest.skip(
            f"embedded multiprocess tests require pylibseekdb > {MIN_PYLIBSEEKDB_VERSION}, got {installed_version}"
        )


def _run_processes(
    target: Callable[..., None],
    args_list: list[tuple[Any, ...]],
    expected_count: int,
    timeout: float = WORKER_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    result_queue: mp.Queue[dict[str, Any]] = mp.Queue()
    processes = [mp.Process(target=target, args=(*args, result_queue)) for args in args_list]

    for process in processes:
        process.start()

    deadline = time.time() + timeout
    while time.time() < deadline and any(process.is_alive() for process in processes):
        time.sleep(0.1)

    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)

    results: list[dict[str, Any]] = []
    while len(results) < expected_count:
        try:
            results.append(result_queue.get(timeout=0.5))
        except Empty:
            break

    return results


def _get_or_create_worker(
    db_path: str,
    database: str,
    collection_name: str,
    delay: float,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()
    time.sleep(delay)
    try:
        pyseekdb = _import_pyseekdb()
        client = pyseekdb.Client(path=db_path, database=database)
        collection = client.get_or_create_collection(
            collection_name,
            configuration=pyseekdb.HNSWConfiguration(dimension=EMBED_DIM, distance="cosine"),
            embedding_function=None,
        )
        output.put({"ok": True, "name": collection.name})
    except Exception as exc:
        output.put({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})


def _add_worker(
    db_path: str,
    database: str,
    collection_name: str,
    process_id: int,
    threads_per_process: int,
    items_per_thread: int,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()

    def add_in_thread(thread_id: int) -> int:
        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        ids = [_record_id(process_id, thread_id, seq) for seq in range(items_per_thread)]
        embeddings = [_build_embedding(process_id * 10_000 + thread_id * 100 + seq) for seq in range(items_per_thread)]
        documents = [f"insert p={process_id} t={thread_id} seq={seq}" for seq in range(items_per_thread)]
        metadatas = [{"process_id": process_id, "thread_id": thread_id, "seq": seq} for seq in range(items_per_thread)]
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(ids)

    try:
        inserted = 0
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            futures = [executor.submit(add_in_thread, thread_id) for thread_id in range(threads_per_process)]
            for future in as_completed(futures):
                inserted += future.result()
        output.put({"ok": True, "process_id": process_id, "inserted": inserted})
    except Exception as exc:
        output.put({"ok": False, "process_id": process_id, "error_type": type(exc).__name__, "error": str(exc)})


def _get_worker(
    db_path: str,
    database: str,
    collection_name: str,
    process_id: int,
    threads_per_process: int,
    items_per_thread: int,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()

    def get_in_thread(thread_id: int) -> int:
        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        ids = [_record_id(process_id, thread_id, seq) for seq in range(items_per_thread)]
        result = collection.get(ids=ids, include=["documents", "metadatas"])
        if len(result["ids"]) != len(ids):
            raise AssertionError(f"expected {len(ids)} rows, got {len(result['ids'])}")
        return len(result["ids"])

    try:
        fetched = 0
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            futures = [executor.submit(get_in_thread, thread_id) for thread_id in range(threads_per_process)]
            for future in as_completed(futures):
                fetched += future.result()
        output.put({"ok": True, "process_id": process_id, "fetched": fetched})
    except Exception as exc:
        output.put({"ok": False, "process_id": process_id, "error_type": type(exc).__name__, "error": str(exc)})


def _query_worker(
    db_path: str,
    database: str,
    collection_name: str,
    process_id: int,
    threads_per_process: int,
    queries_per_thread: int,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()

    def query_in_thread(thread_id: int) -> int:
        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        completed = 0
        for query_idx in range(queries_per_thread):
            seed = process_id * 10_000 + thread_id * 100 + query_idx
            result = collection.query(
                query_embeddings=_build_embedding(seed),
                n_results=3,
                include=["documents", "metadatas"],
            )
            if not result["ids"] or not result["ids"][0]:
                raise AssertionError("query returned empty result set")
            completed += 1
        return completed

    try:
        queried = 0
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            futures = [executor.submit(query_in_thread, thread_id) for thread_id in range(threads_per_process)]
            for future in as_completed(futures):
                queried += future.result()
        output.put({"ok": True, "process_id": process_id, "queried": queried})
    except Exception as exc:
        output.put({"ok": False, "process_id": process_id, "error_type": type(exc).__name__, "error": str(exc)})


def _update_worker(
    db_path: str,
    database: str,
    collection_name: str,
    process_id: int,
    threads_per_process: int,
    items_per_thread: int,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()

    def update_in_thread(thread_id: int) -> int:
        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        ids = [_record_id(process_id, thread_id, seq) for seq in range(items_per_thread)]
        metadatas = [
            {"process_id": process_id, "thread_id": thread_id, "seq": seq, "updated": True}
            for seq in range(items_per_thread)
        ]
        collection.update(ids=ids, metadatas=metadatas)
        result = collection.get(ids=ids, include=["metadatas"])
        for metadata in result["metadatas"]:
            if not metadata.get("updated"):
                raise AssertionError(f"metadata not updated: {metadata}")
        return len(ids)

    try:
        updated = 0
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            futures = [executor.submit(update_in_thread, thread_id) for thread_id in range(threads_per_process)]
            for future in as_completed(futures):
                updated += future.result()
        output.put({"ok": True, "process_id": process_id, "updated": updated})
    except Exception as exc:
        output.put({"ok": False, "process_id": process_id, "error_type": type(exc).__name__, "error": str(exc)})


def _delete_worker(
    db_path: str,
    database: str,
    collection_name: str,
    process_id: int,
    threads_per_process: int,
    items_per_thread: int,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()

    def delete_in_thread(thread_id: int) -> int:
        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        ids = [_record_id(process_id, thread_id, seq) for seq in range(items_per_thread)]
        collection.delete(ids=ids)
        result = collection.get(ids=ids, include=["documents"])
        if result["ids"]:
            raise AssertionError(f"expected deleted ids to be gone, still have {result['ids']}")
        return len(ids)

    try:
        deleted = 0
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            futures = [executor.submit(delete_in_thread, thread_id) for thread_id in range(threads_per_process)]
            for future in as_completed(futures):
                deleted += future.result()
        output.put({"ok": True, "process_id": process_id, "deleted": deleted})
    except Exception as exc:
        output.put({"ok": False, "process_id": process_id, "error_type": type(exc).__name__, "error": str(exc)})


def _mixed_crud_worker(
    db_path: str,
    database: str,
    collection_name: str,
    process_id: int,
    threads_per_process: int,
    items_per_thread: int,
    output: mp.Queue[dict[str, Any]],
) -> None:
    _purge_pyseekdb_modules()

    def mixed_in_thread(thread_id: int) -> dict[str, int]:
        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        ids = [_record_id(process_id, thread_id, seq) for seq in range(items_per_thread)]
        embeddings = [_build_embedding(process_id * 10_000 + thread_id * 100 + seq) for seq in range(items_per_thread)]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[f"mixed-add p={process_id} t={thread_id} seq={seq}" for seq in range(items_per_thread)],
            metadatas=[
                {"phase": "add", "process_id": process_id, "thread_id": thread_id, "seq": seq}
                for seq in range(items_per_thread)
            ],
        )

        get_result = collection.get(ids=ids, include=["documents"])
        if len(get_result["ids"]) != len(ids):
            raise AssertionError("mixed get after add failed")

        collection.refresh_index()
        query_result = collection.query(
            query_embeddings=_build_embedding(process_id * 10_000 + thread_id * 100),
            n_results=min(3, len(ids)),
            include=["documents"],
        )
        if not query_result["ids"] or not query_result["ids"][0]:
            raise AssertionError("mixed query after add failed")

        collection.update(
            ids=ids,
            metadatas=[
                {"phase": "update", "process_id": process_id, "thread_id": thread_id, "seq": seq}
                for seq in range(items_per_thread)
            ],
        )

        delete_ids = ids[: max(1, items_per_thread // 2)]
        collection.delete(ids=delete_ids)
        remaining = collection.get(ids=ids, include=["documents"])
        remaining_ids = set(remaining["ids"])
        for deleted_id in delete_ids:
            if deleted_id in remaining_ids:
                raise AssertionError(f"mixed delete left id behind: {deleted_id}")

        return {
            "added": len(ids),
            "got": len(get_result["ids"]),
            "queried": len(query_result["ids"][0]),
            "updated": len(ids),
            "deleted": len(delete_ids),
        }

    try:
        totals = {"added": 0, "got": 0, "queried": 0, "updated": 0, "deleted": 0}
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            futures = [executor.submit(mixed_in_thread, thread_id) for thread_id in range(threads_per_process)]
            for future in as_completed(futures):
                thread_totals = future.result()
                for key, value in thread_totals.items():
                    totals[key] += value
        output.put({"ok": True, "process_id": process_id, **totals})
    except Exception as exc:
        output.put({"ok": False, "process_id": process_id, "error_type": type(exc).__name__, "error": str(exc)})


@pytest.fixture
def embedded_multiprocess_db():
    _require_embedded_pylibseekdb()

    db_path = Path(tempfile.mkdtemp(prefix="seekdb-mp-"))
    database = "test_mp"

    pyseekdb = _import_pyseekdb()
    admin = pyseekdb.AdminClient(path=str(db_path))
    admin.create_database(database)
    del admin
    gc.collect()

    yield str(db_path), database


@pytest.fixture
def crud_collection(embedded_multiprocess_db):
    db_path, database = embedded_multiprocess_db
    collection_name = f"mp_crud_{uuid.uuid4().hex}"

    pyseekdb = _import_pyseekdb()
    client = pyseekdb.Client(path=db_path, database=database)
    client.create_collection(
        name=collection_name,
        configuration=pyseekdb.HNSWConfiguration(dimension=EMBED_DIM, distance="cosine"),
        embedding_function=None,
    )

    yield db_path, database, collection_name

    with contextlib.suppress(Exception):
        client.delete_collection(collection_name)


def _seed_collection_rows(
    db_path: str,
    database: str,
    collection_name: str,
    num_processes: int,
    threads_per_process: int,
    items_per_thread: int,
) -> int:
    pyseekdb = _import_pyseekdb()
    client = pyseekdb.Client(path=db_path, database=database)
    collection = _get_collection(client, collection_name)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, int]] = []

    for process_id in range(num_processes):
        for thread_id in range(threads_per_process):
            for seq in range(items_per_thread):
                ids.append(_record_id(process_id, thread_id, seq))
                seed = process_id * 10_000 + thread_id * 100 + seq
                embeddings.append(_build_embedding(seed))
                documents.append(f"seed p={process_id} t={thread_id} seq={seq}")
                metadatas.append({"process_id": process_id, "thread_id": thread_id, "seq": seq})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    _refresh_collection(db_path, database, collection_name)
    return len(ids)


class TestGetOrCreateCollectionMultiprocess:
    def test_concurrent_get_or_create_collection(self, embedded_multiprocess_db):
        db_path, database = embedded_multiprocess_db
        collection_name = f"mp_collection_{uuid.uuid4().hex}"

        results = _run_processes(
            _get_or_create_worker,
            [
                (db_path, database, collection_name, 0.0),
                (db_path, database, collection_name, 0.05),
                (db_path, database, collection_name, 0.1),
            ],
            expected_count=3,
        )

        assert len(results) == 3
        assert all(result["ok"] for result in results), results

        client = _make_client(db_path, database)
        try:
            assert client.has_collection(collection_name)
            collection = _get_collection(client, collection_name)
            assert collection.name == collection_name
        finally:
            with contextlib.suppress(Exception):
                client.delete_collection(collection_name)


class TestMultiprocessMultithreadCrud:
    def test_concurrent_add(self, crud_collection):
        db_path, database, collection_name = crud_collection

        results = _run_processes(
            _add_worker,
            [
                (db_path, database, collection_name, process_id, THREADS_PER_PROCESS, ITEMS_PER_THREAD)
                for process_id in range(NUM_PROCESSES)
            ],
            expected_count=NUM_PROCESSES,
        )

        assert len(results) == NUM_PROCESSES
        assert all(result["ok"] for result in results), results

        expected_rows = NUM_PROCESSES * THREADS_PER_PROCESS * ITEMS_PER_THREAD
        total_inserted = sum(result["inserted"] for result in results)
        assert total_inserted == expected_rows

        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        assert collection.count() == expected_rows

    def test_concurrent_get(self, crud_collection):
        db_path, database, collection_name = crud_collection
        expected_rows = _seed_collection_rows(
            db_path, database, collection_name, NUM_PROCESSES, THREADS_PER_PROCESS, ITEMS_PER_THREAD
        )

        results = _run_processes(
            _get_worker,
            [
                (db_path, database, collection_name, process_id, THREADS_PER_PROCESS, ITEMS_PER_THREAD)
                for process_id in range(NUM_PROCESSES)
            ],
            expected_count=NUM_PROCESSES,
        )

        assert len(results) == NUM_PROCESSES
        assert all(result["ok"] for result in results), results
        assert sum(result["fetched"] for result in results) == expected_rows

    def test_concurrent_query(self, crud_collection):
        db_path, database, collection_name = crud_collection
        _seed_collection_rows(db_path, database, collection_name, NUM_PROCESSES, THREADS_PER_PROCESS, ITEMS_PER_THREAD)

        results = _run_processes(
            _query_worker,
            [
                (db_path, database, collection_name, process_id, THREADS_PER_PROCESS, ITEMS_PER_THREAD)
                for process_id in range(NUM_PROCESSES)
            ],
            expected_count=NUM_PROCESSES,
        )

        assert len(results) == NUM_PROCESSES
        assert all(result["ok"] for result in results), results
        expected_queries = NUM_PROCESSES * THREADS_PER_PROCESS * ITEMS_PER_THREAD
        assert sum(result["queried"] for result in results) == expected_queries

    def test_concurrent_update(self, crud_collection):
        db_path, database, collection_name = crud_collection
        expected_rows = _seed_collection_rows(
            db_path, database, collection_name, NUM_PROCESSES, THREADS_PER_PROCESS, ITEMS_PER_THREAD
        )

        results = _run_processes(
            _update_worker,
            [
                (db_path, database, collection_name, process_id, THREADS_PER_PROCESS, ITEMS_PER_THREAD)
                for process_id in range(NUM_PROCESSES)
            ],
            expected_count=NUM_PROCESSES,
        )

        assert len(results) == NUM_PROCESSES
        assert all(result["ok"] for result in results), results
        assert sum(result["updated"] for result in results) == expected_rows

    def test_concurrent_delete(self, crud_collection):
        db_path, database, collection_name = crud_collection
        expected_rows = _seed_collection_rows(
            db_path, database, collection_name, NUM_PROCESSES, THREADS_PER_PROCESS, ITEMS_PER_THREAD
        )

        results = _run_processes(
            _delete_worker,
            [
                (db_path, database, collection_name, process_id, THREADS_PER_PROCESS, ITEMS_PER_THREAD)
                for process_id in range(NUM_PROCESSES)
            ],
            expected_count=NUM_PROCESSES,
        )

        assert len(results) == NUM_PROCESSES
        assert all(result["ok"] for result in results), results
        assert sum(result["deleted"] for result in results) == expected_rows

        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        assert collection.count() == 0

    def test_concurrent_mixed_crud(self, crud_collection):
        db_path, database, collection_name = crud_collection

        results = _run_processes(
            _mixed_crud_worker,
            [
                (db_path, database, collection_name, process_id, THREADS_PER_PROCESS, ITEMS_PER_THREAD)
                for process_id in range(NUM_PROCESSES)
            ],
            expected_count=NUM_PROCESSES,
        )

        assert len(results) == NUM_PROCESSES
        assert all(result["ok"] for result in results), results

        expected_added = NUM_PROCESSES * THREADS_PER_PROCESS * ITEMS_PER_THREAD
        expected_deleted = NUM_PROCESSES * THREADS_PER_PROCESS * max(1, ITEMS_PER_THREAD // 2)
        assert sum(result["added"] for result in results) == expected_added
        assert sum(result["deleted"] for result in results) == expected_deleted

        client = _make_client(db_path, database)
        collection = _get_collection(client, collection_name)
        assert collection.count() == expected_added - expected_deleted


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
