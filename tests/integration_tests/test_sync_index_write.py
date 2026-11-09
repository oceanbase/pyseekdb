"""Integration coverage for write-time synchronous index refresh.

Runs for embedded, server, and oceanbase through the db_client fixture.
``sync=True`` waits for index refresh where seekdb supports it, and raises
before writing where the backend cannot provide that wait.
"""

import contextlib
import uuid

import pytest

from pyseekdb import HNSWConfiguration


def _new_collection(db_client):
    name = f"test_sync_{uuid.uuid4().hex}"
    return db_client.get_or_create_collection(
        name=name,
        configuration=HNSWConfiguration(dimension=3, distance="l2"),
        embedding_function=None,
    )


def _refresh_statements(sqls: list[str]) -> list[str]:
    return [sql for sql in sqls if "dbms_index_manager.refresh" in sql.lower()]


def _spy_execute(monkeypatch, server):
    executed: list[str] = []
    original = server._execute

    def wrapped(sql, *args, **kwargs):
        executed.append(sql)
        return original(sql, *args, **kwargs)

    monkeypatch.setattr(server, "_execute", wrapped)
    return executed


class TestSyncIndexWrite:
    """Synchronous writes are visible to approximate search when refresh exists."""

    def test_omitted_sync_does_not_refresh(self, db_client, monkeypatch):
        collection = _new_collection(db_client)
        executed = _spy_execute(monkeypatch, collection.client)
        row_id = f"async_{uuid.uuid4().hex}"
        try:
            collection.add(ids=row_id, embeddings=[0.2, 0.2, 0.2], documents="async write")
            assert _refresh_statements(executed) == []
            stored = collection.get(ids=row_id)
            assert stored["ids"] == [row_id]
        finally:
            with contextlib.suppress(Exception):
                db_client.delete_collection(collection.name)

    def test_sync_write_is_visible_or_rejected(self, db_client, monkeypatch):
        collection = _new_collection(db_client)
        supported = collection.client._refresh_enabled()
        executed = _spy_execute(monkeypatch, collection.client)
        row_id = f"sync_{uuid.uuid4().hex}"
        vector = [0.1, 0.2, 0.3]
        try:
            if not supported:
                with pytest.raises(ValueError, match="sync=True"):
                    collection.add(ids=row_id, embeddings=vector, documents="should not land", sync=True)
                assert _refresh_statements(executed) == []
                assert collection.get(ids=row_id)["ids"] == []
                return

            collection.add(ids=row_id, embeddings=vector, documents="visible now", sync=True)
            refresh_sql = _refresh_statements(executed)
            assert refresh_sql, "sync=True should call dbms_index_manager.refresh()"
            assert "INSERT" in executed[0].upper() or any("INSERT" in sql.upper() for sql in executed)
            results = collection.query(query_embeddings=vector, n_results=5)
            assert row_id in results["ids"][0]

            updated = [0.9, 0.1, 0.0]
            collection.update(ids=row_id, embeddings=updated, documents="updated", sync=True)
            updated_hits = collection.query(query_embeddings=updated, n_results=5)
            assert row_id in updated_hits["ids"][0]

            upsert_id = f"upsert_{uuid.uuid4().hex}"
            upsert_vector = [0.0, 0.8, 0.2]
            collection.upsert(ids=upsert_id, embeddings=upsert_vector, documents="upserted", sync=True)
            upsert_hits = collection.query(query_embeddings=upsert_vector, n_results=5)
            assert upsert_id in upsert_hits["ids"][0]

            collection.delete(ids=row_id, sync=True)
            remaining = collection.query(query_embeddings=updated, n_results=5)
            assert row_id not in remaining["ids"][0]
            assert collection.get(ids=row_id)["ids"] == []
        finally:
            with contextlib.suppress(Exception):
                db_client.delete_collection(collection.name)

    def test_namespace_sync_follows_refresh_support(self, db_client):
        """Namespace writes accept sync and refuse it when refresh is unavailable."""
        from namespace_dml_helpers import cleanup, create_ns_collection

        try:
            collection = create_ns_collection(db_client, suffix="_sync")
        except ValueError as exc:
            message = str(exc).lower()
            if "namespace" in message or "lakebase" in message or "use_namespace" in message:
                pytest.skip(str(exc))
            raise

        try:
            namespace = collection.create_namespace("sync_ns")
            supported = collection.client._refresh_enabled()
            if supported:
                namespace.add(ids="doc_1", embeddings=[0.1, 0.2, 0.3], documents="ns", sync=True)
                hits = namespace.query(query_embeddings=[0.1, 0.2, 0.3], n_results=5)
                assert "doc_1" in hits["ids"][0]
                namespace.delete(ids="doc_1", sync=True)
                after_delete = namespace.query(query_embeddings=[0.1, 0.2, 0.3], n_results=5)
                assert "doc_1" not in after_delete["ids"][0]
            else:
                with pytest.raises(ValueError, match="sync=True"):
                    namespace.add(ids="doc_1", embeddings=[0.1, 0.2, 0.3], documents="ns", sync=True)
                assert namespace.get(ids="doc_1")["ids"] == []
        finally:
            cleanup(db_client, collection)
