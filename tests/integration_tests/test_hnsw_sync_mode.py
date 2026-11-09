"""Integration tests for HNSW sync_mode on collection creation."""

import time
import uuid

import pytest

from pyseekdb import HNSWConfiguration
from pyseekdb.client.client_base import _hnsw_sync_mode_supported
from pyseekdb.client.meta_info import CollectionNames


def _create_statement(client, collection) -> str:
    if collection.id is not None:
        table_name = CollectionNames.table_name_v2(collection.id)
    else:
        table_name = CollectionNames.table_name(collection.name)
    rows = client._server._execute(f"SHOW CREATE TABLE `{table_name}`")
    row = rows[0]
    if isinstance(row, dict):
        statement = row.get("Create Table", row.get("create table", ""))
    elif isinstance(row, (tuple, list)):
        statement = row[1] if len(row) > 1 else ""
    else:
        statement = str(row)
    return statement.replace(" ", "").replace("`", "").upper()


class TestHNSWSyncMode:
    def test_sync_mode_on_create(self, db_client):
        name = f"test_sync_mode_{int(time.time() * 1000)}"
        db_type, version = db_client._server.detect_db_type_and_version()
        config = HNSWConfiguration(dimension=4, distance="l2", sync_mode="immediate")

        if not _hnsw_sync_mode_supported(db_type, version):
            with pytest.raises(ValueError, match="sync_mode"):
                db_client.create_collection(name=name, configuration=config, embedding_function=None)
            assert db_client.has_collection(name) is False
            return

        collection = db_client.create_collection(name=name, configuration=config, embedding_function=None)
        try:
            statement = _create_statement(db_client, collection)
            assert "SYNC_MODE=IMMEDIATE" in statement

            row_id = str(uuid.uuid4())
            vector = [0.1, 0.2, 0.3, 0.4]
            collection.add(ids=row_id, embeddings=vector, documents="immediate row")
            found = collection.query(query_embeddings=vector, n_results=1)
            assert found["ids"][0][0] == row_id
        finally:
            db_client.delete_collection(name)

    def test_explicit_async_mode(self, db_client):
        name = f"test_sync_mode_async_{int(time.time() * 1000)}"
        db_type, version = db_client._server.detect_db_type_and_version()
        config = HNSWConfiguration(dimension=4, distance="cosine", sync_mode="async")
        if not _hnsw_sync_mode_supported(db_type, version):
            pytest.skip(f"{db_type} {version} does not accept HNSW sync_mode")

        collection = db_client.create_collection(name=name, configuration=config, embedding_function=None)
        try:
            statement = _create_statement(db_client, collection)
            assert "SYNC_MODE=ASYNC" in statement
        finally:
            db_client.delete_collection(name)
