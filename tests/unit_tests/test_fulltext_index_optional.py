"""Tests for optional fulltext index provisioning and later add/drop."""

import pytest

from pyseekdb.client.client_base import BaseClient
from pyseekdb.client.configuration import FulltextIndexConfig, HNSWConfiguration, VectorIndexConfig
from pyseekdb.client.schema import Schema

FULLTEXT_CLAUSE = "FULLTEXT INDEX idx_fts(document)"
DEFAULT_CLAUSE = "FULLTEXT INDEX idx_fts(document) WITH PARSER ik"
COLLECTION_ID = "cid123"
TABLE_NAME = f"c$v2${COLLECTION_ID}"


class RecordingClient(BaseClient):
    """Concrete BaseClient that records SQL instead of executing it."""

    def __init__(self):
        """Init."""
        self.database = "test"
        self.executed_sqls = []

    # The base implementations are used where the test needs real SQL construction.
    create_collection = BaseClient.create_collection

    def get_collection(self, name, embedding_function=None):
        """Not needed by these tests."""
        raise NotImplementedError

    def delete_collection(self, name):
        """Not needed by these tests."""
        raise NotImplementedError

    def list_collections(self):
        """Not needed by these tests."""
        return []

    @property
    def mode(self):
        """Report a stable client mode."""
        return "test"

    def is_connected(self):
        """Report a live connection."""
        return True

    def _cleanup(self):
        """Nothing to clean up."""

    def get_raw_connection(self):
        """No raw connection in these tests."""
        return None

    def _ensure_connection(self):
        """No real connection is opened by these tests."""
        return None

    def _execute(self, sql, params=None):
        """Record the SQL and report success."""
        self.executed_sqls.append(sql)
        return 0

    def has_collection(self, collection_name):
        """Report that no collection exists yet."""
        return False

    def _create_collection_meta_v2(
        self, collection_name, embedding_function, sparse_vector_index_config=None, **kwargs
    ):
        """Return fixed catalog metadata without touching a server."""
        return {"collection_id": COLLECTION_ID, "table_name": TABLE_NAME}


def _schema(**kwargs):
    """Build a schema with an explicit vector index so no default embedding function is loaded."""
    vector_index = VectorIndexConfig(
        hnsw=HNSWConfiguration(dimension=3, distance="l2"),
        embedding_function=None,
    )
    return Schema(vector_index=vector_index, **kwargs)


def _create_collection_sql(**schema_kwargs):
    """Create a collection on the recording client and return the CREATE TABLE statement."""
    client = RecordingClient()
    client.create_collection("docs", schema=_schema(**schema_kwargs))
    statements = [sql for sql in client.executed_sqls if "CREATE TABLE" in sql]
    assert len(statements) == 1
    return statements[0]


def test_default_schema_keeps_fulltext_index():
    """A schema without fulltext options keeps provisioning the default IK index."""
    assert DEFAULT_CLAUSE in _create_collection_sql()


def test_explicit_fulltext_config_is_used():
    """An explicit fulltext configuration still reaches the CREATE TABLE statement."""
    sql = _create_collection_sql(fulltext_index=FulltextIndexConfig(analyzer="space"))
    assert "FULLTEXT INDEX idx_fts(document) WITH PARSER space" in sql


def test_fulltext_index_can_be_disabled():
    """``fulltext_index=False`` creates the collection without a fulltext index."""
    sql = _create_collection_sql(fulltext_index=False)
    assert FULLTEXT_CLAUSE not in sql
    assert "VECTOR INDEX idx_vec (embedding)" in sql


def test_fulltext_index_rejects_other_values():
    """Only a FulltextIndexConfig, False, or None are accepted."""
    with pytest.raises(TypeError):
        _schema(fulltext_index=True)
    with pytest.raises(TypeError):
        _schema(fulltext_index="ik")


def test_add_and_drop_fulltext_index():
    """A collection created without a fulltext index can add and drop it later."""
    client = RecordingClient()
    collection = client.create_collection("docs", schema=_schema(fulltext_index=False))
    client.executed_sqls.clear()

    collection.create_fulltext_index()
    assert client.executed_sqls == [f"CREATE FULLTEXT INDEX idx_fts ON `{TABLE_NAME}` (document) WITH PARSER ik"]

    client.executed_sqls.clear()
    collection.create_fulltext_index(FulltextIndexConfig(analyzer="space"))
    assert client.executed_sqls == [f"CREATE FULLTEXT INDEX idx_fts ON `{TABLE_NAME}` (document) WITH PARSER space"]

    client.executed_sqls.clear()
    collection.drop_fulltext_index()
    assert client.executed_sqls == [f"DROP INDEX idx_fts ON `{TABLE_NAME}`"]
