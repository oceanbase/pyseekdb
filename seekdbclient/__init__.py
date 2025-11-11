"""
SeekDBClient - Unified vector database client wrapper

Based on seekdb and pymysql, providing a simple and unified API.

Supports three modes:
1. Embedded mode - using local seekdb
2. Server mode - connecting to remote seekdb via pymysql
3. OceanBase mode - connecting to OceanBase via pymysql

Examples:
    >>> import seekdbclient

    >>> # Embedded mode - Collection management
    >>> client = seekdbclient.Client(path="./seekdb", database="test")

    >>> # Server mode - Collection management
    >>> client = seekdbclient.Client(
    ...     host='localhost',
    ...     port=2881,
    ...     database="test",
    ...     user="root",
    ...     password="pass"
    ... )

    >>> # OceanBase mode - Collection management
    >>> ob_client = seekdbclient.OBClient(
    ...     host='localhost',
    ...     port=2881,
    ...     tenant="test",
    ...     database="test",
    ...     user="root",
    ...     password=""
    ... )

    >>> # Admin client - Database management
    >>> admin = seekdbclient.AdminClient(path="./seekdb")
    >>> admin.create_database("new_db")
    >>> databases = admin.list_databases()
"""
import logging
import importlib.metadata

from .client import (
    BaseConnection,
    BaseClient,
    ClientAPI,
    HNSWConfiguration,
    DEFAULT_VECTOR_DIMENSION,
    DEFAULT_DISTANCE_METRIC,
    EmbeddingFunction,
    DefaultEmbeddingFunction,
    get_default_embedding_function,
    SeekdbEmbeddedClient,
    SeekdbServerClient,
    OceanBaseServerClient,
    Client,
    OBClient,
    AdminAPI,
    AdminClient,
    OBAdminClient,
    Database,
)
from .client.collection import Collection
from .client.query_result import QueryResult

try:
  __version__ = importlib.metadata.version("seekdbclient")
except importlib.metadata.PackageNotFoundError:
  __version__ = "0.0.1.dev1"

__author__ = "SeekDBClient Team"

__all__ = [
    'BaseConnection',
    'BaseClient',
    'ClientAPI',
    'HNSWConfiguration',
    'DEFAULT_VECTOR_DIMENSION',
    'DEFAULT_DISTANCE_METRIC',
    'EmbeddingFunction',
    'DefaultEmbeddingFunction',
    'get_default_embedding_function',
    'SeekdbEmbeddedClient',
    'SeekdbServerClient',
    'OceanBaseServerClient',
    'Client',
    'OBClient',
    'Collection',
    'QueryResult',
    'AdminAPI',
    'AdminClient',
    'OBAdminClient',
    'Database',
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

