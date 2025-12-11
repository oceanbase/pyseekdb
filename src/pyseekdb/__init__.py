"""
pyseekdb - Unified vector database client wrapper

Based on seekdb and pymysql, providing a simple and unified API.

Supports two modes:
1. Embedded mode - using local seekdb
2. Remote server mode - connecting to remote server via pymysql (supports both seekdb Server and OceanBase Server)

Examples:
    >>> import pyseekdb

    >>> # Embedded mode - Collection management
    >>> client = pyseekdb.Client(path="./seekdb", database="test")

    >>> # Remote server mode (seekdb Server) - Collection management
    >>> client = pyseekdb.Client(
    ...     host='localhost',
    ...     port=2881,
    ...     tenant="sys",
    ...     database="test",
    ...     user="root",
    ...     password="pass"
    ... )

    >>> # Remote server mode (OceanBase Server) - Collection management
    >>> client = pyseekdb.Client(
    ...     host='localhost',
    ...     port=2881,
    ...     tenant="test",
    ...     database="test",
    ...     user="root",
    ...     password="pass"
    ... )

    >>> # Admin client - Database management
    >>> admin = pyseekdb.AdminClient(path="./seekdb")
    >>> admin.create_database("new_db")
    >>> databases = admin.list_databases()
"""
import importlib.metadata

from .client import (
    BaseConnection,
    BaseClient,
    ClientAPI,
    HNSWConfiguration,
    EmbeddingFunction,
    DefaultEmbeddingFunction,
    get_default_embedding_function,
    SeekdbEmbeddedClient,
    RemoteServerClient,
    Client,
    AdminAPI,
    AdminClient,
    Database,
)
from .client.collection import Collection
from .client.hybrid_search import (
    HybridSearch,
    DOCUMENT,
    TEXT,
    EMBEDDINGS,
    K,
    IDS,
    DOCUMENTS,
    METADATAS,
    EMBEDDINGS_FIELD,
    SCORES,
)

try:
  __version__ = importlib.metadata.version("pyseekdb")
except importlib.metadata.PackageNotFoundError:
  __version__ = "0.0.1.dev1"

__author__ = "OceanBase <open_oceanbase@oceanbase.com>"

__all__ = [
    'BaseConnection',
    'BaseClient',
    'ClientAPI',
    'HNSWConfiguration',
    'EmbeddingFunction',
    'DefaultEmbeddingFunction',
    'get_default_embedding_function',
    'SeekdbEmbeddedClient',
    'RemoteServerClient',
    'Client',
    'Collection',
    'AdminAPI',
    'AdminClient',
    'Database',
    'HybridSearch',
    'DOCUMENT',
    'TEXT',
    'EMBEDDINGS',
    'K',
    'IDS',
    'DOCUMENTS',
    'METADATAS',
    'EMBEDDINGS_FIELD',
    'SCORES',
]

