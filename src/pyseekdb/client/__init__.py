"""
pyseekdb client module

Provides client and admin factory functions with strict separation:

Collection Management (ClientAPI):
- Client() - Smart factory for Embedded/Remote Server mode
- Returns: _ClientProxy (collection operations only)

Database Management (AdminAPI):
- AdminClient() - Smart factory for Embedded/Remote Server mode  
- Returns: _AdminClientProxy (database operations only)

All factories use the underlying ServerAPI implementations:
- SeekdbEmbeddedClient - Local seekdb (requires pylibseekdb, Linux only)
- RemoteServerClient - Remote server via pymysql (supports both seekdb Server and OceanBase Server)
"""

from .base_connection import BaseConnection
from .client_base import (
    BaseClient,
    ClientAPI,
    HNSWConfiguration,
    DEFAULT_VECTOR_DIMENSION,
    DEFAULT_DISTANCE_METRIC
)
from .embedding_function import (
    EmbeddingFunction,
    DefaultEmbeddingFunction,
    get_default_embedding_function
)
from .client_seekdb_embedded import SeekdbEmbeddedClient
from .client_seekdb_server import RemoteServerClient
from .client import Client, AdminClient
from .database import Database

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
    'RemoteServerClient',
    'Client',
    'AdminAPI',
    'AdminClient',
    'Database',
]

