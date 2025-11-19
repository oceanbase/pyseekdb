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
- SeekdbEmbeddedClient - Local seekdb
- RemoteServerClient - Remote server via pymysql (supports both seekdb Server and OceanBase Server)
"""
import logging
import os
from typing import Optional, Union

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
from .admin_client import AdminAPI, _AdminClientProxy, _ClientProxy
from .database import Database

logger = logging.getLogger(__name__)

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


def Client(
    path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    tenant: str = "sys",
    database: str = "test",
    user: Optional[str] = None,
    password: str = "", # Can be retrieved from SEEKDB_PASSWORD environment variable
    **kwargs
) -> _ClientProxy:
    """
    Smart client factory function (returns ClientProxy for collection operations only)
    
    Automatically selects embedded or remote server mode based on parameters:
    - If path is provided, uses embedded mode
    - If host/port is provided, uses remote server mode (supports both seekdb Server and OceanBase Server)
    - If neither path nor host is provided, defaults to embedded mode with current working directory as path
    
    Returns a ClientProxy that only exposes collection operations.
    For database management, use AdminClient().
    
    Args:
        path: seekdb data directory path (embedded mode). If not provided and host is also not provided, 
              defaults to current working directory
        host: server address (remote server mode)
        port: server port (remote server mode, default 2881)
        tenant: tenant name (remote server mode, default "sys" for seekdb Server, "test" for OceanBase)
        database: database name
        user: username (remote server mode, without tenant suffix)
        password: password (remote server mode). If not provided, will be retrieved from SEEKDB_PASSWORD environment variable
        **kwargs: other parameters
    
    Returns:
        _ClientProxy: A proxy that only exposes collection operations
    
    Examples:
        >>> # Embedded mode with explicit path
        >>> client = Client(path="/path/to/seekdb", database="db1")
        >>> client.create_collection("my_collection")  # ✅ Available
        
        >>> # Embedded mode (default, uses current working directory)
        >>> client = Client(database="db1")
        >>> client.create_collection("my_collection")  # ✅ Available
        
        >>> # Remote server mode (seekdb Server)
        >>> client = Client(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="sys",
        ...     database="db1",
        ...     user="root",
        ...     password="pass"
        ... )
        
        >>> # Remote server mode (OceanBase Server)
        >>> client = Client(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="test",
        ...     database="db1",
        ...     user="root",
        ...     password="pass"
        ... )
    """
    # Get password from environment variable if not provided
    if not password:
        password = os.environ.get("SEEKDB_PASSWORD", "")
    
    # Determine mode and create appropriate server
    if path is not None:
        # Embedded mode
        logger.info(f"Creating embedded client: path={path}, database={database}")
        server = SeekdbEmbeddedClient(
            path=path,
            database=database,
            **kwargs
        )
    
    elif host is not None:
        # Remote server mode (supports both seekdb Server and OceanBase Server)
        if port is None:
            port = 2881  # Default port
        if user is None:
            user = "root"
        
        logger.info(
            f"Creating remote server client: {user}@{tenant}@{host}:{port}/{database}"
        )
        server = RemoteServerClient(
            host=host,
            port=port,
            tenant=tenant,
            database=database,
            user=user,
            password=password,
            **kwargs
        )
    
    else:
        # Default to embedded mode with current working directory as path
        default_path = os.path.abspath("seekdb.db")
        logger.info(f"Creating embedded client (default): path={default_path}, database={database}")
        server = SeekdbEmbeddedClient(
            path=default_path,
            database=database,
            **kwargs
        )
    
    # Return ClientProxy (only exposes collection operations)
    return _ClientProxy(server=server)


def AdminClient(
    path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    tenant: str = "sys",
    user: Optional[str] = None,
    password: str = "", # Can be retrieved from SEEKDB_PASSWORD environment variable
    **kwargs
) -> _AdminClientProxy:
    """
    Smart admin client factory function (proxy pattern)
    
    Automatically selects embedded or remote server mode based on parameters:
    - If path is provided, uses embedded mode
    - If host/port is provided, uses remote server mode (supports both seekdb Server and OceanBase Server)
    
    Returns a lightweight AdminClient proxy that only exposes database operations.
    For collection management, use Client().
    
    Args:
        path: seekdb data directory path (embedded mode)
        host: server address (remote server mode)
        port: server port (remote server mode, default 2881)
        tenant: tenant name (remote server mode, default "sys" for seekdb Server, "test" for OceanBase)
        user: username (remote server mode, without tenant suffix)
        password: password (remote server mode). If not provided, will be retrieved from SEEKDB_PASSWORD environment variable
        **kwargs: other parameters
    
    Returns:
        _AdminClientProxy: A proxy that only exposes database operations
    
    Examples:
        >>> # Embedded mode
        >>> admin = AdminClient(path="/path/to/seekdb")
        >>> admin.create_database("new_db")  # ✅ Available
        >>> # admin.create_collection("coll")  # ❌ Not available
        
        >>> # Remote server mode (seekdb Server)
        >>> admin = AdminClient(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="sys",
        ...     user="root",
        ...     password="pass"
        ... )
        
        >>> # Remote server mode (OceanBase Server)
        >>> admin = AdminClient(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="test",
        ...     user="root",
        ...     password="pass"
        ... )
    """
    # Get password from environment variable if not provided
    if not password:
        password = os.environ.get("SEEKDB_PASSWORD", "")
    
    # Determine mode and create appropriate server
    if path is not None:
        # Embedded mode
        logger.info(f"Creating embedded admin client: path={path}")
        server = SeekdbEmbeddedClient(
            path=path,
            database="information_schema",  # Use system database for admin operations
            **kwargs
        )
    
    elif host is not None:
        # Remote server mode (supports both seekdb Server and OceanBase Server)
        if port is None:
            port = 2881  # Default port
        if user is None:
            user = "root"
        
        logger.info(
            f"Creating remote server admin client: {user}@{tenant}@{host}:{port}"
        )
        server = RemoteServerClient(
            host=host,
            port=port,
            tenant=tenant,
            database="information_schema",  # Use system database
            user=user,
            password=password,
            **kwargs
        )
    
    else:
        raise ValueError(
            "Must provide either path (embedded mode) or host (remote server mode) parameter"
        )
    
    # Return AdminClient proxy (only exposes database operations)
    return _AdminClientProxy(server=server)



