"""
SeekDBClient client module

Provides client and admin factory functions with strict separation:

Collection Management (ClientAPI):
- Client() - Smart factory for Embedded/Server mode
- OBClient() - OceanBase client factory
- Returns: _ClientProxy (collection operations only)

Database Management (AdminAPI):
- AdminClient() - Smart factory for Embedded/Server mode  
- OBAdminClient() - OceanBase admin factory
- Returns: _AdminClientProxy (database operations only)

All factories use the underlying ServerAPI implementations:
- SeekdbEmbeddedClient - Local seekdb
- SeekdbServerClient - Remote seekdb via pymysql
- OceanBaseServerClient - OceanBase via pymysql
"""
import logging
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
from .client_seekdb_server import SeekdbServerClient
from .client_oceanbase_server import OceanBaseServerClient
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
    'SeekdbServerClient',
    'OceanBaseServerClient',
    'Client',
    'OBClient',
    'AdminAPI',
    'AdminClient',
    'OBAdminClient',
    'Database',
]


def Client(
    path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: str = "test",
    user: Optional[str] = None,
    password: str = "",
    **kwargs
) -> _ClientProxy:
    """
    Smart client factory function (returns ClientProxy for collection operations only)
    
    Automatically selects embedded or server mode based on parameters:
    - If path is provided, uses embedded mode
    - If host/port is provided, uses server mode
    
    Returns a ClientProxy that only exposes collection operations.
    For database management, use AdminClient().
    
    Args:
        path: seekdb data directory path (embedded mode)
        host: server address (server mode)
        port: server port (server mode)
        database: database name
        user: username (server mode)
        password: password (server mode)
        **kwargs: other parameters
    
    Returns:
        _ClientProxy: A proxy that only exposes collection operations
    
    Examples:
        >>> # Embedded mode
        >>> client = Client(path="/path/to/seekdb", database="db1")
        >>> client.create_collection("my_collection")  # ✅ Available
        >>> # client.create_database("new_db")  # ❌ Not available
        
        >>> # Server mode
        >>> client = Client(
        ...     host='localhost',
        ...     port=2882,
        ...     database="db1",
        ...     user="u01",
        ...     password="pass"
        ... )
    """
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
        # Server mode
        if port is None:
            port = 2882  # Default port
        if user is None:
            user = "root"
        
        logger.info(
            f"Creating server mode client: {user}@{host}:{port}/{database}"
        )
        server = SeekdbServerClient(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            **kwargs
        )
    
    else:
        raise ValueError(
            "Must provide either path (embedded mode) or host (server mode) parameter"
        )
    
    # Return ClientProxy (only exposes collection operations)
    return _ClientProxy(server=server)


def OBClient(
    host: str = "localhost",
    port: int = 2881,
    tenant: str = "test",
    database: str = "test",
    user: str = "root",
    password: str = "",
    **kwargs
) -> _ClientProxy:
    """
    OceanBase client factory function (returns ClientProxy for collection operations only)
    
    Returns a ClientProxy that only exposes collection operations.
    For database management, use OBAdminClient().
    
    Args:
        host: server address
        port: server port (default 2881)
        tenant: tenant name
        database: database name
        user: username (without tenant suffix)
        password: password
        **kwargs: other parameters
    
    Returns:
        _ClientProxy: A proxy that only exposes collection operations
    
    Examples:
        >>> client = OBClient(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="tenant1",
        ...     database="db1",
        ...     user="u01",
        ...     password="pass"
        ... )
        >>> client.create_collection("my_collection")  # ✅ Available
        >>> # client.create_database("new_db")  # ❌ Not available
    """
    logger.info(
        f"Creating OceanBase client: {user}@{tenant}@{host}:{port}/{database}"
    )
    
    server = OceanBaseServerClient(
        host=host,
        port=port,
        tenant=tenant,
        database=database,
        user=user,
        password=password,
        **kwargs
    )
    
    # Return ClientProxy (only exposes collection operations)
    return _ClientProxy(server=server)


def AdminClient(
    path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: str = "",
    **kwargs
) -> _AdminClientProxy:
    """
    Smart admin client factory function (proxy pattern)
    
    Automatically selects embedded or server mode based on parameters:
    - If path is provided, uses embedded mode
    - If host/port is provided, uses server mode
    
    Returns a lightweight AdminClient proxy that only exposes database operations.
    For collection management, use Client().
    
    Args:
        path: seekdb data directory path (embedded mode)
        host: server address (server mode)
        port: server port (server mode)
        user: username (server mode)
        password: password (server mode)
        **kwargs: other parameters
    
    Returns:
        _AdminClientProxy: A proxy that only exposes database operations
    
    Examples:
        >>> # Embedded mode
        >>> admin = AdminClient(path="/path/to/seekdb")
        >>> admin.create_database("new_db")  # ✅ Available
        >>> # admin.create_collection("coll")  # ❌ Not available
        
        >>> # Server mode
        >>> admin = AdminClient(
        ...     host='localhost',
        ...     port=2882,
        ...     user="root",
        ...     password="pass"
        ... )
    """
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
        # Server mode
        if port is None:
            port = 2882  # Default port
        if user is None:
            user = "root"
        
        logger.info(
            f"Creating server mode admin client: {user}@{host}:{port}"
        )
        server = SeekdbServerClient(
            host=host,
            port=port,
            database="information_schema",  # Use system database
            user=user,
            password=password,
            **kwargs
        )
    
    else:
        raise ValueError(
            "Must provide either path (embedded mode) or host (server mode) parameter"
        )
    
    # Return AdminClient proxy (only exposes database operations)
    return _AdminClientProxy(server=server)


def OBAdminClient(
    host: str = "localhost",
    port: int = 2881,
    tenant: str = "test",
    user: str = "root",
    password: str = "",
    **kwargs
) -> _AdminClientProxy:
    """
    OceanBase admin client factory function (proxy pattern)
    
    Returns a lightweight AdminClient proxy that only exposes database operations.
    For collection management, use OBClient().
    
    Args:
        host: server address
        port: server port (default 2881)
        tenant: tenant name
        user: username (without tenant suffix)
        password: password
        **kwargs: other parameters
    
    Returns:
        _AdminClientProxy: A proxy that only exposes database operations
    
    Examples:
        >>> admin = OBAdminClient(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="tenant1",
        ...     user="root",
        ...     password="pass"
        ... )
        >>> admin.create_database("new_db")  # ✅ Available
        >>> # admin.create_collection("coll")  # ❌ Not available
    """
    logger.info(
        f"Creating OceanBase admin client: {user}@{tenant}@{host}:{port}"
    )
    
    server = OceanBaseServerClient(
        host=host,
        port=port,
        tenant=tenant,
        database="information_schema",  # Use system database
        user=user,
        password=password,
        **kwargs
    )
    
    # Return AdminClient proxy (only exposes database operations)
    return _AdminClientProxy(server=server)

