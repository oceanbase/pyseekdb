import os
import logging
from typing import Optional, Union

from .admin_client import AdminAPI, _AdminClientProxy, _ClientProxy
from .client_seekdb_embedded import _PYLIBSEEKDB_AVAILABLE

logger = logging.getLogger(__name__)


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
        # Embedded mode (requires pylibseekdb)
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
        # Default behavior: embedded mode if available, otherwise require host
        if _PYLIBSEEKDB_AVAILABLE:
            # Default to embedded mode with current working directory as path
            default_path = os.path.abspath("seekdb.db")
            logger.info(f"Creating embedded client (default): path={default_path}, database={database}")
            server = SeekdbEmbeddedClient(
                path=default_path,
                database=database,
                **kwargs
            )
        else:
            raise ValueError(
                "Default embedded mode is not available because pylibseekdb could not be imported. "
                "Please provide host/port parameters to use RemoteServerClient."
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
        # Embedded mode (requires pylibseekdb)
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
        # No parameters provided
        if _PYLIBSEEKDB_AVAILABLE:
            raise ValueError(
                "Must provide either path (embedded mode) or host (remote server mode) parameter"
            )
        else:
            raise ValueError(
                "Must provide host (remote server mode) parameter"
            )

    # Return AdminClient proxy (only exposes database operations)
    return _AdminClientProxy(server=server)



