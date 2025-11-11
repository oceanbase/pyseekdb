"""
Admin client interface and implementation for database management

Also includes ClientProxy for strict separation of Collection vs Database operations
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, TYPE_CHECKING, Any

from .database import Database

if TYPE_CHECKING:
    from .client_base import BaseClient, ClientAPI, HNSWConfiguration, ConfigurationParam, EmbeddingFunctionParam
    from .embedding_function import EmbeddingFunction, Documents as EmbeddingDocuments
    from .collection import Collection

# Delay import to avoid circular import
# We'll import these lazily in the functions that need them
# For now, create a placeholder that we can detect and replace
_PLACEHOLDER = object()  # Unique placeholder object

def _get_not_provided():
    """Get the real _NOT_PROVIDED from client_base"""
    from .client_base import _NOT_PROVIDED
    return _NOT_PROVIDED

# Use placeholder for default parameter - will be replaced in function
_NOT_PROVIDED = _PLACEHOLDER
ConfigurationParam = Any  # Type hint placeholder
EmbeddingFunctionParam = Any  # Type hint placeholder

DEFAULT_TENANT = "test"


class AdminAPI(ABC):
    """
    Abstract admin API interface for database management.
    Defines the contract for database operations.
    """
    
    @abstractmethod
    def create_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Create database
        
        Args:
            name: database name
            tenant: tenant name (for OceanBase)
        """
        pass
    
    @abstractmethod
    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Get database object
        
        Args:
            name: database name
            tenant: tenant name (for OceanBase)
            
        Returns:
            Database object
        """
        pass
    
    @abstractmethod
    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Delete database
        
        Args:
            name: database name
            tenant: tenant name (for OceanBase)
        """
        pass
    
    @abstractmethod
    def list_databases(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tenant: str = DEFAULT_TENANT
    ) -> Sequence[Database]:
        """
        List all databases
        
        Args:
            limit: maximum number of results to return
            offset: number of results to skip
            tenant: tenant name (for OceanBase)
            
        Returns:
            Sequence of Database objects
        """
        pass


class _AdminClientProxy(AdminAPI):
    """
    A lightweight facade that delegates all operations to the underlying ServerAPI (BaseClient).
    The actual logic is in the specific client implementations (Embedded/Server/OceanBase).
    
    Note: This is an internal class. Users should use the AdminClient() factory function.
    """
    
    _server: "BaseClient"
    
    def __init__(self, server: "BaseClient") -> None:
        """
        Initialize admin client with a server implementation
        
        Args:
            server: The underlying client that implements the actual logic
        """
        self._server = server
    
    def create_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """Proxy to server implementation"""
        return self._server.create_database(name=name, tenant=tenant)
    
    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """Proxy to server implementation"""
        return self._server.get_database(name=name, tenant=tenant)
    
    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """Proxy to server implementation"""
        return self._server.delete_database(name=name, tenant=tenant)
    
    def list_databases(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tenant: str = DEFAULT_TENANT
    ) -> Sequence[Database]:
        """Proxy to server implementation"""
        return self._server.list_databases(limit=limit, offset=offset, tenant=tenant)
    
    def __repr__(self):
        return f"<AdminClient server={self._server}>"
    
    def __enter__(self):
        """Context manager support - delegate to server"""
        self._server.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - delegate to server"""
        return self._server.__exit__(exc_type, exc_val, exc_tb)


class _ClientProxy:
    """
    Internal client proxy for collection operations only.
    Strictly separates collection management from database management.
    
    Note: This is an internal class. Users should use the Client() factory function.
    """
    
    _server: "BaseClient"
    
    def __init__(self, server: "BaseClient") -> None:
        """
        Initialize client with a server implementation
        
        Args:
            server: The underlying client that implements the actual logic
        """
        self._server = server
    
    def create_collection(
        self,
        name: str,
        configuration: ConfigurationParam = _PLACEHOLDER,
        embedding_function: EmbeddingFunctionParam = _PLACEHOLDER,
        **kwargs
    ) -> "Collection":
        """Proxy to server implementation - collection operations only"""
        # Replace placeholder with real _NOT_PROVIDED if needed
        real_not_provided = _get_not_provided()
        if configuration is _PLACEHOLDER:
            configuration = real_not_provided
        if embedding_function is _PLACEHOLDER:
            embedding_function = real_not_provided
        return self._server.create_collection(
            name=name,
            configuration=configuration,
            embedding_function=embedding_function,
            **kwargs
        )
    
    def get_collection(
        self,
        name: str,
        embedding_function: EmbeddingFunctionParam = _PLACEHOLDER
    ) -> "Collection":
        """Proxy to server implementation - collection operations only"""
        # Replace placeholder with real _NOT_PROVIDED if needed
        real_not_provided = _get_not_provided()
        if embedding_function is _PLACEHOLDER:
            embedding_function = real_not_provided
        return self._server.get_collection(name=name, embedding_function=embedding_function)
    
    def delete_collection(self, name: str) -> None:
        """Proxy to server implementation - collection operations only"""
        return self._server.delete_collection(name=name)
    
    def list_collections(self) -> List["Collection"]:
        """Proxy to server implementation - collection operations only"""
        return self._server.list_collections()
    
    def has_collection(self, name: str) -> bool:
        """Proxy to server implementation - collection operations only"""
        return self._server.has_collection(name=name)
    
    def get_or_create_collection(
        self,
        name: str,
        configuration: ConfigurationParam = _PLACEHOLDER,
        embedding_function: EmbeddingFunctionParam = _PLACEHOLDER,
        **kwargs
    ) -> "Collection":
        """Proxy to server implementation - collection operations only"""
        # Replace placeholder with real _NOT_PROVIDED if needed
        real_not_provided = _get_not_provided()
        if configuration is _PLACEHOLDER:
            configuration = real_not_provided
        if embedding_function is _PLACEHOLDER:
            embedding_function = real_not_provided
        return self._server.get_or_create_collection(
            name=name,
            configuration=configuration,
            embedding_function=embedding_function,
            **kwargs
        )
    
    def count_collection(self) -> int:
        """Proxy to server implementation - collection operations only"""
        return self._server.count_collection()
    
    def __repr__(self):
        return f"<Client server={self._server}>"
    
    def __enter__(self):
        """Context manager support - delegate to server"""
        self._server.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - delegate to server"""
        return self._server.__exit__(exc_type, exc_val, exc_tb)

