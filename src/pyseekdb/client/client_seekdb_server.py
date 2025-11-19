"""
Remote server mode client - based on pymysql
Supports both seekdb Server and OceanBase Server
"""
import logging
from typing import Any, List, Optional, Sequence, Dict, Union

import pymysql
from pymysql.cursors import DictCursor

from .client_base import BaseClient
from .collection import Collection
from .database import Database
from .admin_client import DEFAULT_TENANT

logger = logging.getLogger(__name__)


class RemoteServerClient(BaseClient):
    """Remote server mode client (connecting via pymysql, lazy loading)
    
    Supports both seekdb Server and OceanBase Server.
    Uses user@tenant format for authentication.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2881,
        tenant: str = "sys",
        database: str = "test",
        user: str = "root",
        password: str = "",
        charset: str = "utf8mb4",
        **kwargs
    ):
        """
        Initialize remote server mode client (no immediate connection)
        
        Args:
            host: server address
            port: server port (default 2881)
            tenant: tenant name (default "sys" for seekdb Server, "test" for OceanBase)
            database: database name
            user: username (without tenant suffix)
            password: password
            charset: charset (default "utf8mb4")
            **kwargs: other pymysql connection parameters
        """
        self.host = host
        self.port = port
        self.tenant = tenant
        self.database = database
        self.user = user
        self.password = password
        self.charset = charset
        self.kwargs = kwargs
        
        # Remote server username format: user@tenant
        self.full_user = f"{user}@{tenant}"
        self._connection = None
        
        logger.info(
            f"Initialize RemoteServerClient: {self.full_user}@{self.host}:{self.port}/{self.database}"
        )
    
    # ==================== Connection Management ====================
    
    def _ensure_connection(self) -> pymysql.Connection:
        """Ensure connection is established (internal method)"""
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.full_user,  # Remote server format: user@tenant
                password=self.password,
                database=self.database,
                charset=self.charset,
                cursorclass=DictCursor,
                autocommit=True,
                **self.kwargs
            )
            logger.info(f"✅ Connected to remote server: {self.host}:{self.port}/{self.database}")
        
        return self._connection
    
    def _cleanup(self):
        """Internal cleanup method: close connection)"""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Connection closed")
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connection is not None and self._connection.open
    
    def execute(self, sql: str) -> Any:
        conn = self._ensure_connection()
        
        with conn.cursor() as cursor:
            cursor.execute(sql)
            
            sql_upper = sql.strip().upper()
            if (sql_upper.startswith('SELECT') or
                sql_upper.startswith('SHOW') or
                sql_upper.startswith('DESCRIBE') or
                sql_upper.startswith('DESC')):
                return cursor.fetchall()
            
            return cursor
    
    def get_raw_connection(self) -> pymysql.Connection:
        """Get raw connection object"""
        return self._ensure_connection()
    
    @property
    def mode(self) -> str:
        return "RemoteServerClient"
    
    # ==================== Collection Management (framework) ====================
    
    # create_collection is inherited from BaseClient - no override needed
    # get_collection is inherited from BaseClient - no override needed
    # delete_collection is inherited from BaseClient - no override needed
    # list_collections is inherited from BaseClient - no override needed
    # has_collection is inherited from BaseClient - no override needed
    
    # ==================== Collection Internal Operations ====================
    # These methods are called by Collection objects
    
    # -------------------- DML Operations --------------------
    # _collection_add is inherited from BaseClient
    # _collection_update is inherited from BaseClient
    # _collection_upsert is inherited from BaseClient
    # _collection_delete is inherited from BaseClient
    
    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() use base class implementation
    
    # _collection_hybrid_search is inherited from BaseClient
    
    # -------------------- Collection Info --------------------
    
    # _collection_count is inherited from BaseClient - no override needed
    
    # ==================== Database Management ====================
    
    def create_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Create database (remote server has tenant concept, uses client's tenant)
        
        Args:
            name: database name
            tenant: tenant name (if different from client tenant, will use client tenant)
        
        Note:
            Remote server has multi-tenant architecture. Database is scoped to client's tenant.
        """
        if tenant != self.tenant and tenant != DEFAULT_TENANT:
            logger.warning(f"Specified tenant '{tenant}' differs from client tenant '{self.tenant}', using client tenant")
        
        logger.info(f"Creating database: {name} in tenant: {self.tenant}")
        sql = f"CREATE DATABASE IF NOT EXISTS `{name}`"
        self.execute(sql)
        logger.info(f"✅ Database created: {name} in tenant: {self.tenant}")
    
    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Get database object (remote server has tenant concept, uses client's tenant)
        
        Args:
            name: database name
            tenant: tenant name (if different from client tenant, will use client tenant)
        
        Returns:
            Database object with tenant information
        
        Note:
            Remote server has multi-tenant architecture. Database is scoped to client's tenant.
        """
        if tenant != self.tenant and tenant != DEFAULT_TENANT:
            logger.warning(f"Specified tenant '{tenant}' differs from client tenant '{self.tenant}', using client tenant")
        
        logger.info(f"Getting database: {name} in tenant: {self.tenant}")
        sql = f"SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '{name}'"
        result = self.execute(sql)
        
        if not result:
            raise ValueError(f"Database not found: {name}")
        
        row = result[0]
        return Database(
            name=row['SCHEMA_NAME'],
            tenant=self.tenant,  # Remote server has tenant concept
            charset=row['DEFAULT_CHARACTER_SET_NAME'],
            collation=row['DEFAULT_COLLATION_NAME']
        )
    
    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Delete database (remote server has tenant concept, uses client's tenant)
        
        Args:
            name: database name
            tenant: tenant name (if different from client tenant, will use client tenant)
        
        Note:
            Remote server has multi-tenant architecture. Database is scoped to client's tenant.
        """
        if tenant != self.tenant and tenant != DEFAULT_TENANT:
            logger.warning(f"Specified tenant '{tenant}' differs from client tenant '{self.tenant}', using client tenant")
        
        logger.info(f"Deleting database: {name} in tenant: {self.tenant}")
        sql = f"DROP DATABASE IF EXISTS `{name}`"
        self.execute(sql)
        logger.info(f"✅ Database deleted: {name} in tenant: {self.tenant}")
    
    def list_databases(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tenant: str = DEFAULT_TENANT
    ) -> Sequence[Database]:
        """
        List all databases (remote server has tenant concept, uses client's tenant)
        
        Args:
            limit: maximum number of results to return
            offset: number of results to skip
            tenant: tenant name (if different from client tenant, will use client tenant)
        
        Returns:
            Sequence of Database objects with tenant information
        
        Note:
            Remote server has multi-tenant architecture. Lists databases in client's tenant.
        """
        if tenant != self.tenant and tenant != DEFAULT_TENANT:
            logger.warning(f"Specified tenant '{tenant}' differs from client tenant '{self.tenant}', using client tenant")
        
        logger.info(f"Listing databases in tenant: {self.tenant}")
        sql = "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA"
        
        if limit is not None:
            if offset is not None:
                sql += f" LIMIT {offset}, {limit}"
            else:
                sql += f" LIMIT {limit}"
        
        result = self.execute(sql)
        
        databases = []
        for row in result:
            databases.append(Database(
                name=row['SCHEMA_NAME'],
                tenant=self.tenant,  # Remote server has tenant concept
                charset=row['DEFAULT_CHARACTER_SET_NAME'],
                collation=row['DEFAULT_COLLATION_NAME']
            ))
        
        logger.info(f"✅ Found {len(databases)} databases in tenant {self.tenant}")
        return databases
    
    def __repr__(self):
        status = "connected" if self.is_connected() else "disconnected"
        return f"<RemoteServerClient {self.full_user}@{self.host}:{self.port}/{self.database} status={status}>"
