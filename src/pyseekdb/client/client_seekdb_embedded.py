"""
Embedded mode client - based on seekdb
"""
import os
import logging
from typing import Any, List, Optional, Sequence, Dict, Union

import pylibseekdb as seekdb  # type: ignore

from .client_base import BaseClient
from .collection import Collection
from .database import Database
from .admin_client import DEFAULT_TENANT

logger = logging.getLogger(__name__)


class SeekdbEmbeddedClient(BaseClient):
    """Embedded seekdb client (lazy connection)"""
    
    def __init__(
        self,
        path: str = "./seekdb",
        database: str = "test",
        **kwargs
    ):
        """
        Initialize embedded client (no immediate connection)
        
        Args:
            path: seekdb data directory path
            database: database name
        """
        self.path = os.path.abspath(path)
        # Create directory if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            logger.info(f"Created directory: {self.path}")
        elif not os.path.isdir(self.path):
            raise ValueError(f"Path exists but is not a directory: {self.path}")
        
        self.database = database
        self._connection = None
        self._initialized = False
        
        logger.info(f"Initialize SeekdbEmbeddedClient: path={self.path}, database={self.database}")
    
    # ==================== Connection Management ====================
    
    def _ensure_connection(self) -> Any:  # seekdb.Connection
        """Ensure connection is established (internal method)"""
        if not self._initialized:
            
            # 1. open seekdb
            try:
                seekdb.open(db_dir=self.path)  # type: ignore
                logger.info(f"✅ seekdb opened: {self.path}")
            except Exception as e:
                if "initialized twice" not in str(e):
                    raise
                logger.debug(f"seekdb already opened: {e}")
            
            self._initialized = True
        
        # 3. Create connection
        if self._connection is None:
            self._connection = seekdb.connect(  # type: ignore
                database=self.database,
                autocommit=True
            )
            logger.info(f"✅ Connected to database: {self.database}")
        
        return self._connection
    
    def _cleanup(self):
        """Internal cleanup method: close connection)"""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Connection closed")
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connection is not None and self._initialized
    
    def execute(self, sql: str) -> Any:   
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql)
            
            sql_upper = sql.strip().upper()
            if (sql_upper.startswith('SELECT') or 
                sql_upper.startswith('SHOW') or 
                sql_upper.startswith('DESCRIBE') or
                sql_upper.startswith('DESC')):
                return cursor.fetchall()
            
            return cursor
        except Exception:
            raise
    
    def get_raw_connection(self) -> Any:  # seekdb.Connection
        """Get raw connection object"""
        return self._ensure_connection()
    
    @property
    def mode(self) -> str:
        return "SeekdbEmbeddedClient"
    
    def _use_context_manager_for_cursor(self) -> bool:
        """
        Override to use try/finally instead of context manager for cursor
        (seekdb embedded client doesn't support context manager)
        """
        return False
    
    def _execute_query_with_cursor(
        self,
        conn: Any,
        sql: str,
        params: List[Any],
        use_context_manager: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return normalized rows
        Override base class to handle pyseekdb cursor which doesn't support parameterized queries
        
        Args:
            conn: Database connection
            sql: SQL query string with %s placeholders
            params: Query parameters to embed in SQL
            use_context_manager: Whether to use context manager (ignored for embedded client)
            
        Returns:
            List of normalized row dictionaries
        """
        # pyseekdb.Cursor.execute() only accepts SQL string, not parameters
        # Embed parameters directly into SQL
        embedded_sql = sql
        for param in params:
            if param is None:
                embedded_sql = embedded_sql.replace('%s', 'NULL', 1)
            elif isinstance(param, (int, float)):
                embedded_sql = embedded_sql.replace('%s', str(param), 1)
            elif isinstance(param, str):
                escaped = param.replace("'", "''")
                embedded_sql = embedded_sql.replace('%s', f"'{escaped}'", 1)
            else:
                # For other types (like lists in IN clauses), convert to string
                escaped = str(param).replace("'", "''")
                embedded_sql = embedded_sql.replace('%s', f"'{escaped}'", 1)

        cursor = conn.cursor()
        try:
            cursor.execute(embedded_sql)
            
            # Check if this is a query statement (SELECT, SHOW, DESCRIBE, DESC)
            # Only query statements return result sets that need fetchall()
            sql_upper = embedded_sql.strip().upper()
            is_query = (sql_upper.startswith('SELECT') or 
                       sql_upper.startswith('SHOW') or 
                       sql_upper.startswith('DESCRIBE') or
                       sql_upper.startswith('DESC'))
            
            if not is_query:
                # For non-query statements (DELETE, UPDATE, INSERT, etc.), return empty list
                return []
            
            # For query statements, fetch results
            rows = cursor.fetchall()
            
            # pyseekdb.Cursor doesn't have description, extract column names from SQL
            cursor_description = getattr(cursor, 'description', None)
            if cursor_description is None and rows:
                import re
                # Extract column names from SELECT clause using simple regex
                select_match = re.search(r'SELECT\s+(.+?)\s+FROM', embedded_sql, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1).strip()
                    # Split by comma, but skip commas inside parentheses (for function calls)
                    parts = []
                    depth = 0
                    current = ""
                    for char in select_clause:
                        if char == '(':
                            depth += 1
                        elif char == ')':
                            depth -= 1
                        elif char == ',' and depth == 0:
                            parts.append(current.strip())
                            current = ""
                            continue
                        current += char
                    if current:
                        parts.append(current.strip())
                    
                    # Extract column names: look for AS alias, otherwise use column name
                    column_names = []
                    for part in parts:
                        # Match "AS alias" pattern
                        as_match = re.search(r'\s+AS\s+(\w+)', part, re.IGNORECASE)
                        if as_match:
                            column_names.append(as_match.group(1))
                        else:
                            # No alias, extract column name (remove backticks, get identifier)
                            col = part.replace('`', '').strip().split()[-1]
                            column_names.append(col)
                    
                    cursor_description = [(name,) for name in column_names]
            
            normalized_rows = []
            for row in rows:
                normalized_rows.append(self._normalize_row(row, cursor_description))
            return normalized_rows
        finally:
            cursor.close()
    
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
        Create database (tenant parameter ignored for embedded mode)
        
        Args:
            name: database name
            tenant: ignored for embedded mode (no tenant concept)
        """
        logger.info(f"Creating database: {name}")
        sql = f"CREATE DATABASE IF NOT EXISTS `{name}`"
        self.execute(sql)
        logger.info(f"✅ Database created: {name}")
    
    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Get database object (tenant parameter ignored for embedded mode)
        
        Args:
            name: database name
            tenant: ignored for embedded mode (no tenant concept)
        """
        logger.info(f"Getting database: {name}")
        sql = f"SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '{name}'"
        result = self.execute(sql)
        
        if not result:
            raise ValueError(f"Database not found: {name}")
        
        row = result[0]
        return Database(
            name=row[0] if isinstance(row, tuple) else row.get('SCHEMA_NAME'),
            tenant=None,  # No tenant concept in embedded mode
            charset=row[1] if isinstance(row, tuple) else row.get('DEFAULT_CHARACTER_SET_NAME'),
            collation=row[2] if isinstance(row, tuple) else row.get('DEFAULT_COLLATION_NAME')
        )
    
    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Delete database (tenant parameter ignored for embedded mode)
        
        Args:
            name: database name
            tenant: ignored for embedded mode (no tenant concept)
        """
        logger.info(f"Deleting database: {name}")
        sql = f"DROP DATABASE IF EXISTS `{name}`"
        self.execute(sql)
        logger.info(f"✅ Database deleted: {name}")
    
    def list_databases(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tenant: str = DEFAULT_TENANT
    ) -> Sequence[Database]:
        """
        List all databases (tenant parameter ignored for embedded mode)
        
        Args:
            limit: maximum number of results to return
            offset: number of results to skip
            tenant: ignored for embedded mode (no tenant concept)
        """
        logger.info("Listing databases")
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
                name=row[0] if isinstance(row, tuple) else row.get('SCHEMA_NAME'),
                tenant=None,  # No tenant concept in embedded mode
                charset=row[1] if isinstance(row, tuple) else row.get('DEFAULT_CHARACTER_SET_NAME'),
                collation=row[2] if isinstance(row, tuple) else row.get('DEFAULT_COLLATION_NAME')
            ))
        
        logger.info(f"✅ Found {len(databases)} databases")
        return databases
    
    def __repr__(self):
        status = "connected" if self.is_connected() else "disconnected"
        return f"<SeekdbEmbeddedClient path={self.path} database={self.database} status={status}>"
