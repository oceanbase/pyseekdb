"""
Base client interface definition
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Dict, Any, Union, TYPE_CHECKING, Tuple, Callable
from dataclasses import dataclass

from .base_connection import BaseConnection
from .admin_client import AdminAPI, DEFAULT_TENANT
from .meta_info import CollectionNames, CollectionFieldNames
from .query_result import QueryResult
from .filters import FilterBuilder
from .embedding_function import (
    EmbeddingFunction,
    DefaultEmbeddingFunction,
    get_default_embedding_function,
    Documents as EmbeddingDocuments,
    Embeddings as EmbeddingVectors
)

from .collection import Collection

from .database import Database

logger = logging.getLogger(__name__)

# Default configuration constants
# Note: Default embedding function (DefaultEmbeddingFunction) produces 384-dim vectors
# So we use 384 as the default dimension to match
DEFAULT_VECTOR_DIMENSION = 384  # Matches DefaultEmbeddingFunction dimension
DEFAULT_DISTANCE_METRIC = 'cosine'

# Sentinel object to distinguish between "parameter not provided" and "explicitly set to None"
class _NotProvided:
    """Sentinel object to indicate a parameter was not provided"""
    pass

_NOT_PROVIDED = _NotProvided()

# Type alias for embedding_function parameter that can be EmbeddingFunction, None, or sentinel
EmbeddingFunctionParam = Union[EmbeddingFunction[EmbeddingDocuments], None, Any]


@dataclass
class HNSWConfiguration:
    """
    HNSW (Hierarchical Navigable Small World) index configuration
    
    Args:
        dimension: Vector dimension (number of elements in each vector)
        distance: Distance metric for similarity calculation (e.g., 'l2', 'cosine', 'inner_product')
    """
    dimension: int
    distance: str = 'l2'
    
    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")
        if self.distance not in ['l2', 'cosine', 'inner_product']:
            raise ValueError(f"distance must be one of ['l2', 'cosine', 'inner_product'], got {self.distance}")

# Type alias for configuration parameter that can be HNSWConfiguration, None, or sentinel
ConfigurationParam = Union[HNSWConfiguration, None, Any]

class ClientAPI(ABC):
    """
    Client API interface for collection operations only.
    This is what end users interact with through the Client proxy.
    """
    
    @abstractmethod
    def create_collection(
        self,
        name: str,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        **kwargs
    ) -> "Collection":
        """
        Create collection
        
        Args:
            name: Collection name
            configuration: HNSW index configuration (HNSWConfiguration)
            embedding_function: Embedding function to convert documents to vectors.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.
                               If provided, the dimension in configuration should match the
                               embedding function's output dimension.
            **kwargs: Additional parameters
        """
        pass
    
    @abstractmethod
    def get_collection(
        self,
        name: str,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED
    ) -> "Collection":
        """
        Get collection object
        
        Args:
            name: Collection name
            embedding_function: Embedding function to convert documents to vectors.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.
        """
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete collection"""
        pass
    
    @abstractmethod
    def list_collections(self) -> List["Collection"]:
        """List all collections"""
        pass
    
    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """Check if collection exists"""
        pass


class BaseClient(BaseConnection, AdminAPI):
    """
    Abstract base class for all clients.
    
    Design Pattern:
    1. Provides public collection management methods (create_collection, get_collection, etc.)
    2. Defines internal operation interfaces (_collection_* methods) called by Collection objects
    3. Subclasses implement all abstract methods to provide specific business logic
    
    Benefits of this design:
    - Collection object interface is unified regardless of which client created it
    - Different clients can have completely different underlying implementations (SQL/gRPC/REST)
    - Easy to extend with new client types
    
    Inherits connection management from BaseConnection and database operations from AdminAPI.
    """
    
    # ==================== Collection Management (User-facing) ====================
    
    def create_collection(
        self,
        name: str,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        **kwargs
    ) -> "Collection":
        """
        Create a collection (user-facing API)
        
        Args:
            name: Collection name
            configuration: HNSW index configuration (HNSWConfiguration)
                          If not provided, uses default configuration (dimension=384, distance='cosine').
                          If explicitly set to None, will try to calculate dimension from embedding_function.
                          If embedding_function is also None, will raise an error.
            embedding_function: Embedding function to convert documents to vectors.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.
                               If provided, the actual dimension will be calculated by calling
                               embedding_function.__call__("seekdb"), and this dimension will be used
                               to create the table. If configuration.dimension is set and doesn't match
                               the calculated dimension, a ValueError will be raised.
            **kwargs: Additional parameters 
            
        Returns:
            Collection object
            
        Raises:
            ValueError: If configuration is explicitly set to None and embedding_function is also None
                       (cannot determine dimension), or if embedding_function is provided and
                       configuration.dimension doesn't match the calculated dimension from embedding_function
            
        Examples:
            # Using default configuration and default embedding function
            >>> collection = client.create_collection('my_collection')
            
            # Using custom embedding function (dimension will be calculated automatically)
            >>> from pyseekdb import DefaultEmbeddingFunction
            >>> ef = DefaultEmbeddingFunction(model_name='all-MiniLM-L6-v2')
            >>> config = HNSWConfiguration(dimension=384, distance='cosine')  # Must match EF dimension
            >>> collection = client.create_collection(
            ...     'my_collection',
            ...     configuration=config,
            ...     embedding_function=ef
            ... )
            
            # Explicitly set configuration=None, use embedding function to determine dimension
            >>> collection = client.create_collection('my_collection', configuration=None, embedding_function=ef)
            
            # Explicitly disable embedding function (use configuration dimension)
            >>> config = HNSWConfiguration(dimension=128, distance='cosine')
            >>> collection = client.create_collection('my_collection', configuration=config, embedding_function=None)
        """
        # Handle embedding function first
        # If not provided (sentinel), use default embedding function
        if embedding_function is _NOT_PROVIDED:
            embedding_function = get_default_embedding_function()
        
        # Calculate actual dimension from embedding function if provided
        actual_dimension = None
        if embedding_function is not None:
            try:
                # Call embedding function with "seekdb" to get actual dimension
                test_embeddings = embedding_function.__call__("seekdb")
                if test_embeddings and len(test_embeddings) > 0:
                    actual_dimension = len(test_embeddings[0])
                    logger.info(f"Calculated embedding function dimension: {actual_dimension}")
                else:
                    raise ValueError("Embedding function returned empty result when called with 'seekdb'")
            except Exception as e:
                raise ValueError(
                    f"Failed to calculate dimension from embedding function: {e}. "
                    f"Please ensure the embedding function can be called with a string input."
                ) from e
        
        # Handle configuration
        # If not provided (sentinel), use default configuration
        if configuration is _NOT_PROVIDED:
            # Use default configuration, but if embedding_function is provided, use its dimension
            if actual_dimension is not None:
                configuration = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
            else:
                configuration = HNSWConfiguration(dimension=DEFAULT_VECTOR_DIMENSION, distance=DEFAULT_DISTANCE_METRIC)
        elif configuration is None:
            # Configuration is explicitly set to None
            # Try to calculate dimension from embedding_function
            if embedding_function is None:
                raise ValueError(
                    "Cannot create collection: configuration is explicitly set to None and "
                    "embedding_function is also None. Cannot determine dimension without either a configuration "
                    "or an embedding function. Please either:\n"
                    "  1. Provide a configuration with dimension specified (e.g., HNSWConfiguration(dimension=128, distance='cosine')), or\n"
                    "  2. Provide an embedding_function to calculate dimension automatically, or\n"
                    "  3. Do not set configuration=None (use default configuration)."
                )
            
            # Use calculated dimension from embedding function and default distance metric
            if actual_dimension is not None:
                configuration = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
            else:
                raise ValueError(
                    "Failed to calculate dimension from embedding function. "
                    "Please ensure the embedding function can be called with a string input."
                )
        
        # Validate configuration type
        if not isinstance(configuration, HNSWConfiguration):
            raise TypeError(f"configuration must be HNSWConfiguration, got {type(configuration)}")
        
        # If embedding_function is provided, validate configuration dimension matches
        if embedding_function is not None and actual_dimension is not None:
            if configuration.dimension != actual_dimension:
                raise ValueError(
                    f"Configuration dimension ({configuration.dimension}) doesn't match "
                    f"embedding function dimension ({actual_dimension}). "
                    f"Please update configuration to use dimension={actual_dimension} or remove dimension from configuration."
                )
            # Use actual dimension from embedding function
            dimension = actual_dimension
        else:
            # No embedding function, use configuration dimension
            dimension = configuration.dimension
        
        # Extract distance from configuration
        distance = configuration.distance
        
        # HNSW is the only supported index type
        index_type = 'hnsw'
        
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Construct CREATE TABLE SQL statement with HEAP organization
        sql = f"""CREATE TABLE `{table_name}` (
            _id varbinary(512) PRIMARY KEY NOT NULL,
            document string,
            embedding vector({dimension}),
            metadata json,
            FULLTEXT INDEX idx_fts(document),
            VECTOR INDEX idx_vec (embedding) with(distance={distance}, type={index_type}, lib=vsag)
        ) ORGANIZATION = HEAP;"""
        
        # Execute SQL to create table
        self.execute(sql)
        
        # Store embedding function in kwargs to pass to Collection
        kwargs['embedding_function'] = embedding_function
        
        # Create and return Collection object
        return Collection(client=self, name=name, dimension=dimension, **kwargs)
    
    def get_collection(
        self,
        name: str,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED
    ) -> "Collection":
        """
        Get a collection object (user-facing API)
        
        Args:
            name: Collection name
            embedding_function: Embedding function to convert documents to vectors.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.
            
        Returns:
            Collection object
            
        Raises:
            ValueError: If collection does not exist
        """
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Check if table exists by describing it
        try:
            table_info = self.execute(f"DESCRIBE `{table_name}`")
            if not table_info or len(table_info) == 0:
                raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)")
        except Exception as e:
            # If DESCRIBE fails, check if it's because table doesn't exist
            error_msg = str(e).lower()
            if "doesn't exist" in error_msg or "not found" in error_msg or "table" in error_msg:
                raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)") from e
            raise
        
        # Extract dimension from embedding column
        dimension = None
        for row in table_info:
            # Handle both dict and tuple formats
            if isinstance(row, dict):
                field_name = row.get('Field', row.get('field', ''))
                field_type = row.get('Type', row.get('type', ''))
            elif isinstance(row, (tuple, list)):
                field_name = row[0] if len(row) > 0 else ''
                field_type = row[1] if len(row) > 1 else ''
            else:
                continue
            
            if field_name == 'embedding' and 'vector' in str(field_type).lower():
                # Extract dimension from vector(dimension) format
                match = re.search(r'vector\s*\(\s*(\d+)\s*\)', str(field_type), re.IGNORECASE)
                if match:
                    dimension = int(match.group(1))
                break
        
        # Handle embedding function
        # If not provided (sentinel), use default embedding function
        if embedding_function is _NOT_PROVIDED:
            embedding_function = get_default_embedding_function()
        
        # Create and return Collection object
        return Collection(client=self, name=name, dimension=dimension, embedding_function=embedding_function)
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection (user-facing API)
        
        Args:
            name: Collection name
            
        Raises:
            ValueError: If collection does not exist
        """
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Check if table exists first
        if not self.has_collection(name):
            raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)")
        
        # Execute DROP TABLE SQL
        self.execute(f"DROP TABLE IF EXISTS `{table_name}`")
    
    def list_collections(self) -> List["Collection"]:
        """
        List all collections (user-facing API)
        
        Returns:
            List of Collection objects
        """
        # List all tables that start with 'c$v1'
        # Use SHOW TABLES LIKE 'c$v1%' to filter collection tables
        try:
            tables = self.execute("SHOW TABLES LIKE 'c$v1$%'")
        except Exception:
            # Fallback: try to query information_schema
            try:
                # Get current database name
                db_result = self.execute("SELECT DATABASE()")
                if db_result and len(db_result) > 0:
                    db_name = db_result[0][0] if isinstance(db_result[0], (tuple, list)) else db_result[0].get('DATABASE()', '')
                    tables = self.execute(
                        f"SELECT TABLE_NAME FROM information_schema.TABLES "
                        f"WHERE TABLE_SCHEMA = '{db_name}' AND TABLE_NAME LIKE 'c$v1$%'"
                    )
                else:
                    return []
            except Exception:
                return []
        
        collections = []
        for row in tables:
            # Extract table name
            if isinstance(row, dict):
                # Server client returns dict, get the first value
                table_name = list(row.values())[0] if row else ''
            elif isinstance(row, (tuple, list)):
                # Embedded client returns tuple, first element is table name
                table_name = row[0] if len(row) > 0 else ''
            else:
                table_name = str(row)
            
            # Extract collection name from table name (remove 'c$v1$' prefix)
            if table_name.startswith('c$v1$'):
                collection_name = table_name[5:]  # Remove 'c$v1$' prefix
                
                # Get collection with dimension
                try:
                    collection = self.get_collection(collection_name)
                    collections.append(collection)
                except Exception:
                    # Skip if we can't get collection info
                    continue
        
        return collections
    
    def count_collection(self) -> int:
        """
        Count the number of collections in the current database
        
        Returns:
            Number of collections
            
        Examples:
            count = client.count_collection()
            print(f"Database has {count} collections")
        """
        collections = self.list_collections()
        return len(collections)
    
    def has_collection(self, name: str) -> bool:
        """
        Check if a collection exists (user-facing API)
        
        Args:
            name: Collection name
            
        Returns:
            True if exists, False otherwise
        """
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Check if table exists
        try:
            # Try to describe the table
            table_info = self.execute(f"DESCRIBE `{table_name}`")
            return table_info is not None and len(table_info) > 0
        except Exception:
            # If DESCRIBE fails, table doesn't exist
            return False
    
    def get_or_create_collection(
        self,
        name: str,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        **kwargs
    ) -> "Collection":
        """
        Get an existing collection or create it if it doesn't exist (user-facing API)
        
        Args:
            name: Collection name
            configuration: HNSW index configuration (HNSWConfiguration)
                          If not provided, uses default configuration (dimension=384, distance='cosine').
                          If explicitly set to None, will try to calculate dimension from embedding_function.
                          If embedding_function is also None, will raise an error.
            embedding_function: Embedding function to convert documents to vectors.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.
                               If provided when creating a new collection, the actual dimension will be
                               calculated by calling embedding_function.__call__("seekdb"), and this
                               dimension will be used to create the table. If configuration.dimension is
                               set and doesn't match the calculated dimension, a ValueError will be raised.
            **kwargs: Additional parameters for create_collection
            
        Returns:
            Collection object
            
        Raises:
            ValueError: If creating a new collection and configuration is explicitly set to None and
                       embedding_function is also None (cannot determine dimension), or if embedding_function
                       is provided and configuration.dimension doesn't match the calculated dimension
        """
        # First, try to get the collection
        if self.has_collection(name):
            # Collection exists, return it
            # Pass embedding_function (could be _NOT_PROVIDED, None, or an EmbeddingFunction instance)
            return self.get_collection(name, embedding_function=embedding_function)
        
        # Collection doesn't exist, create it with provided or default configuration
        return self.create_collection(
            name=name,
            configuration=configuration,
            embedding_function=embedding_function,
            **kwargs
        )
    
    # ==================== Collection Internal Operations (Called by Collection) ====================
    # These methods are called by Collection objects, different clients implement different logic
    
    # -------------------- DML Operations --------------------
    
    def _collection_add(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Add data to collection - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            vectors: Single vector or list of vectors (optional)
            metadatas: Single metadata dict or list of metadata dicts (optional)
            documents: Single document or list of documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to vectors.
                               Required if documents provided but vectors not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.info(f"Adding data to collection '{collection_name}'")
        
        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if vectors is not None:
            if isinstance(vectors, list) and len(vectors) > 0 and not isinstance(vectors[0], list):
                vectors = [vectors]
        
        # Handle vector generation logic:
        # 1. If vectors are provided, use them directly without embedding
        # 2. If vectors are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate vectors from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither vectors nor documents are provided, raise an error
        
        if vectors:
            # Vectors provided, use them directly without embedding
            pass
        elif documents:
            # Vectors not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.info(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    vectors = embedding_function(documents)
                    logger.info(f"✅ Successfully generated {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {e}")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}")
            else:
                raise ValueError(
                    "Documents provided but no vectors and no embedding function. "
                    "Either:\n"
                    "  1. Provide vectors directly when calling add(), or\n"
                    "  2. Provide embedding_function to auto-generate vectors from documents."
                )
        else:
            # Neither vectors nor documents provided, raise an error
            raise ValueError(
                "Neither vectors nor documents provided. "
                "Please provide either:\n"
                "  1. vectors directly, or\n"
                "  2. documents with embedding_function to generate vectors."
            )
        
        # Determine number of items
        num_items = 0
        if ids:
            num_items = len(ids)
        elif documents:
            num_items = len(documents)
        elif vectors:
            num_items = len(vectors)
        elif metadatas:
            num_items = len(metadatas)
        
        if num_items == 0:
            raise ValueError("No items to add")
        
        # Validate lengths match
        if ids and len(ids) != num_items:
            raise ValueError(f"Number of ids ({len(ids)}) does not match number of items ({num_items})")
        if documents and len(documents) != num_items:
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of items ({num_items})")
        if metadatas and len(metadatas) != num_items:
            raise ValueError(f"Number of metadatas ({len(metadatas)}) does not match number of items ({num_items})")
        if vectors and len(vectors) != num_items:
            raise ValueError(f"Number of vectors ({len(vectors)}) does not match number of items ({num_items})")
        
        # Get table name
        table_name = CollectionNames.table_name(collection_name)
        
        # Build INSERT SQL
        values_list = []
        for i in range(num_items):
            # Process ID - support any string format
            id_val = ids[i] if ids else None
            if id_val:
                if not isinstance(id_val, str):
                    id_val = str(id_val)
                id_sql = self._convert_id_to_sql(id_val)
            else:
                raise ValueError("ids must be provided for add operation")
            
            # Process document
            doc_val = documents[i] if documents else None
            if doc_val is not None:
                # Escape single quotes
                doc_val_escaped = doc_val.replace("'", "''")
                doc_sql = f"'{doc_val_escaped}'"
            else:
                doc_sql = "NULL"
            
            # Process metadata
            meta_val = metadatas[i] if metadatas else None
            if meta_val is not None:
                # Convert to JSON string and escape
                meta_json = json.dumps(meta_val, ensure_ascii=False)
                meta_json_escaped = meta_json.replace("'", "''")
                meta_sql = f"'{meta_json_escaped}'"
            else:
                meta_sql = "NULL"
            
            # Process vector
            vec_val = vectors[i] if vectors else None
            if vec_val is not None:
                # Convert vector to string format: [1.0,2.0,3.0]
                vec_str = "[" + ",".join(map(str, vec_val)) + "]"
                vec_sql = f"'{vec_str}'"
            else:
                vec_sql = "NULL"
            
            values_list.append(f"({id_sql}, {doc_sql}, {meta_sql}, {vec_sql})")
        
        # Build final SQL
        sql = f"""INSERT INTO `{table_name}` ({CollectionFieldNames.ID}, {CollectionFieldNames.DOCUMENT}, {CollectionFieldNames.METADATA}, {CollectionFieldNames.EMBEDDING}) 
                 VALUES {','.join(values_list)}"""
        
        logger.debug(f"Executing SQL: {sql}")
        self.execute(sql)
        logger.info(f"✅ Successfully added {num_items} item(s) to collection '{collection_name}'")
    
    def _collection_update(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Update data in collection - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to update
            vectors: New vectors (optional)
            metadatas: New metadata (optional)
            documents: New documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to vectors.
                               Required if documents provided but vectors not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.info(f"Updating data in collection '{collection_name}'")
        
        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if vectors is not None:
            if isinstance(vectors, list) and len(vectors) > 0 and not isinstance(vectors[0], list):
                vectors = [vectors]
        
        # Handle vector generation logic:
        # 1. If vectors are provided, use them directly without embedding
        # 2. If vectors are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate vectors from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither vectors nor documents are provided:
        #    - If metadatas are provided, allow update (metadata-only update)
        #    - If metadatas are not provided, raise an error
        
        if vectors:
            # Vectors provided, use them directly without embedding
            pass
        elif documents:
            # Vectors not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.info(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    vectors = embedding_function(documents)
                    logger.info(f"✅ Successfully generated {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {e}")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}")
            else:
                raise ValueError(
                    "Documents provided but no vectors and no embedding function. "
                    "Either:\n"
                    "  1. Provide vectors directly when calling update(), or\n"
                    "  2. Provide embedding_function to auto-generate vectors from documents."
                )
        elif not metadatas:
            # Neither vectors nor documents nor metadatas provided, raise an error
            raise ValueError(
                "Neither vectors nor documents nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. vectors directly, or\n"
                "  2. documents with embedding_function to generate vectors, or\n"
                "  3. metadatas to update metadata only."
            )
        
        # Validate inputs
        if not ids:
            raise ValueError("ids must not be empty")
        
        # Validate lengths match
        if documents and len(documents) != len(ids):
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of ids ({len(ids)})")
        if metadatas and len(metadatas) != len(ids):
            raise ValueError(f"Number of metadatas ({len(metadatas)}) does not match number of ids ({len(ids)})")
        if vectors and len(vectors) != len(ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) does not match number of ids ({len(ids)})")
        
        # Get table name
        table_name = CollectionNames.table_name(collection_name)
        
        # Update each item
        for i in range(len(ids)):
            # Process ID - support any string format
            id_val = ids[i]
            if not isinstance(id_val, str):
                id_val = str(id_val)
            id_sql = self._convert_id_to_sql(id_val)
            
            # Build SET clause
            set_clauses = []
            
            if documents:
                doc_val = documents[i]
                if doc_val is not None:
                    doc_val_escaped = doc_val.replace("'", "''")
                    set_clauses.append(f"{CollectionFieldNames.DOCUMENT} = '{doc_val_escaped}'")
            
            if metadatas:
                meta_val = metadatas[i]
                if meta_val is not None:
                    meta_json = json.dumps(meta_val, ensure_ascii=False)
                    meta_json_escaped = meta_json.replace("'", "''")
                    set_clauses.append(f"{CollectionFieldNames.METADATA} = '{meta_json_escaped}'")
            
            if vectors:
                vec_val = vectors[i]
                if vec_val is not None:
                    vec_str = "[" + ",".join(map(str, vec_val)) + "]"
                    set_clauses.append(f"{CollectionFieldNames.EMBEDDING} = '{vec_str}'")
            
            if not set_clauses:
                continue
            
            # Build UPDATE SQL
            sql = f"UPDATE `{table_name}` SET {', '.join(set_clauses)} WHERE {CollectionFieldNames.ID} = {id_sql}"
            
            logger.debug(f"Executing SQL: {sql}")
            self.execute(sql)
        
        logger.info(f"✅ Successfully updated {len(ids)} item(s) in collection '{collection_name}'")
    
    def _collection_upsert(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Insert or update data in collection - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            vectors: Vectors (optional)
            metadatas: Metadata (optional)
            documents: Documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to vectors.
                               Required if documents provided but vectors not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.info(f"Upserting data in collection '{collection_name}'")
        
        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if vectors is not None:
            if isinstance(vectors, list) and len(vectors) > 0 and not isinstance(vectors[0], list):
                vectors = [vectors]
        
        # Handle vector generation logic:
        # 1. If vectors are provided, use them directly without embedding
        # 2. If vectors are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate vectors from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither vectors nor documents are provided:
        #    - If metadatas are provided, allow upsert (metadata-only upsert)
        #    - If metadatas are not provided, raise an error
        
        if vectors:
            # Vectors provided, use them directly without embedding
            pass
        elif documents:
            # Vectors not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.info(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    vectors = embedding_function(documents)
                    logger.info(f"✅ Successfully generated {len(vectors)} embeddings")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {e}")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}")
            else:
                raise ValueError(
                    "Documents provided but no vectors and no embedding function. "
                    "Either:\n"
                    "  1. Provide vectors directly when calling upsert(), or\n"
                    "  2. Provide embedding_function to auto-generate vectors from documents."
                )
        elif not metadatas:
            # Neither vectors nor documents nor metadatas provided, raise an error
            raise ValueError(
                "Neither vectors nor documents nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. vectors directly, or\n"
                "  2. documents with embedding_function to generate vectors, or\n"
                "  3. metadatas to update metadata only."
            )
        
        # Validate inputs
        if not ids:
            raise ValueError("ids must not be empty")
        
        # Validate lengths match
        if documents and len(documents) != len(ids):
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of ids ({len(ids)})")
        if metadatas and len(metadatas) != len(ids):
            raise ValueError(f"Number of metadatas ({len(metadatas)}) does not match number of ids ({len(ids)})")
        if vectors and len(vectors) != len(ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) does not match number of ids ({len(ids)})")
        
        # Get table name
        table_name = CollectionNames.table_name(collection_name)
        
        # Upsert each item
        for i in range(len(ids)):
            # Process ID - support any string format
            id_val = ids[i]
            if not isinstance(id_val, str):
                id_val = str(id_val)
            id_sql = self._convert_id_to_sql(id_val)
            
            # Check if record exists
            existing = self._collection_get(
                collection_id=collection_id,
                collection_name=collection_name,
                ids=[ids[i]],  # Use original string ID for query
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Get values for this item
            doc_val = documents[i] if documents else None
            meta_val = metadatas[i] if metadatas else None
            vec_val = vectors[i] if vectors else None
            
            if existing and len(existing) > 0:
                # Update existing record - only update provided fields
                existing_item = existing[0]
                existing_doc = existing_item.document if hasattr(existing_item, 'document') else None
                existing_meta = existing_item.metadata if hasattr(existing_item, 'metadata') else None
                existing_vec = existing_item.embedding if hasattr(existing_item, 'embedding') else None
                
                # Use provided values or keep existing values
                final_document = doc_val if doc_val is not None else existing_doc
                final_metadata = meta_val if meta_val is not None else existing_meta
                final_vector = vec_val if vec_val is not None else existing_vec
                
                # Build SET clause
                set_clauses = []
                
                if doc_val is not None:
                    doc_val_escaped = final_document.replace("'", "''") if final_document else "NULL"
                    set_clauses.append(f"{CollectionFieldNames.DOCUMENT} = '{doc_val_escaped}'")
                
                if meta_val is not None:
                    meta_json = json.dumps(final_metadata, ensure_ascii=False) if final_metadata else "{}"
                    meta_json_escaped = meta_json.replace("'", "''")
                    set_clauses.append(f"{CollectionFieldNames.METADATA} = '{meta_json_escaped}'")
                
                if vec_val is not None:
                    vec_str = "[" + ",".join(map(str, final_vector)) + "]" if final_vector else "NULL"
                    set_clauses.append(f"{CollectionFieldNames.EMBEDDING} = '{vec_str}'")
                
                if set_clauses:
                    sql = f"UPDATE `{table_name}` SET {', '.join(set_clauses)} WHERE {CollectionFieldNames.ID} = {id_sql}"
                    logger.debug(f"Executing SQL: {sql}")
                    self.execute(sql)
            else:
                # Insert new record
                if doc_val:
                    doc_val_escaped = doc_val.replace("'", "''")
                    doc_sql = f"'{doc_val_escaped}'"
                else:
                    doc_sql = "NULL"
                
                if meta_val is not None:
                    meta_json = json.dumps(meta_val, ensure_ascii=False)
                    meta_json_escaped = meta_json.replace("'", "''")
                    meta_sql = f"'{meta_json_escaped}'"
                else:
                    meta_sql = "NULL"
                
                if vec_val is not None:
                    vec_str = "[" + ",".join(map(str, vec_val)) + "]"
                    vec_sql = f"'{vec_str}'"
                else:
                    vec_sql = "NULL"
                
                sql = f"""INSERT INTO `{table_name}` ({CollectionFieldNames.ID}, {CollectionFieldNames.DOCUMENT}, {CollectionFieldNames.METADATA}, {CollectionFieldNames.EMBEDDING}) 
                         VALUES ({id_sql}, {doc_sql}, {meta_sql}, {vec_sql})"""
                logger.debug(f"Executing SQL: {sql}")
                self.execute(sql)
        
        logger.info(f"✅ Successfully upserted {len(ids)} item(s) in collection '{collection_name}'")
    
    def _collection_delete(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Delete data from collection - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to delete (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            **kwargs: Additional parameters
        """
        logger.info(f"Deleting data from collection '{collection_name}'")
        
        # Validate that at least one filter is provided
        if not ids and not where and not where_document:
            raise ValueError("At least one of ids, where, or where_document must be provided")
        
        # Normalize ids to list
        id_list = None
        if ids is not None:
            if isinstance(ids, str):
                id_list = [ids]
            else:
                id_list = ids
        
        # Get table name
        table_name = CollectionNames.table_name(collection_name)
        
        # Build WHERE clause
        where_clause, params = self._build_where_clause(where, where_document, id_list)
        
        # Build DELETE SQL
        sql = f"DELETE FROM `{table_name}` {where_clause}"
        
        logger.debug(f"Executing SQL: {sql}")
        logger.debug(f"Parameters: {params}")
        
        # Execute DELETE using parameterized query
        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()
        self._execute_query_with_cursor(conn, sql, params, use_context_manager)
        
        logger.info(f"✅ Successfully deleted data from collection '{collection_name}'")
    
    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() are implemented below with common SQL-based logic
    
    def _normalize_query_vectors(
        self,
        query_embeddings: Optional[Union[List[float], List[List[float]]]]
    ) -> List[List[float]]:
        """
        Normalize query vectors to list of lists format
        
        Args:
            query_embeddings: Single vector or list of vectors
            
        Returns:
            List of vectors (each vector is a list of floats)
        """
        if query_embeddings is None:
            return []
        
        # Check if it's a single vector (list of numbers)
        if query_embeddings and isinstance(query_embeddings[0], (int, float)):
            return [query_embeddings]
        
        return query_embeddings
    
    def _normalize_include_fields(
        self,
        include: Optional[List[str]]
    ) -> Dict[str, bool]:
        """
        Normalize include parameter to a dictionary
        
        Args:
            include: List of fields to include (e.g., ["documents", "metadatas", "embeddings"])
            
        Returns:
            Dictionary with field names as keys and True as values
            Default includes: documents, metadatas (but not embeddings)
        """
        # Default includes documents and metadatas
        default_fields = {"documents": True, "metadatas": True}
        
        if include is None:
            return default_fields
        
        # Build include dict from list
        include_dict = {}
        for field in include:
            include_dict[field] = True
        
        return include_dict
    
    def _embed_texts(
        self,
        texts: Union[str, List[str]],
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Embed text(s) to vector(s)
        
        Args:
            texts: Single text or list of texts
            embedding_function: EmbeddingFunction instance to convert texts to vectors.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
                               If not provided, raises NotImplementedError.
            **kwargs: Additional parameters for embedding (unused for now)
            
        Returns:
            List of vectors (List[List[float]]), where each inner list is an embedding vector
            
        Raises:
            NotImplementedError: If embedding_function is not provided
        """
        if embedding_function is None:
            raise NotImplementedError(
                "Text embedding is not implemented. "
                "Please provide query_embeddings directly or set embedding_function in collection."
            )
        
        # Normalize texts to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Use embedding function to generate embeddings
        return embedding_function(texts)
    
    def _normalize_row(self, row: Any, cursor_description: Optional[Any] = None) -> Dict[str, Any]:
        """
        Normalize database row to dictionary format
        
        Args:
            row: Database row (can be dict or tuple)
            cursor_description: Cursor description for tuple rows
            
        Returns:
            Dictionary with column names as keys
        """
        if isinstance(row, dict):
            return row
        
        # Convert tuple to dict using cursor description
        if cursor_description is not None:
            row_dict = {}
            for idx, col_desc in enumerate(cursor_description):
                row_dict[col_desc[0]] = row[idx]
            return row_dict
        
        # Fallback: assume it's already a dict or try to convert
        return dict(row) if hasattr(row, '_asdict') else row
    
    def _execute_query_with_cursor(
        self,
        conn: Any,
        sql: str,
        params: List[Any],
        use_context_manager: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return normalized rows
        
        Args:
            conn: Database connection
            sql: SQL query string
            params: Query parameters
            use_context_manager: Whether to use context manager for cursor (default: True)
            
        Returns:
            List of normalized row dictionaries
        """
        if use_context_manager:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                # Normalize rows
                normalized_rows = []
                for row in rows:
                    normalized_rows.append(self._normalize_row(row, cursor.description))
                return normalized_rows
        else:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                # Normalize rows
                normalized_rows = []
                for row in rows:
                    normalized_rows.append(self._normalize_row(row, cursor.description))
                return normalized_rows
            finally:
                cursor.close()
    
    def _build_select_clause(self, include_fields: Dict[str, bool]) -> str:
        """
        Build SELECT clause based on include fields
        
        Args:
            include_fields: Dictionary of fields to include
            
        Returns:
            SELECT clause string
        """
        select_fields = ["_id"]
        if include_fields.get("embeddings") or include_fields.get("embedding"):
            select_fields.append("embedding")
        if include_fields.get("documents") or include_fields.get("document"):
            select_fields.append("document")
        if include_fields.get("metadatas") or include_fields.get("metadata"):
            select_fields.append("metadata")
        
        return ", ".join(select_fields)
    
    def _build_where_clause(
        self,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        id_list: Optional[List[str]] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build WHERE clause from filters
        
        Args:
            where: Metadata filter
            where_document: Document filter
            id_list: List of IDs to filter
            
        Returns:
            Tuple of (where_clause, params)
        """
        where_clauses = []
        params = []
        
        # Add ids filter if provided
        if id_list:
            # Process IDs for varbinary(512) _id field - support any string format
            processed_ids = []
            for id_val in id_list:
                if not isinstance(id_val, str):
                    id_val = str(id_val)
                processed_ids.append(self._convert_id_to_sql(id_val))
            
            where_clauses.append(f"_id IN ({','.join(processed_ids)})")
        
        # Add metadata filter
        if where:
            meta_clause, meta_params = FilterBuilder.build_metadata_filter(where, "metadata")
            if meta_clause:
                where_clauses.append(meta_clause)
                params.extend(meta_params)
        
        # Add document filter
        if where_document:
            doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
            if doc_clause:
                where_clauses.append(doc_clause)
                params.extend(doc_params)
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return where_clause, params
    
    def _parse_row_value(self, value: Any) -> Any:
        """
        Parse row value (handle JSON strings)
        
        Args:
            value: Raw value from database
            
        Returns:
            Parsed value
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        
        return value
    
    def _convert_id_to_sql(self, id_val: str) -> str:
        """
        Convert string ID to SQL format for varbinary(512) _id field
        
        Args:
            id_val: String ID (can be any string like "id1", "item-123", etc.)
            
        Returns:
            SQL expression to convert string to binary (e.g., "CAST('id1' AS BINARY)")
        """
        if not isinstance(id_val, str):
            id_val = str(id_val)
        
        # Escape single quotes in the ID
        id_val_escaped = id_val.replace("'", "''")
        # Use CAST to convert string to binary for varbinary(512) field
        return f"CAST('{id_val_escaped}' AS BINARY)"
    
    def _convert_id_from_bytes(self, record_id: Any) -> str:
        """
        Convert _id from bytes to string format
        
        Args:
            record_id: Record ID from database (can be bytes, str, or other format)
            
        Returns:
            String ID
        """
        # If it's already a string, return as is
        if isinstance(record_id, str):
            return record_id
        
        # Convert bytes to string (UTF-8 decode)
        if isinstance(record_id, bytes):
            try:
                return record_id.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decode fails, return hex representation as fallback
                return record_id.hex()
        
        # For other formats, convert to string
        return str(record_id)
    
    def _process_query_row(
        self,
        row: Dict[str, Any],
        include_fields: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Process a row from query results
        
        Args:
            row: Normalized row dictionary
            include_fields: Fields to include
            
        Returns:
            Result item dictionary
        """
        # Convert _id from bytes to string format
        record_id = self._convert_id_from_bytes(row["_id"])
        result_item = {"_id": record_id}
        
        if "document" in row and row["document"] is not None:
            result_item["document"] = row["document"]
        
        if "embedding" in row and row["embedding"] is not None:
            result_item["embedding"] = self._parse_row_value(row["embedding"])
        
        if "metadata" in row and row["metadata"] is not None:
            result_item["metadata"] = self._parse_row_value(row["metadata"])
        
        if "distance" in row:
            result_item["distance"] = float(row["distance"])
        
        return result_item
    
    def _process_get_row(
        self,
        row: Dict[str, Any],
        include_fields: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Process a row from get results
        
        Args:
            row: Normalized row dictionary
            include_fields: Fields to include
            
        Returns:
            Result item dictionary with id, document, embedding, metadata
        """
        # Convert _id from bytes to string format
        record_id = self._convert_id_from_bytes(row["_id"])
        
        document = None
        embedding = None
        metadata = None
        
        # Include document if requested
        if include_fields.get("documents") or include_fields.get("document"):
            if "document" in row:
                document = row["document"]
        
        # Include metadata if requested
        if include_fields.get("metadatas") or include_fields.get("metadata"):
            if "metadata" in row and row["metadata"] is not None:
                metadata = self._parse_row_value(row["metadata"])
        
        # Include embedding if requested
        if include_fields.get("embeddings") or include_fields.get("embedding"):
            if "embedding" in row and row["embedding"] is not None:
                embedding = self._parse_row_value(row["embedding"])
        
        return {
            "id": record_id,
            "document": document,
            "embedding": embedding,
            "metadata": metadata
        }
    
    def _use_context_manager_for_cursor(self) -> bool:
        """
        Whether to use context manager for cursor
        
        Returns:
            True if context manager should be used, False otherwise
        """
        # Default implementation: use context manager
        # Subclasses can override this if they need different behavior
        return True
    
    # -------------------- DQL Operations (Common Implementation) --------------------
    
    def _collection_query(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        query_texts: Optional[Union[str, List[str]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Union[QueryResult, List[QueryResult]]:
        """
        [Internal] Query collection by vector similarity - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query_embeddings: Query vector(s) (preferred)
            query_texts: Query text(s) - will be embedded if provided (preferred)
            n_results: Number of results (default: 10)
            where: Metadata filter
            where_document: Document filter
            include: Fields to include
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts to vectors.
                                   Required if query_texts is provided and collection doesn't have
                                   an embedding_function set. Must implement __call__ method that
                                   accepts Documents and returns Embeddings (List[List[float]]).
            
        Returns:
            - If single vector/text provided: QueryResult object containing query results
            - If multiple vectors/texts provided: List of QueryResult objects, one for each query vector
        """
        logger.info(f"Querying collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()
        
        # Convert collection name to table name
        table_name = f"c$v1${collection_name}"
        
        # Handle vector generation logic:
        # 1. If query_embeddings are provided, use them directly without embedding
        # 2. If query_embeddings are not provided but query_texts are provided:
        #    - If embedding_function is provided, use it to generate vectors from query_texts
        #    - If embedding_function is not provided, raise an error
        # 3. If neither query_embeddings nor query_texts are provided, raise an error
        
        embedding_function = kwargs.get('embedding_function')
        
        if query_embeddings is not None:
            # Query embeddings provided, use them directly without embedding
            pass
        elif query_texts is not None:
            # Query embeddings not provided but query_texts are provided, check for embedding_function
            if embedding_function is not None:
                logger.info("Embedding query texts...")
                query_embeddings = self._embed_texts(query_texts, embedding_function=embedding_function)
            else:
                raise ValueError(
                    "query_texts provided but no query_embeddings and no embedding_function. "
                    "Either:\n"
                    "  1. Provide query_embeddings directly, or\n"
                    "  2. Provide embedding_function to auto-generate vectors from query_texts."
                )
        else:
            # Neither query_embeddings nor query_texts provided, raise an error
            raise ValueError(
                "Neither query_embeddings nor query_texts provided. "
                "Please provide either:\n"
                "  1. query_embeddings directly, or\n"
                "  2. query_texts with embedding_function to generate vectors."
            )
        
        # Normalize query vectors to list of lists
        query_vectors = self._normalize_query_vectors(query_embeddings)
        
        # Check if multiple vectors provided
        is_multiple_vectors = len(query_vectors) > 1
        
        # Normalize include fields
        include_fields = self._normalize_include_fields(include)
        
        # Build SELECT clause
        select_clause = self._build_select_clause(include_fields)
        
        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document)
        
        use_context_manager = self._use_context_manager_for_cursor()
        
        # Collect results for each query vector separately
        query_results = []
        
        for query_vector in query_vectors:
            # Convert vector to string format for SQL
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
            # Build SQL query with vector distance calculation
            # Reference: SELECT id, vec FROM t2 ORDER BY l2_distance(vec, '[0.1, 0.2, 0.3]') APPROXIMATE LIMIT 5;
            # Need to include distance in SELECT for result processing
            sql = f"""
                SELECT {select_clause}, 
                       l2_distance(embedding, '{vector_str}') AS distance
                FROM `{table_name}`
                {where_clause}
                ORDER BY l2_distance(embedding, '{vector_str}')
                APPROXIMATE
                LIMIT %s
            """
            
            # Execute query
            query_params = params + [n_results]
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")
            
            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)
            
            # Create QueryResult for this vector
            query_result = QueryResult()
            for row in rows:
                result_item = self._process_query_row(row, include_fields)
                query_result.add_item(
                    id=result_item.get("_id"),
                    document=result_item.get("document"),
                    embedding=result_item.get("embedding"),
                    metadata=result_item.get("metadata"),
                    distance=result_item.get("distance")
                )
            
            query_results.append(query_result)
        
        # Return single QueryResult if only one vector, otherwise return list
        if is_multiple_vectors:
            logger.info(f"✅ Query completed for '{collection_name}' with {len(query_vectors)} vectors, returning {len(query_results)} QueryResult objects")
            return query_results
        else:
            logger.info(f"✅ Query completed for '{collection_name}', found {len(query_results[0])} results")
            return query_results[0]
    
    def _collection_get(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Union[QueryResult, List[QueryResult]]:
        """
        [Internal] Get data from collection by IDs or filters - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            limit: Maximum number of results (optional)
            offset: Number of results to skip (optional)
            include: Fields to include in results (optional)
            **kwargs: Additional parameters
            
        Returns:
            - If single ID provided: QueryResult object containing get results for that ID
            - If multiple IDs provided (and no filters): List of QueryResult objects, one for each ID
            - If filters provided (no IDs or multiple IDs with filters): QueryResult object containing all matching results
        """
        logger.info(f"Getting data from collection '{collection_name}'")
        conn = self._ensure_connection()
        
        # Convert collection name to table name
        table_name = f"c$v1${collection_name}"
        
        # Set defaults
        if limit is None:
            limit = 100
        if offset is None:
            offset = 0
        
        # Normalize ids to list
        id_list = None
        is_single_id = False
        if ids is not None:
            if isinstance(ids, str):
                id_list = [ids]
                is_single_id = True
            else:
                id_list = ids
                is_single_id = len(id_list) == 1
        
        # Check if we should return multiple QueryResults (multiple IDs and no filters)
        has_filters = where is not None or where_document is not None
        is_multiple_ids = id_list is not None and len(id_list) > 1
        should_return_multiple = is_multiple_ids and not has_filters
        
        # Normalize include fields (default includes documents and metadatas)
        include_fields = self._normalize_include_fields(include)
        
        # Build SELECT clause - always include _id
        select_clause = self._build_select_clause(include_fields)
        
        use_context_manager = self._use_context_manager_for_cursor()
        
        # If multiple IDs and no filters, get each ID separately
        if should_return_multiple:
            query_results = []
            for single_id in id_list:
                # Build WHERE clause for this single ID
                where_clause, params = self._build_where_clause(where, where_document, [single_id])
                
                # Build SQL query
                sql = f"""
                    SELECT {select_clause}
                    FROM `{table_name}`
                    {where_clause}
                    LIMIT %s OFFSET %s
                """
                
                # Execute query
                query_params = params + [limit, offset]
                logger.debug(f"Executing SQL: {sql}")
                logger.debug(f"Parameters: {query_params}")
                
                rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)
                
                # Build QueryResult for this ID
                query_result = QueryResult()
                for row in rows:
                    processed_row = self._process_get_row(row, include_fields)
                    query_result.add_item(
                        id=processed_row["id"],
                        document=processed_row["document"],
                        embedding=processed_row["embedding"],
                        metadata=processed_row["metadata"]
                    )
                
                query_results.append(query_result)
            
            logger.info(f"✅ Get completed for '{collection_name}' with {len(id_list)} IDs, returning {len(query_results)} QueryResult objects")
            return query_results
        else:
            # Single ID or filters: return single QueryResult
            # Build WHERE clause from filters
            where_clause, params = self._build_where_clause(where, where_document, id_list)
            
            # Build SQL query
            sql = f"""
                SELECT {select_clause}
                FROM `{table_name}`
                {where_clause}
                LIMIT %s OFFSET %s
            """
            
            # Execute query
            query_params = params + [limit, offset]
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")
            
            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)
            
            # Build QueryResult
            query_result = QueryResult()
            
            for row in rows:
                # Process row
                processed_row = self._process_get_row(row, include_fields)
                
                query_result.add_item(
                    id=processed_row["id"],
                    document=processed_row["document"],
                    embedding=processed_row["embedding"],
                    metadata=processed_row["metadata"]
                )
            
            logger.info(f"✅ Get completed for '{collection_name}', found {len(query_result)} results")
            return query_result
    
    def _collection_hybrid_search(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
        knn: Optional[Dict[str, Any]] = None,
        rank: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        [Internal] Hybrid search combining full-text search and vector similarity search - Common SQL-based implementation
        
        Supports:
        1. Scalar query (metadata filtering only)
        2. Full-text search (with optional metadata filtering)
        3. Vector search (with optional metadata filtering)
        4. Scalar + vector search (with optional metadata filtering)
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query: Full-text search configuration dict with:
                - where_document: Document filter conditions (e.g., {"$contains": "text"})
                - where: Metadata filter conditions (e.g., {"page": {"$gte": 5}})
            knn: Vector search configuration dict with:
                - query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
                - query_embeddings: Query vector(s) (optional if query_texts provided)
                - where: Metadata filter conditions (optional)
                - n_results: Number of results for vector search (optional)
            rank: Ranking configuration dict (e.g., {"rrf": {"rank_window_size": 60, "rank_constant": 60}})
            n_results: Final number of results to return after ranking (default: 10)
            include: Fields to include in results (optional)
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts in knn to vectors.
                                   Required if knn.query_texts is provided and collection doesn't have
                                   an embedding_function set. Must implement __call__ method that
                                   accepts Documents and returns Embeddings (List[List[float]]).
            
        Returns:
            Search results dictionary containing ids, distances, metadatas, documents, embeddings, etc.
        """
        logger.info(f"Hybrid search in collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()
        
        # Build table name
        table_name = f"c$v1${collection_name}"
        
        # Build search_parm JSON
        search_parm = self._build_search_parm(query, knn, rank, n_results, **kwargs)
        
        # Convert search_parm to JSON string
        search_parm_json = json.dumps(search_parm, ensure_ascii=False)
        
        # Use variable binding to avoid datatype issues
        use_context_manager = self._use_context_manager_for_cursor()
        
        # Set the search_parm variable first
        escaped_params = search_parm_json.replace("'", "''")
        set_sql = f"SET @search_parm = '{escaped_params}'"
        logger.debug(f"Setting search_parm: {set_sql}")
        logger.debug(f"Search parm JSON: {search_parm_json}")
        
        # Execute SET statement
        self._execute_query_with_cursor(conn, set_sql, [], use_context_manager)
        
        # Get SQL query from DBMS_HYBRID_SEARCH.GET_SQL
        get_sql_query = f"SELECT DBMS_HYBRID_SEARCH.GET_SQL('{table_name}', @search_parm) as query_sql FROM dual"
        logger.debug(f"Getting SQL query: {get_sql_query}")
        
        rows = self._execute_query_with_cursor(conn, get_sql_query, [], use_context_manager)
        
        if not rows or not rows[0].get("query_sql"):
            logger.warning(f"No SQL query returned from GET_SQL")
            return {
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": [],
                "embeddings": []
            }
        
        # Get the SQL query string
        query_sql = rows[0]["query_sql"]
        if isinstance(query_sql, str):
            # Remove any surrounding quotes if present
            query_sql = query_sql.strip().strip("'\"")
        
        logger.debug(f"Executing query SQL: {query_sql}")
        
        # Execute the returned SQL query
        result_rows = self._execute_query_with_cursor(conn, query_sql, [], use_context_manager)
        
        # Transform SQL query results to standard format
        return self._transform_sql_result(result_rows, include)
    
    def _build_search_parm(
        self,
        query: Optional[Dict[str, Any]],
        knn: Optional[Dict[str, Any]],
        rank: Optional[Dict[str, Any]],
        n_results: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build search_parm JSON from query, knn, and rank parameters
        
        Args:
            query: Full-text search configuration dict
            knn: Vector search configuration dict
            rank: Ranking configuration dict
            n_results: Final number of results to return
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts in knn to vectors.
                                   Required if knn.query_texts is provided. Must implement __call__
                                   method that accepts Documents and returns Embeddings (List[List[float]]).
            
        Returns:
            search_parm dictionary
        """
        search_parm = {}
        
        # Build query part (full-text search or scalar query)
        if query:
            query_expr = self._build_query_expression(query)
            if query_expr:
                search_parm["query"] = query_expr
        
        # Build knn part (vector search)
        if knn:
            knn_expr = self._build_knn_expression(knn, **kwargs)
            if knn_expr:
                search_parm["knn"] = knn_expr
        
        # Build rank part
        if rank:
            search_parm["rank"] = rank
        
        return search_parm
    
    def _build_query_expression(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build query expression from query dict
        
        Supports:
        - Scalar query (metadata filtering only): query.range or query.term
        - Full-text search: query.query_string
        - Full-text search with metadata filtering: query.bool with must and filter
        """
        where_document = query.get("where_document")
        where = query.get("where")
        
        # Case 1: Scalar query (metadata filtering only, no full-text search)
        if not where_document and where:
            filter_conditions = self._build_metadata_filter_for_search_parm(where)
            if filter_conditions:
                # If only one filter condition, check its type
                if len(filter_conditions) == 1:
                    filter_cond = filter_conditions[0]
                    # Check if it's a range query
                    if "range" in filter_cond:
                        return {"range": filter_cond["range"]}
                    # Check if it's a term query
                    elif "term" in filter_cond:
                        return {"term": filter_cond["term"]}
                    # Otherwise, it's a bool query, wrap in filter
                    else:
                        return {"bool": {"filter": filter_conditions}}
                # Multiple filter conditions, wrap in bool
                return {"bool": {"filter": filter_conditions}}
        
        # Case 2: Full-text search (with or without metadata filtering)
        if where_document:
            # Build document query using query_string
            doc_query = self._build_document_query(where_document)
            if doc_query:
                # Build filter from where condition
                filter_conditions = self._build_metadata_filter_for_search_parm(where)
                
                if filter_conditions:
                    # Full-text search with metadata filtering
                    return {
                        "bool": {
                            "must": [doc_query],
                            "filter": filter_conditions
                        }
                    }
                else:
                    # Full-text search only
                    return doc_query
        
        return None
    
    def _build_document_query(self, where_document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build document query from where_document condition using query_string
        
        Args:
            where_document: Document filter conditions
            
        Returns:
            query_string query dict
        """
        if not where_document:
            return None
        
        # Handle $contains - use query_string
        if "$contains" in where_document:
            return {
                "query_string": {
                    "fields": ["document"],
                    "query": where_document["$contains"]
                }
            }
        
        # Handle $and with $contains
        if "$and" in where_document:
            and_conditions = where_document["$and"]
            contains_queries = []
            for condition in and_conditions:
                if isinstance(condition, dict) and "$contains" in condition:
                    contains_queries.append(condition["$contains"])
            
            if contains_queries:
                # Combine multiple $contains with AND
                return {
                    "query_string": {
                        "fields": ["document"],
                        "query": " ".join(contains_queries)
                    }
                }
        
        # Handle $or with $contains
        if "$or" in where_document:
            or_conditions = where_document["$or"]
            contains_queries = []
            for condition in or_conditions:
                if isinstance(condition, dict) and "$contains" in condition:
                    contains_queries.append(condition["$contains"])
            
            if contains_queries:
                # Combine multiple $contains with OR
                return {
                    "query_string": {
                        "fields": ["document"],
                        "query": " OR ".join(contains_queries)
                    }
                }
        
        # Default: if it's a string, treat as $contains
        if isinstance(where_document, str):
            return {
                "query_string": {
                    "fields": ["document"],
                    "query": where_document
                }
            }
        
        return None
    
    def _build_metadata_filter_for_search_parm(self, where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build metadata filter conditions for search_parm using JSON_EXTRACT format
        
        Args:
            where: Metadata filter conditions
            
        Returns:
            List of filter conditions in search_parm format
            Format: {"term": {"(JSON_EXTRACT(metadata, '$.field_name'))": "value"}}
            or {"range": {"(JSON_EXTRACT(metadata, '$.field_name'))": {"gte": 30, "lte": 90}}}
        """
        if not where:
            return []
        
        return self._build_metadata_filter_conditions(where)
    
    def _build_metadata_filter_conditions(self, condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recursively build metadata filter conditions from nested dictionary
        
        Args:
            condition: Filter condition dictionary
            
        Returns:
            List of filter conditions
        """
        if not condition:
            return []
        
        result = []
        
        # Handle logical operators
        if "$and" in condition:
            must_conditions = []
            for sub_condition in condition["$and"]:
                sub_filters = self._build_metadata_filter_conditions(sub_condition)
                must_conditions.extend(sub_filters)
            if must_conditions:
                result.append({"bool": {"must": must_conditions}})
            return result
        
        if "$or" in condition:
            should_conditions = []
            for sub_condition in condition["$or"]:
                sub_filters = self._build_metadata_filter_conditions(sub_condition)
                should_conditions.extend(sub_filters)
            if should_conditions:
                result.append({"bool": {"should": should_conditions}})
            return result
        
        if "$not" in condition:
            not_filters = self._build_metadata_filter_conditions(condition["$not"])
            if not_filters:
                result.append({"bool": {"must_not": not_filters}})
            return result
        
        # Handle field conditions
        for key, value in condition.items():
            if key in ["$and", "$or", "$not"]:
                continue
            
            # Build field name with JSON_EXTRACT format
            field_name = f"(JSON_EXTRACT(metadata, '$.{key}'))"
            
            if isinstance(value, dict):
                # Handle comparison operators
                range_conditions = {}
                term_value = None
                
                for op, op_value in value.items():
                    if op == "$eq":
                        term_value = op_value
                    elif op == "$ne":
                        # $ne should be in must_not
                        result.append({"bool": {"must_not": [{"term": {field_name: op_value}}]}})
                    elif op == "$lt":
                        range_conditions["lt"] = op_value
                    elif op == "$lte":
                        range_conditions["lte"] = op_value
                    elif op == "$gt":
                        range_conditions["gt"] = op_value
                    elif op == "$gte":
                        range_conditions["gte"] = op_value
                    elif op == "$in":
                        # For $in, create multiple term queries wrapped in should
                        in_conditions = [{"term": {field_name: val}} for val in op_value]
                        if in_conditions:
                            result.append({"bool": {"should": in_conditions}})
                    elif op == "$nin":
                        # For $nin, create multiple term queries wrapped in must_not
                        nin_conditions = [{"term": {field_name: val}} for val in op_value]
                        if nin_conditions:
                            result.append({"bool": {"must_not": nin_conditions}})
                
                if range_conditions:
                    result.append({"range": {field_name: range_conditions}})
                elif term_value is not None:
                    result.append({"term": {field_name: term_value}})
            else:
                # Direct equality
                result.append({"term": {field_name: value}})
        
        return result
    
    def _build_knn_expression(self, knn: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Build knn expression from knn dict
        
        Args:
            knn: Vector search configuration dict with:
                - query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
                - query_embeddings: Query vector(s) (optional if query_texts provided)
                - where: Metadata filter conditions (optional)
                - n_results: Number of results for vector search (optional)
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts to vectors.
                                   Required if query_texts is provided. Must implement __call__
                                   method that accepts Documents and returns Embeddings (List[List[float]]).
            
        Returns:
            knn expression dict with optional filter
        """
        query_texts = knn.get("query_texts")
        query_embeddings = knn.get("query_embeddings")
        where = knn.get("where")
        n_results = knn.get("n_results", 10)
        
        # Handle vector generation logic:
        # 1. If query_embeddings are provided, use them directly without embedding
        # 2. If query_embeddings are not provided but query_texts are provided:
        #    - If embedding_function is provided, use it to generate vectors from query_texts
        #    - If embedding_function is not provided, raise an error
        # 3. If neither query_embeddings nor query_texts are provided, raise an error
        
        embedding_function = kwargs.get('embedding_function')
        
        # Get query vector
        query_vector = None
        if query_embeddings:
            # Query embeddings provided, use them directly without embedding
            if isinstance(query_embeddings, list) and len(query_embeddings) > 0:
                if isinstance(query_embeddings[0], list):
                    query_vector = query_embeddings[0]  # Use first vector
                else:
                    query_vector = query_embeddings
        elif query_texts:
            # Query embeddings not provided but query_texts are provided, check for embedding_function
            if embedding_function is not None:
                try:
                    texts = query_texts if isinstance(query_texts, list) else [query_texts]
                    embeddings = self._embed_texts(texts[0] if len(texts) > 0 else texts, embedding_function=embedding_function)
                    if embeddings and len(embeddings) > 0:
                        query_vector = embeddings[0]
                except Exception as e:
                    logger.error(f"Failed to generate embeddings from query_texts: {e}")
                    raise ValueError(f"Failed to generate embeddings from query_texts: {e}")
            else:
                raise ValueError(
                    "knn.query_texts provided but no knn.query_embeddings and no embedding_function. "
                    "Either:\n"
                    "  1. Provide knn.query_embeddings directly, or\n"
                    "  2. Provide embedding_function to auto-generate vectors from knn.query_texts."
                )
        else:
            # Neither query_embeddings nor query_texts provided, raise an error
            raise ValueError(
                "knn requires either query_embeddings or query_texts. "
                "Please provide either:\n"
                "  1. knn.query_embeddings directly, or\n"
                "  2. knn.query_texts with embedding_function to generate vectors."
            )
        
        if not query_vector:
            return None
        
        # Build knn expression
        knn_expr = {
            "field": "embedding",
            "k": n_results,
            "query_vector": query_vector
        }
        
        # Add filter using JSON_EXTRACT format
        filter_conditions = self._build_metadata_filter_for_search_parm(where)
        if filter_conditions:
            knn_expr["filter"] = filter_conditions
        
        return knn_expr
    
    def _build_source_fields(self, include: Optional[List[str]]) -> List[str]:
        """Build _source fields list from include parameter"""
        if not include:
            return ["document", "metadata", "embedding"]
        
        source_fields = []
        field_mapping = {
            "documents": "document",
            "metadatas": "metadata",
            "embeddings": "embedding"
        }
        
        for field in include:
            mapped = field_mapping.get(field.lower(), field)
            if mapped not in source_fields:
                source_fields.append(mapped)
        
        return source_fields if source_fields else ["document", "metadata", "embedding"]
    
    def _transform_sql_result(self, result_rows: List[Dict[str, Any]], include: Optional[List[str]]) -> Dict[str, Any]:
        """
        Transform SQL query results to standard format
        
        Args:
            result_rows: List of row dictionaries from SQL query
            include: Fields to include in results (optional)
            
        Returns:
            Standard format dictionary with ids, distances, metadatas, documents, embeddings
        """
        if not result_rows:
            return {
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": [],
                "embeddings": []
            }
        
        ids = []
        distances = []
        metadatas = []
        documents = []
        embeddings = []
        
        for row in result_rows:
            # Extract id (may be in different column names)
            row_id = row.get("id") or row.get("_id") or row.get("ID")
            # Convert bytes _id to string format
            row_id = self._convert_id_from_bytes(row_id)
            ids.append(row_id)
            
            # Extract distance/score (may be in different column names)
            distance = row.get("_distance") or row.get("distance") or row.get("_score") or row.get("score") or row.get("DISTANCE") or row.get("_DISTANCE") or row.get("SCORE") or 0.0
            distances.append(distance)
            
            # Extract metadata
            if include is None or "metadatas" in include or "metadata" in include:
                metadata = row.get("metadata") or row.get("METADATA")
                # Parse JSON string if needed
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        pass
                metadatas.append(metadata)
            else:
                metadatas.append(None)
            
            # Extract document
            if include is None or "documents" in include or "document" in include:
                document = row.get("document") or row.get("DOCUMENT")
                documents.append(document)
            else:
                documents.append(None)
            
            # Extract embedding
            if include and ("embeddings" in include or "embedding" in include):
                embedding = row.get("embedding") or row.get("EMBEDDING")
                # Parse JSON string or list if needed
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except (json.JSONDecodeError, TypeError):
                        pass
                embeddings.append(embedding)
            else:
                embeddings.append(None)
        
        return {
            "ids": ids,
            "distances": distances,
            "metadatas": metadatas,
            "documents": documents,
            "embeddings": embeddings
        }
    
    def _transform_search_result(self, search_result: Dict[str, Any], include: Optional[List[str]]) -> Dict[str, Any]:
        """Transform OceanBase search result to standard format"""
        # OceanBase SEARCH function returns results in a specific format
        # This needs to be adapted based on actual return format
        # For now, assuming it returns hits array
        
        hits = search_result.get("hits", {}).get("hits", [])
        
        ids = []
        distances = []
        metadatas = []
        documents = []
        embeddings = []
        
        for hit in hits:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            
            ids.append(hit.get("_id"))
            distances.append(score)
            
            if include is None or "metadatas" in include or "metadata" in include:
                metadatas.append(source.get("metadata"))
            else:
                metadatas.append(None)
            
            if include is None or "documents" in include or "document" in include:
                documents.append(source.get("document"))
            else:
                documents.append(None)
            
            if include and ("embeddings" in include or "embedding" in include):
                embeddings.append(source.get("embedding"))
            else:
                embeddings.append(None)
        
        return {
            "ids": ids,
            "distances": distances,
            "metadatas": metadatas,
            "documents": documents,
            "embeddings": embeddings
        }
    
    # -------------------- Collection Info --------------------
    
    def _collection_count(
        self,
        collection_id: Optional[str],
        collection_name: str
    ) -> int:
        """
        [Internal] Get the number of items in collection - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            
        Returns:
            Item count
        """
        logger.info(f"Counting items in collection '{collection_name}'")
        conn = self._ensure_connection()
        
        # Convert collection name to table name
        table_name = CollectionNames.table_name(collection_name)
        
        # Execute COUNT query
        sql = f"SELECT COUNT(*) as cnt FROM `{table_name}`"
        logger.debug(f"Executing SQL: {sql}")
        
        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, [], use_context_manager)
        
        if not rows:
            count = 0
        else:
            # Extract count from result
            row = rows[0]
            if isinstance(row, dict):
                count = row.get('cnt', 0)
            elif isinstance(row, (tuple, list)):
                count = row[0] if len(row) > 0 else 0
            else:
                count = int(row) if row else 0
        
        logger.info(f"✅ Collection '{collection_name}' has {count} items")
        return count
