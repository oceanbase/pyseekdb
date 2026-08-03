"""
Base client interface definition
"""

import contextlib
import json
import logging
import os
import re
import struct
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pymysql.converters import escape_string

from .admin_client import DEFAULT_TENANT, AdminAPI
from .base_connection import BaseConnection
from .collection import Collection
from .configuration import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_VECTOR_DIMENSION,
    LOGIC_DATA_TABLE_LOB_INROW_THRESHOLD,
    MAX_HNSW_VECTOR_DIMENSION,
    MAX_IVF_VECTOR_DIMENSION,
    Configuration,
    ConfigurationParam,
    FulltextIndexConfig,
    HNSWConfiguration,
    IVFConfiguration,
    IVFIndexType,
    VectorIndexConfig,
)
from .database import Database
from .document_query_builder import (
    build_document_hybrid_expression,
    doc_matches_where_document,
    document_expr_as_knn_filter,
    where_document_knn_prefilterable,
)
from .embedding_function import (
    Documents as EmbeddingDocuments,
)
from .embedding_function import (
    EmbeddingFunction,
    EmbeddingFunctionRegistry,
    get_default_embedding_function,
)
from .filters import FilterBuilder
from .kernel_errors import maybe_reraise_friendly_kernel_error, namespace_kernel_error_guard
from .meta_info import CollectionFieldNames, CollectionNames, NamespaceCollectionNames, NamespaceFieldNames
from .query_types import QueryHint
from .schema import Schema, SparseVectorIndexConfig
from .sparse_embedding_function import (
    SparseEmbeddingFunction,
    SparseEmbeddingFunctionRegistry,
    SparseVector,
    _sparse_vector_to_sql,
)
from .sql_utils import _query_hint_to_sql, is_query_sql
from .types import K as FieldKey
from .validators import (
    _MAX_N_RESULTS,
    _MAX_NAMESPACE_BATCH_SIZE,
    _quote_sql_identifier,
    _validate_database_name,
    _validate_namespace_explicit_embedding_dimensions,
    _validate_record_ids,
)
from .version import Version

# Type alias for embedding_function parameter that can be EmbeddingFunction, None, or sentinel
EmbeddingFunctionParam = EmbeddingFunction[EmbeddingDocuments] | None | Any

_COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

# DBMS_HYBRID_SEARCH.GET_SQL may quote a JSON_EXTRACT expression as though it
# were a column identifier.  Require JSON_EXTRACT to start the quoted content
# so the match cannot span from one ordinary quoted identifier to another.
_QUOTED_JSON_EXTRACT_EXPRESSION_PATTERN = re.compile(
    r"`(?P<expression>\s*\(*\s*JSON_EXTRACT\s*\([^`]*\)\s*\)*\s*)`",
    re.IGNORECASE,
)

# Maximum allowed length for user-facing collection names.
_MAX_COLLECTION_NAME_LENGTH = 512

# Minimum LakeBase (OceanBase Database AI) version for namespace-enabled collections.
NAMESPACE_MIN_LAKEBASE_VERSION = Version("4.6.1.0")
# Backward-compatible alias used by existing tests and skip helpers.
NAMESPACE_MIN_OB_VERSION = NAMESPACE_MIN_LAKEBASE_VERSION

_LAKEBASE_VERSION_MARKER = "database ai"

logger = logging.getLogger(__name__)

from .types import _NOT_PROVIDED, _NotProvided  # noqa: E402, F401


def _unquote_json_extract_expressions(query_sql: str) -> str:
    """Unquote JSON_EXTRACT expressions without touching adjacent SQL identifiers."""
    return _QUOTED_JSON_EXTRACT_EXPRESSION_PATTERN.sub(r"\g<expression>", query_sql)


def is_lakebase_version_string(version_str: str) -> bool:
    """Return whether a ``SELECT version()`` string identifies a LakeBase cluster."""
    return _LAKEBASE_VERSION_MARKER in version_str.lower()


def _extract_collection_id_from_sdk_row(row: Any) -> str:
    """Extract the collection_id from an sdk_collections row (dict/tuple/scalar)."""
    if isinstance(row, dict):
        collection_id = row.get("COLLECTION_ID") or row.get("collection_id") or ""
    elif isinstance(row, (tuple, list)):
        collection_id = row[0] if len(row) > 0 else ""
    else:
        collection_id = str(row)
    return str(collection_id or "")


def _is_collection_conflict_error(exc: BaseException) -> bool:
    """Whether the exception (or its cause chain) indicates a collection/table already exists."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "already exists" in message and ("collection" in message or "table" in message or "code=1050" in message):
            return True
        if type(current).__name__ == "SeekdbError" and "already exists" in message:
            return True
        current = current.__cause__
    return False


def _is_namespace_catalog_conflict_error(exc: BaseException) -> bool:
    """Whether the exception indicates a duplicate sdk_namespaces/sdk_ltables unique-key conflict (1062)."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "duplicate entry" in message and (
            "uk_sdk_ns_coll_name" in message or "uk_sdk_lt_coll_ns_name" in message or "code=1062" in message
        ):
            return True
        if type(current).__name__ == "IntegrityError" and "1062" in message:
            return True
        current = current.__cause__
    return False


def _is_sdk_collection_catalog_conflict_error(exc: BaseException) -> bool:
    """Whether the exception indicates a duplicate sdk_collections collection_name conflict (1062)."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "duplicate entry" in message and (
            "uk_sdk_coll_name" in message
            or "idx_name" in message
            or "collection_name" in message
            or "code=1062" in message
        ):
            return True
        if type(current).__name__ == "IntegrityError" and "1062" in message:
            return True
        current = current.__cause__
    return False


def _reraise_unless_unique_index_exists(exc: BaseException) -> None:
    """Re-raise unless the exception indicates the unique index is already present."""
    message = str(exc).lower()
    if (
        "already exists" in message
        or "duplicate key name" in message
        or "code=1061" in message
        or ("1061" in message and "duplicate" in message)
    ):
        return
    raise exc


def _extract_hnsw_config(config: ConfigurationParam) -> HNSWConfiguration | None:
    """Return the HNSW config from a Configuration/HNSWConfiguration, or None."""
    if config is None:
        return None
    elif isinstance(config, HNSWConfiguration):
        return config
    elif isinstance(config, Configuration):
        return config.hnsw
    else:
        raise TypeError(f"configuration must be Configuration, HNSWConfiguration, or None, got {type(config)}")


def _extract_fulltext_config(
    config: ConfigurationParam,
) -> FulltextIndexConfig | None:
    """Return the fulltext index config from a Configuration, or None."""
    if config is None:
        return None
    elif isinstance(config, HNSWConfiguration):
        # HNSWConfiguration doesn't have fulltext config, return None (will use default)
        return None
    elif isinstance(config, Configuration):
        # If Configuration has fulltext_config, return it; otherwise return None (will use default)
        return config.fulltext_config
    else:
        # Should not reach here due to type checking, but handle gracefully
        return None


def _validate_collection_name(name: str) -> None:
    """
    Validate collection name against allowed charset and length constraints.

    Rules:
    - Type must be str
    - Length between 1 and _MAX_COLLECTION_NAME_LENGTH
    - Only [a-zA-Z0-9_]

    Raises:
        TypeError: If name is not a string.
        ValueError: If name is empty, too long, or contains invalid characters.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"Invalid collection name: '{name}'. Collection name must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError(f"Invalid collection name: '{name}'. Collection name must not be empty")
    if len(name) > _MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Invalid collection name: '{name}'. Collection name too long: {len(name)} characters; maximum allowed is {_MAX_COLLECTION_NAME_LENGTH}."
        )
    if _COLLECTION_NAME_PATTERN.match(name) is None:
        raise ValueError(
            f"Invalid collection name: '{name}'. Collection name contains invalid characters. "
            "Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]"
        )


_DEFAULT_PARTITION_COUNT = 1000
# Unquoted id for WHERE/CASE; plain JSON_EXTRACT returns a quoted JSON string and
# can route through SEARCH INDEX on SS logic tables, breaking cross-namespace id lookups.
_NS_DATA_CONTENT_ID_EXPR = "JSON_UNQUOTE(JSON_EXTRACT(data_content, '$.id'))"


def _build_default_ltable_schema(
    *,
    has_fulltext: bool = False,
    has_ivf: bool = False,
) -> dict:
    """Return the logical-table schema matching the indexes actually provisioned."""
    index_info: list[dict[str, Any]] = [
        {"index_seq": 0, "index_type": "PRIMARY", "indexed_columns": []},
        {"index_seq": 1, "index_type": "SEARCH_INDEX", "indexed_columns": [1]},
    ]
    next_seq = 2
    if has_fulltext:
        index_info.append({"index_seq": next_seq, "index_type": "FULLTEXT", "indexed_columns": [2]})
        next_seq += 1
    if has_ivf:
        index_info.append({"index_seq": next_seq, "index_type": "IVF", "indexed_columns": [3]})
    return {
        "col_info": [
            {"col_idx": 1, "col_name": "metadata", "col_type": "JSON"},
            {"col_idx": 2, "col_name": "content", "col_type": "TEXT"},
            {"col_idx": 3, "col_name": "embedding", "col_type": "VECTOR"},
        ],
        "index_info": index_info,
    }


def _get_fulltext_index_sql(
    fulltext_config: FulltextIndexConfig | None = None,
) -> str:
    """
    Generate FULLTEXT INDEX SQL clause from fulltext configuration.

    Args:
        fulltext_config: FulltextIndexConfig or None. If None, defaults to IK parser.

    Returns:
        SQL clause string for FULLTEXT INDEX (e.g., "WITH PARSER ik" or "WITH PARSER ngram PARSER_PROPERTIES=(size=2)")
    """
    if fulltext_config is None:
        # Default to IK parser for backward compatibility
        return "WITH PARSER ik"

    parser_name = fulltext_config.analyzer
    properties = fulltext_config.properties or {}

    # Build SQL clause with parser name
    if properties:
        # Format parameters as key=value pairs
        # Quote string values, leave numbers and booleans as-is
        param_parts = []
        for k, v in properties.items():
            if isinstance(v, str):
                param_parts.append(f"{k}='{v}'")
            else:
                param_parts.append(f"{k}={v}")
        param_str = ", ".join(param_parts)
        return f"WITH PARSER {parser_name} PARSER_PROPERTIES=({param_str})"
    else:
        return f"WITH PARSER {parser_name}"


def _get_vector_index_sql(hnsw_config: HNSWConfiguration) -> str:
    """
    Generate VECTOR INDEX SQL clause from HNSWConfiguration.
    """
    properties = hnsw_config.properties or {}
    property_parts = []
    for k, v in properties.items():
        if isinstance(v, str):
            property_parts.append(f"{k}='{v}'")
        else:
            property_parts.append(f"{k}={v}")
    optional_fields = (
        ("M", hnsw_config.M),
        ("ef_construction", hnsw_config.ef_construction),
        ("ef_search", hnsw_config.ef_search),
        ("extra_info_max_size", hnsw_config.extra_info_max_size),
        ("refine_k", hnsw_config.refine_k),
        ("refine_type", hnsw_config.refine_type),
        ("bq_bits_query", hnsw_config.bq_bits_query),
        ("bq_use_fht", hnsw_config.bq_use_fht),
    )
    for key, value in optional_fields:
        if value is not None:
            if isinstance(value, str):
                property_parts.append(f"{key}='{value}'")
            elif isinstance(value, bool):
                property_parts.append(f"{key}={str(value).lower()}")
            else:
                property_parts.append(f"{key}={value}")
    property_str = ", ".join(property_parts)
    properties_str = f", {property_str}" if property_str else ""
    return f"WITH (DISTANCE={hnsw_config.distance}, TYPE={hnsw_config.type}, LIB={hnsw_config.lib}{properties_str})"


def _get_ivf_vector_index_sql(ivf_config: "IVFConfiguration") -> str:
    """Build the IVF vector index DDL fragment from an IVFConfiguration."""
    property_parts = []
    if ivf_config.properties:
        for k, v in ivf_config.properties.items():
            if isinstance(v, str):
                property_parts.append(f"{k}='{v}'")
            else:
                property_parts.append(f"{k}={v}")
    if ivf_config.centroids_fresh_mode is not None:
        property_parts.append(f"centroids_fresh_mode={ivf_config.centroids_fresh_mode}")
    property_str = ", ".join(property_parts)
    properties_str = f", {property_str}" if property_str else ""
    return f"WITH (DISTANCE={ivf_config.distance}, TYPE={ivf_config.type.upper()}, LIB={ivf_config.lib.upper()}{properties_str})"


def _get_sparse_vector_index_sql(sparse_config: SparseVectorIndexConfig) -> str:
    """
    Generate VECTOR INDEX SQL clause for sparse vector index from SparseVectorIndexConfig.

    Example output:
        WITH (DISTANCE=inner_product, TYPE=sindi, LIB=vsag)
    """
    parts = [
        f"DISTANCE={sparse_config.distance}",
        f"TYPE={sparse_config.type}",
        f"LIB={sparse_config.lib}",
    ]
    # Add optional parameters only if they differ from defaults
    if sparse_config.prune is not None:
        parts.append(f"prune={str(sparse_config.prune).lower()}")
    if sparse_config.refine is not None:
        parts.append(f"refine={str(sparse_config.refine).lower()}")
    if sparse_config.drop_ratio_build is not None:
        parts.append(f"drop_ratio_build={sparse_config.drop_ratio_build}")
    if sparse_config.drop_ratio_search is not None:
        parts.append(f"drop_ratio_search={sparse_config.drop_ratio_search}")
    if sparse_config.refine_k is not None:
        parts.append(f"refine_k={sparse_config.refine_k}")
    if sparse_config.properties:
        for k, v in sparse_config.properties.items():
            if isinstance(v, str):
                parts.append(f"{k}='{v}'")
            else:
                parts.append(f"{k}={v}")
    return f"WITH ({', '.join(parts)})"


def _embedding_to_hexstring(embedding: list[float]) -> str:
    """
    Convert embedding (list of floats) to a hex string.

    Args:
        embedding: List of floats

    Returns:
        Hex string representing the binary serialization of all floats in the list.
    """
    if not embedding:
        return ""
    # Pack as binary (float32 for compactness, common in vector DBs)
    binary = struct.pack(f"<{len(embedding)}f", *embedding)
    hexstr = binary.hex()
    return f"X'{hexstr}'"


class ClientAPI(ABC):
    """
    Client API interface for collection operations only.
    This is what end users interact with through the Client proxy.
    """

    @abstractmethod
    def create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        use_namespace: bool = False,
        **kwargs,
    ) -> "Collection":
        """
        Create collection

        Args:
            name: Collection name
            schema: Schema configuration for fine-grained index control, including
                   sparse vector index support. When provided, ``configuration`` and
                   ``embedding_function`` parameters are ignored.
            configuration: Index configuration (Configuration or HNSWConfiguration).
                          For backward compatibility, HNSWConfiguration is still accepted.
                          Configuration can include fulltext analyzer configuration (FulltextIndexConfig).
                          Ignored if ``schema`` is provided.
            embedding_function: Embedding function to convert documents to embeddings.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.
                               Ignored if ``schema`` is provided.
            use_namespace: If True, create a namespace-enabled collection. Defaults to False.
            **kwargs: Additional parameters
        """
        pass

    @abstractmethod
    def get_collection(self, name: str, embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED) -> "Collection":
        """Get an existing collection.

        Args:
            name: The name of the collection to retrieve.
            embedding_function: The embedding function to use. If not provided,
                it will try to load the function used when creating the collection.
                If explicitly set to None, no embedding function will be used.

        Returns:
            The ``Collection`` object.

        Raises:
            ValueError: If the collection does not exist.

        Examples:
            >>> collection = client.get_collection("my_collection")
        """
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete collection"""
        pass

    @abstractmethod
    def list_collections(self) -> list["Collection"]:
        """List all collections"""
        pass

    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """Check if collection exists"""
        pass


@dataclass
class _CollectionMeta:
    """
    Collection metadata in sdk_collections table.
    """

    collection_id: str
    collection_name: str
    settings: str | None

    @staticmethod
    def from_row(row: Any) -> "_CollectionMeta":
        """Construct a _CollectionMeta from a catalog row (dict for server, tuple for embedded)."""
        if isinstance(row, dict):
            # Server client returns dict, get the first value
            collection_id = row["COLLECTION_ID"]
            collection_name = row["COLLECTION_NAME"]
            settings = row["SETTINGS"]
        elif isinstance(row, (tuple, list)):
            # Embedded client returns tuple, first element is collection id
            collection_id = row[0] if len(row) > 0 else ""
            collection_name = row[1] if len(row) > 1 else ""
            settings = row[2] if len(row) > 2 else ""
        else:
            raise TypeError(f"Unsupported sdk_collections row type: {type(row).__name__}")
        return _CollectionMeta(collection_id=collection_id, collection_name=collection_name, settings=settings)


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

    # ==================== Database Type Detection ====================

    def _validate_ob_database_type(self) -> None:
        """Validate that the backend is LakeBase and meets the minimum version for namespaces."""
        db_type, version = self.detect_db_type_and_version()
        if db_type.lower() != "oceanbase":
            raise ValueError("use_namespace=True is only supported on LakeBase (OceanBase Database AI)")
        if not self._is_lakebase_cluster():
            raise ValueError(
                "use_namespace=True is only supported on LakeBase (OceanBase Database AI); "
                "the connected cluster is standard OceanBase"
            )
        if version < NAMESPACE_MIN_LAKEBASE_VERSION:
            raise ValueError(
                f"use_namespace=True requires LakeBase version >= {NAMESPACE_MIN_LAKEBASE_VERSION}, "
                f"current version is {version}"
            )

    def _is_lakebase_cluster(self) -> bool:
        """Return whether the connected OceanBase cluster is LakeBase (OceanBase Database AI)."""
        cached = getattr(self, "_lakebase_cluster", None)
        if cached is not None:
            return cached
        result = False
        try:
            rows = self._execute("SELECT version() AS version")
            if rows:
                row = rows[0]
                if isinstance(row, dict):
                    version_str = row.get("version") or row.get("VERSION") or ""
                elif isinstance(row, (tuple, list)) and row:
                    version_str = row[0]
                else:
                    version_str = str(row)
                result = is_lakebase_version_string(str(version_str))
        except Exception:
            result = False
        self._lakebase_cluster = result
        return result

    def _is_shared_storage_mode(self) -> bool:
        """Return whether the OceanBase deployment runs in shared-storage mode."""
        cached = getattr(self, "_shared_storage", None)
        if cached is not None:
            return cached
        result = False
        try:
            rows = self._execute("SELECT VALUE FROM oceanbase.GV$OB_PARAMETERS WHERE name = 'ob_startup_mode'")
            if rows:
                val = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["VALUE"]
                result = str(val).upper() == "SHARED_STORAGE"
        except Exception:
            result = False
        self._shared_storage = result
        return result

    def _stg_cache_policy_clause(self) -> str:
        """SS mode: make catalog tables global-hot so metadata is locally cached.

        STORAGE_CACHE_POLICY is only supported in shared-storage mode; SN mode
        returns an empty string.
        """
        if self._is_shared_storage_mode():
            return 'STORAGE_CACHE_POLICY = (GLOBAL = "hot")'
        return ""

    def detect_db_type_and_version(self) -> tuple[str, "Version"]:
        """
        Detect database type and version.

        Works for all three modes: seekdb-embedded, seekdb-server, and oceanbase.
        Version detection is case-insensitive for seekdb.

        Returns:
            (db_type, version): ("seekdb", Version("x.x.x.x")) or ("oceanbase", Version("x.x.x.x"))

        Raises:
            ValueError: If unable to detect database type or version

        Examples:
            >>> db_type, version = client.detect_db_type_and_version()
            >>> version > Version("1.0.0.0")
            True
        """
        from .version import Version

        def _get_value(result, key: str) -> str | None:
            """Extract value from query result"""
            if not result or len(result) == 0:
                return None
            row = result[0]
            if isinstance(row, dict):
                value = row.get(key, "")
            elif isinstance(row, (tuple, list)) and len(row) > 0:
                value = row[0]
            else:
                value = str(row)
            return str(value).strip() if value else None

        def _query(sql: str, key: str) -> str | None:
            """Execute SQL and return value"""
            try:
                result = self._execute(sql)
                return _get_value(result, key)
            except Exception as e:
                logger.debug(f"Failed to execute {sql}: {e}")
                return None

        def _extract_seekdb_version(version_str: str) -> str | None:
            """Extract version from seekdb version string (case-insensitive)"""
            # Use case-insensitive pattern matching
            for pattern in [
                r"seekdb[-\s]v?(\d+\.\d+\.\d+\.\d+)",
                r"seekdb[-\s]v?(\d+\.\d+\.\d+)",
            ]:
                match = re.search(pattern, version_str, re.IGNORECASE)
                if match:
                    return match.group(1)
            return None

        # Ensure connection is established
        self._ensure_connection()

        # Check version() for seekdb (case-insensitive)
        version_result = _query("SELECT version() as version", "version")
        if version_result and re.search(r"seekdb", version_result, re.IGNORECASE):
            seekdb_version_str = _extract_seekdb_version(version_result)
            if seekdb_version_str:
                return ("seekdb", Version(seekdb_version_str))
            else:
                raise ValueError(f"Detected seekdb in version string, but failed to extract version: {version_result}")
        # Query ob_version() for OceanBase
        ob_version_str = _query("SELECT ob_version() as ob_version", "ob_version")
        if ob_version_str:
            # Try to parse OceanBase version (may have different format)
            try:
                return ("oceanbase", Version(ob_version_str))
            except ValueError as e:
                # If OceanBase version doesn't match standard format, try to extract numeric parts
                parts = re.findall(r"\d+", ob_version_str)
                if len(parts) >= 3:
                    # Take first 3 or 4 parts
                    version_str = ".".join(parts[:4] if len(parts) >= 4 else parts[:3])
                    return ("oceanbase", Version(version_str))
                else:
                    # Fallback: return as-is but wrap in Version with minimal format
                    # This handles edge cases where version format is unusual
                    raise ValueError(f"Unable to parse OceanBase version: {ob_version_str}") from e

        # Truncate potentially verbose or sensitive database responses in error message
        def _truncate(val, length=20):
            """Truncate a value to a short string for logging."""
            if val is None:
                return "None"
            val_str = str(val)
            return val_str[:length] + ("..." if len(val_str) > length else "")

        raise ValueError(
            f"Unable to detect database type. version()={_truncate(version_result)}, "
            f"ob_version()={_truncate(ob_version_str)}"
        )

    # ==================== Database Management (User-facing) ====================

    def _database_tenant(self, tenant: str) -> str | None:
        """Resolve effective tenant for database operations."""
        return None

    def _database_context(self, tenant: str | None) -> str:
        """Yield a context with the active database selected for the connection."""
        return f" in tenant: {tenant}" if tenant else ""

    def _parse_schema_row(self, row: Any) -> tuple[str | None, str | None, str | None]:
        """Parse a raw catalog row into a schema descriptor."""
        if isinstance(row, dict):
            return (
                row.get("SCHEMA_NAME"),
                row.get("DEFAULT_CHARACTER_SET_NAME"),
                row.get("DEFAULT_COLLATION_NAME"),
            )
        if isinstance(row, (tuple, list)):
            name = row[0] if len(row) > 0 else None
            charset = row[1] if len(row) > 1 else None
            collation = row[2] if len(row) > 2 else None
            return name, charset, collation
        return None, None, None

    def create_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Create database

        Args:
            name: database name
            tenant: tenant name (for OceanBase)
        """
        effective_tenant = self._database_tenant(tenant)
        logger.debug(f"Creating database: {name}{self._database_context(effective_tenant)}")
        sql = f"CREATE DATABASE IF NOT EXISTS `{name}`"
        self._execute(sql)
        logger.debug(f"✅ Database created: {name}{self._database_context(effective_tenant)}")

    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Get database object

        Args:
            name: database name
            tenant: tenant name (for OceanBase)
        """
        effective_tenant = self._database_tenant(tenant)
        logger.debug(f"Getting database: {name}{self._database_context(effective_tenant)}")
        sql = (
            "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME "
            "FROM information_schema.SCHEMATA "
            f"WHERE SCHEMA_NAME = '{name}'"
        )
        result = self._execute(sql)

        if not result:
            raise ValueError(f"Database not found: {name}")

        db_name, charset, collation = self._parse_schema_row(result[0])
        if not db_name:
            raise ValueError(f"Database not found: {name}")

        return Database(
            name=db_name,
            tenant=effective_tenant,
            charset=charset,
            collation=collation,
        )

    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Delete database

        Args:
            name: database name
            tenant: tenant name (for OceanBase)
        """
        effective_tenant = self._database_tenant(tenant)
        logger.debug(f"Deleting database: {name}{self._database_context(effective_tenant)}")
        sql = f"DROP DATABASE IF EXISTS `{name}`"
        self._execute(sql)
        logger.debug(f"✅ Database deleted: {name}{self._database_context(effective_tenant)}")

    def list_databases(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tenant: str = DEFAULT_TENANT,
    ) -> Sequence[Database]:
        """
        List all databases

        Args:
            limit: maximum number of results to return
            offset: number of results to skip
            tenant: tenant name (for OceanBase)
        """
        effective_tenant = self._database_tenant(tenant)
        logger.debug(f"Listing databases{self._database_context(effective_tenant)}")
        sql = "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA"

        if limit is not None:
            if offset is not None:
                sql += f" LIMIT {offset}, {limit}"
            else:
                sql += f" LIMIT {limit}"

        result = self._execute(sql)

        databases = []
        for row in result:
            db_name, charset, collation = self._parse_schema_row(row)
            if not db_name:
                continue
            databases.append(
                Database(
                    name=db_name,
                    tenant=effective_tenant,
                    charset=charset,
                    collation=collation,
                )
            )

        logger.debug(f"✅ Found {len(databases)} databases{self._database_context(effective_tenant)}")
        return databases

    def fork_database(self, source_name: str, destination_name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Fork (duplicate) a database to create a new independent copy.

        Args:
            source_name: source database name
            destination_name: destination database name (must not already exist)
            tenant: tenant name (for OceanBase)

        Returns:
            Database object for the newly created destination database
        """
        if not self._fork_database_enabled():
            raise ValueError("Fork database is not enabled (requires seekdb >= 1.2.0)")

        effective_tenant = self._database_tenant(tenant)
        logger.debug(f"Forking database: {source_name} -> {destination_name}{self._database_context(effective_tenant)}")
        sql = f"FORK DATABASE `{source_name}` TO `{destination_name}`"
        try:
            self._execute(sql)
        except Exception as ex:
            args = getattr(ex, "args", ())
            if args and isinstance(args[0], int) and args[0] == 1007:
                raise ValueError(f"Database '{destination_name}' already exists") from ex

            msg = str(ex).lower()
            if ("database exists" in msg or "already exists" in msg) and "database" in msg:
                raise ValueError(f"Database '{destination_name}' already exists") from ex
            raise
        logger.debug(f"✅ Successfully forked database '{source_name}' to '{destination_name}'")
        return self.get_database(destination_name, tenant=tenant)

    # ==================== Collection Management (User-facing) ====================

    def _prepare_schema_parameters(
        self,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
    ) -> Schema:
        # Handle embedding function first
        # If not provided (sentinel), use default embedding function
        """Normalize and validate schema parameters before creating a collection."""
        if embedding_function is _NOT_PROVIDED:
            embedding_function = get_default_embedding_function()

        # Calculate actual dimension from embedding function if provided
        actual_dimension = None
        if embedding_function is not None:
            try:
                # First, try to get dimension from the embedding function's dimension property
                # This avoids initializing the model (e.g., onnxruntime) during collection creation
                if hasattr(embedding_function, "dimension"):
                    actual_dimension = embedding_function.dimension
                    logger.debug(f"Using embedding function dimension: {actual_dimension}")
                else:
                    # Fallback: if no dimension attribute, call the function to calculate dimension
                    # This may trigger model initialization, but is necessary for custom embedding functions
                    test_embeddings = embedding_function.__call__("seekdb")
                    if test_embeddings and len(test_embeddings) > 0:
                        actual_dimension = len(test_embeddings[0])
                        logger.info(f"Calculated embedding function dimension: {actual_dimension}")
                    else:
                        raise ValueError(  # noqa: TRY301
                            "Embedding function returned empty result when called with 'seekdb'"
                        )
            except Exception as e:
                raise ValueError(
                    f"Failed to get dimension from embedding function: {e}. "
                    f"Please ensure the embedding function has a 'dimension' attribute or can be called with a string input."
                ) from e

        # Handle configuration
        # Extract HNSWConfiguration from ConfigurationParam (handles both Configuration and HNSWConfiguration)
        hnsw_config = None

        if configuration is _NOT_PROVIDED:
            # Use default configuration, but if embedding_function is provided, use its dimension
            if actual_dimension is not None:
                hnsw_config = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
            else:
                hnsw_config = HNSWConfiguration(dimension=DEFAULT_VECTOR_DIMENSION, distance=DEFAULT_DISTANCE_METRIC)
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
                hnsw_config = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
            else:
                raise ValueError(
                    "Failed to calculate dimension from embedding function. "
                    "Please ensure the embedding function can be called with a string input."
                )
        else:
            # Extract HNSWConfiguration from Configuration or use HNSWConfiguration directly
            hnsw_config = _extract_hnsw_config(configuration)

            # If Configuration was provided but hnsw is None, create default HNSWConfiguration
            if hnsw_config is None:
                if actual_dimension is not None:
                    hnsw_config = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
                else:
                    hnsw_config = HNSWConfiguration(
                        dimension=DEFAULT_VECTOR_DIMENSION,
                        distance=DEFAULT_DISTANCE_METRIC,
                    )

        # If embedding_function is provided, validate configuration dimension matches
        if embedding_function is not None and actual_dimension is not None:
            if hnsw_config.dimension != actual_dimension:
                raise ValueError(
                    f"Configuration dimension ({hnsw_config.dimension}) doesn't match "
                    f"embedding function dimension ({actual_dimension}). "
                    f"Please update configuration to use dimension={actual_dimension} or remove dimension from configuration."
                )
            # Use actual dimension from embedding function
            dimension = actual_dimension
        else:
            # No embedding function, use configuration dimension
            dimension = hnsw_config.dimension

        hnsw_config.dimension = dimension
        fulltext_config = _extract_fulltext_config(configuration)
        vic = VectorIndexConfig(hnsw=hnsw_config, embedding_function=embedding_function)
        return Schema(
            vector_index=vic,
            fulltext_index=fulltext_config,
        )

    def create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        use_namespace: bool = False,
        partition_count: int | None = None,
        **kwargs,
    ) -> "Collection":
        """Create a new collection.

        Args:
            name: The name of the collection to create. Must contain only alphanumeric
                characters or underscores.
            schema: Schema configuration. Defaults to None (uses default schema). Can be a ``Schema`` object.
            configuration: Index configuration. Defaults to None (uses HNSW with
                Cosine distance and dimension 384). Can be a ``Configuration`` or
                ``HNSWConfiguration`` object. If set to None, the dimension will be
                inferred from the embedding function.
            embedding_function: The embedding function to use for this collection.
                Defaults to ``DefaultEmbeddingFunction`` (all-MiniLM-L6-v2). If set to None,
                no embedding function will be used (embeddings must be provided manually).
            use_namespace: If True, create a namespace-enabled collection. Defaults to False.
            partition_count: Number of partitions for the namespace physical tables.
                Only valid when ``use_namespace=True``. Defaults to 1000 when not provided.
                Passing it for a non-namespace collection raises ``ValueError``.
            **kwargs: Additional parameters for collection creation.

        Returns:
            The created ``Collection`` object.

        Raises:
            ValueError: If the collection name is invalid, already exists, or if the
                configuration/embedding function combination is invalid (e.g., dimension mismatch).
            TypeError: If the configuration object is of an invalid type.

        Examples:
            Create a collection with default settings:

            >>> client.create_collection("my_collection")

            Create a collection with a custom embedding function:

            >>> from pyseekdb import DefaultEmbeddingFunction
            >>> ef = DefaultEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            >>> collection = client.create_collection("my_docs", embedding_function=ef)

            Create a collection with specific configuration:

            >>> from pyseekdb import HNSWConfiguration
            >>> config = HNSWConfiguration(dimension=128, distance="l2")
            >>> collection = client.create_collection(
            ...     "custom_config",
            ...     configuration=config,
            ...     embedding_function=None
            ... )
        """
        _validate_collection_name(name)
        if partition_count is not None and not use_namespace:
            raise ValueError(
                "partition_count is only supported for namespace-enabled collections (use_namespace=True)."
            )
        # Only fully initialized collections (metadata + physical table) count as existing.
        # Metadata without a table is treated as an incomplete create and repaired below.
        if self.has_collection(name) and not (use_namespace and self._is_incomplete_ns_collection(name)):
            raise ValueError(f"Collection '{name}' already exists")

        # Resolve schema: either use the provided schema or build one from legacy params
        if schema is not None:
            if configuration is not _NOT_PROVIDED or embedding_function is not _NOT_PROVIDED:
                warnings.warn(
                    "schema and configuration/embedding_function are both provided, schema will be used",
                    stacklevel=2,
                )
        elif use_namespace:
            raise ValueError(
                "use_namespace=True requires an explicit Schema with an IVF vector index. "
                "When schema is omitted, create_collection builds the default non-namespace "
                "schema (HNSW), which namespace collections do not support. "
                "Pass schema=Schema(vector_index=VectorIndexConfig("
                "ivf=IVFConfiguration(dimension=..., distance=...)), ...). "
                "Or set use_namespace=False for a standard HNSW collection."
            )
        else:
            # Legacy path: convert configuration + embedding_function into a Schema
            schema = self._prepare_schema_parameters(configuration, embedding_function)

        logger.debug(f"schema: {schema}")

        if use_namespace:
            return self._create_namespace_collection(name, schema, partition_count=partition_count, **kwargs)

        # Resolve HNSW configuration dimension if not set
        hnsw_config = schema.vector_index.hnsw
        dense_embedding_function = schema.vector_index.embedding_function
        if hnsw_config is None:
            # Determine dimension from embedding function
            actual_dimension = self._get_embedding_function_dimension(dense_embedding_function)
            hnsw_config = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
        else:
            # Validate dimension matches embedding function if available
            if dense_embedding_function is not None:
                actual_dimension = self._get_embedding_function_dimension(dense_embedding_function)
                if hnsw_config.dimension != actual_dimension:
                    raise ValueError(
                        f"Configuration dimension ({hnsw_config.dimension}) doesn't match "
                        f"embedding function dimension ({actual_dimension})."
                    )

        dimension = hnsw_config.dimension
        if dimension < 1 or dimension > MAX_HNSW_VECTOR_DIMENSION:
            raise ValueError(f"Dimension must be between 1 and {MAX_HNSW_VECTOR_DIMENSION}, got {dimension}")

        # Extract fulltext parser configuration
        fulltext_index_clause = _get_fulltext_index_sql(schema.fulltext_index)

        # Sparse vector index
        sparse_vector_index_config = schema.sparse_vector_index
        sparse_field_sql = (
            f"{CollectionFieldNames.SPARSE_EMBEDDING} SPARSEVECTOR,\n" if sparse_vector_index_config else ""
        )
        sparse_index_sql = (
            f",\n            VECTOR INDEX idx_sparse ({CollectionFieldNames.SPARSE_EMBEDDING}) {_get_sparse_vector_index_sql(sparse_vector_index_config)}"
            if sparse_vector_index_config
            else ""
        )

        # Construct table name
        collection_id = None
        if kwargs.get("_collection_version", 2) == 1:
            # for testing purpose
            table_name = self._create_collection_meta_v1(name)
        else:
            collection_meta = self._create_collection_meta_v2(
                name, dense_embedding_function, sparse_vector_index_config=sparse_vector_index_config
            )
            collection_id = collection_meta.get("collection_id")
            table_name = collection_meta["table_name"]

        # Construct CREATE TABLE SQL statement with HEAP organization
        sql = f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
            _id varbinary(512) PRIMARY KEY NOT NULL,
            document string,
            embedding vector({dimension}),
            {sparse_field_sql}metadata json,
            FULLTEXT INDEX idx_fts(document) {fulltext_index_clause},
            VECTOR INDEX idx_vec (embedding) {_get_vector_index_sql(hnsw_config)}{sparse_index_sql}
        ) ORGANIZATION = HEAP;"""

        # Execute SQL to create table
        logger.debug(f"Creating table: {table_name} with SQL: {sql}")
        self._execute(sql)

        # Create and return Collection object
        return Collection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=dimension,
            embedding_function=schema.vector_index.embedding_function,
            distance=hnsw_config.distance,
            sparse_vector_index_config=sparse_vector_index_config,
            **kwargs,
        )

    def _create_namespace_collection(
        self, name: str, schema: Schema, partition_count: int | None = None, **kwargs
    ) -> "Collection":
        """Create a namespace-enabled collection and its catalog/physical tables."""
        dense_embedding_function = schema.vector_index.embedding_function
        ivf_config = schema.vector_index.ivf
        hnsw_config = schema.vector_index.hnsw

        if schema.sparse_vector_index is not None:
            raise ValueError(
                "use_namespace=True does not support SparseVectorIndexConfig yet. "
                "Remove sparse_vector_index from the schema or set use_namespace=False."
            )
        if hnsw_config is not None:
            raise ValueError(
                "use_namespace=True does not support HNSW. Namespace collections require an IVF "
                "schema: Schema(vector_index=VectorIndexConfig(ivf=IVFConfiguration(dimension=..., "
                "distance=...)), ...)."
            )
        if ivf_config is not None and ivf_config.type != IVFIndexType.IVF_FLAT.value:
            raise ValueError(
                f"use_namespace=True currently only supports IVF index type '{IVFIndexType.IVF_FLAT.value}', "
                f"got '{ivf_config.type}'"
            )
        self._validate_ob_database_type()

        # Resume an incomplete collection (meta row exists but physical tables are
        # missing, e.g. a crash interrupted creation): reuse its id/settings and
        # idempotently (re)build only the missing physical tables. Otherwise create
        # from scratch. Both paths are safe to call repeatedly.
        existing = self._get_ns_collection_meta(name)
        if existing is not None:
            collection_id = existing["collection_id"]
            settings = existing.get("settings", {})
            dimension = settings.get("dimension")
            distance = settings.get("distance", DEFAULT_DISTANCE_METRIC)
            pc = int(settings.get("partition_count", _DEFAULT_PARTITION_COUNT))
            is_ss = settings.get("storage_mode") == "ss"
            logger.info(
                f"Namespace collection '{name}' already has a catalog entry; resuming creation "
                f"(reusing collection_id={collection_id}, rebuilding any missing physical tables)."
            )
        else:
            pc = _DEFAULT_PARTITION_COUNT if partition_count is None else partition_count
            if pc < 1:
                raise ValueError("partition_count must be >= 1")
            if ivf_config is not None:
                dimension = ivf_config.dimension
                distance = ivf_config.distance
            else:
                if dense_embedding_function is not None:
                    dimension = self._get_embedding_function_dimension(dense_embedding_function)
                else:
                    dimension = DEFAULT_VECTOR_DIMENSION
                distance = DEFAULT_DISTANCE_METRIC

            is_ss = self._is_shared_storage_mode()
            settings = {
                "version": 2,
                "use_namespace": True,
                "storage_mode": "ss" if is_ss else "sn",
                "dimension": dimension,
                "distance": distance,
                "partition_count": pc,
            }
            if ivf_config is not None:
                settings["dense_index_type"] = "ivf"
                if ivf_config.centroids_fresh_mode is not None:
                    settings["centroids_fresh_mode"] = ivf_config.centroids_fresh_mode
            if schema.fulltext_index is not None:
                settings["has_fulltext_index"] = True
            if dense_embedding_function is not None and EmbeddingFunction.support_persistence(dense_embedding_function):
                settings["embedding_function"] = {
                    "name": dense_embedding_function.name(),
                    "properties": dense_embedding_function.get_config(),
                }

            collection_meta = self._create_ns_collection_meta(name, settings)
            collection_id = collection_meta["collection_id"]

        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(f"dimension must be an integer, got {type(dimension).__name__}")
        if dimension < 1 or dimension > MAX_IVF_VECTOR_DIMENSION:
            raise ValueError(
                f"Dimension must be between 1 and {MAX_IVF_VECTOR_DIMENSION} for namespace "
                f"IVF collections, got {dimension}"
            )

        self._ensure_namespace_catalogs()

        try:
            self._create_namespace_physical_tables(
                collection_id=collection_id,
                dimension=dimension,
                ivf_config=ivf_config,
                fulltext_config=schema.fulltext_index,
                is_shared_storage=is_ss,
                partition_count=pc,
                # On a fresh create, roll back partial tables so a failure leaves no
                # trace. On resume, keep whatever already exists so a later retry can
                # finish the job.
                cleanup_on_error=existing is None,
            )
        except Exception:
            if existing is None:
                with contextlib.suppress(Exception):
                    collection_id_escaped = escape_string(collection_id)
                    self._execute(
                        f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` "
                        f"WHERE collection_id = '{collection_id_escaped}'"
                    )
            raise

        return Collection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=dimension,
            embedding_function=dense_embedding_function,
            distance=distance,
            use_namespace=True,
            partition_count=pc,
            has_vector_index=settings.get("dense_index_type") == "ivf",
        )

    def _get_embedding_function_dimension(self, embedding_function: EmbeddingFunction) -> int:
        """Get the dimension from an embedding function."""
        try:
            if hasattr(embedding_function, "dimension"):
                dim = embedding_function.dimension
                logger.debug(f"Using embedding function dimension: {dim}")
                return dim
            else:
                test_embeddings = embedding_function.__call__("seekdb")
                if test_embeddings and len(test_embeddings) > 0:
                    dim = len(test_embeddings[0])
                    logger.info(f"Calculated embedding function dimension: {dim}")
                    return dim
                else:
                    raise ValueError(  # noqa: TRY301
                        "Embedding function returned empty result when called with 'seekdb'"
                    )
        except Exception as e:
            raise ValueError(
                f"Failed to get dimension from embedding function: {e}. "
                f"Please ensure the embedding function has a 'dimension' attribute or can be called with a string input."
            ) from e

    def _create_sdk_collections_if_not_exists(self) -> None:
        """Create the sdk_collections catalog table if it does not already exist."""
        try:
            self._use_catalog_database()
            sdk_coll = self._qtable(CollectionNames.sdk_collections_table_name())
            scp = self._stg_cache_policy_clause()
            create_table_sql = f"""CREATE TABLE IF NOT EXISTS {sdk_coll} (
                collection_id CHAR(32) PRIMARY KEY DEFAULT (replace(uuid(), '-', '')),
                collection_name STRING,
                settings JSON COMMENT "Generated by SDK, don't modify",
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_sdk_coll_name (collection_name)
            ) COMMENT='Settings of collections created by SDK' ORGANIZATION INDEX {scp};"""
            self._execute(create_table_sql)
            try:
                self._execute(f"CREATE UNIQUE INDEX uk_sdk_coll_name ON {sdk_coll} (collection_name)")
            except Exception as exc:
                _reraise_unless_unique_index_exists(exc)
        except Exception as e:
            raise ValueError(f"Failed to create sdk_collections table: {e}") from e

    def _create_collection_meta_v2(
        self,
        collection_name: str,
        embedding_function,
        sparse_vector_index_config: SparseVectorIndexConfig | None = None,
    ) -> dict[str, str]:
        """Insert collection metadata into sdk_collections, resolving unique-key conflicts idempotently."""
        try:
            results = {}
            settings = {"version": 2}
            if embedding_function is not None and EmbeddingFunction.support_persistence(embedding_function):
                settings["embedding_function"] = {
                    "name": embedding_function.name(),
                    "properties": embedding_function.get_config(),
                }

            # Persist sparse vector index config
            if sparse_vector_index_config is not None:
                sparse_settings = {}
                source_key = sparse_vector_index_config.source_key
                if source_key is not None:
                    # Convert FieldKey to string for serialization
                    sparse_settings["source_key"] = source_key.name if hasattr(source_key, "name") else str(source_key)
                sparse_ef = sparse_vector_index_config.embedding_function
                if sparse_ef is not None and SparseEmbeddingFunction.support_persistence(sparse_ef):
                    sparse_settings["embedding_function"] = {
                        "name": sparse_ef.name(),
                        "properties": sparse_ef.get_config(),
                    }
                settings["sparse_vector_index"] = sparse_settings

            settings_str = escape_string(json.dumps(settings))

            self._create_sdk_collections_if_not_exists()
            collection_name_in_table = escape_string(collection_name)
            try:
                collection_id = self._get_collection_id(collection_name)
            except ValueError:
                insert_sql = (
                    f"INSERT INTO `{CollectionNames.sdk_collections_table_name()}` "
                    f"(COLLECTION_NAME, SETTINGS) VALUES ('{collection_name_in_table}', '{settings_str}')"
                )
                try:
                    self._execute(insert_sql)
                except Exception as exc:
                    if not _is_sdk_collection_catalog_conflict_error(exc):
                        raise
                    conn_getter = getattr(self, "_ensure_connection", None)
                    if conn_getter is not None:
                        with contextlib.suppress(Exception):
                            conn_getter().rollback()
                collection_id = self._get_collection_id(collection_name)

            results["collection_id"] = collection_id
            results["table_name"] = CollectionNames.table_name_v2(collection_id)
            return results  # noqa: TRY300
        except Exception as e:
            raise ValueError(f"Failed to create collection metadata: {e}") from e

    def _create_collection_meta_v1(self, collection_name: str) -> str:
        """Insert collection metadata using the legacy v1 catalog layout."""
        try:
            table_name = CollectionNames.table_name(collection_name)
            return table_name  # noqa: TRY300
        except Exception as e:
            raise ValueError(f"Failed to create collection metadata: {e}") from e

    # ==================== Namespace Catalog Methods ====================

    def _catalog_database(self) -> str:
        """Database that holds sdk_* catalog tables (must match OB bg scheduler scan target)."""
        db = getattr(self, "database", None)
        if not db:
            raise ValueError("client database is not configured")
        _validate_database_name(db)
        return db

    def _qtable(self, table: str) -> str:
        """Fully-qualified catalog table: `{database}`.`{table}`."""
        db = _quote_sql_identifier(self._catalog_database())
        table_quoted = _quote_sql_identifier(table)
        return f"{db}.{table_quoted}"

    def _use_catalog_database(self) -> None:
        """Align session with pymysql database= so PL (DROP_NAMESPACE) uses the same DB."""
        self._execute(f"USE {_quote_sql_identifier(self._catalog_database())}")

    def _ensure_namespace_catalogs(self) -> None:
        """Create the sdk_namespaces and sdk_ltables catalog tables and their unique indexes."""
        self._use_catalog_database()
        ns_namespaces_q = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        ns_ltables_q = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        scp = self._stg_cache_policy_clause()
        ns_namespaces_sql = f"""CREATE TABLE IF NOT EXISTS {ns_namespaces_q} (
            namespace_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            collection_id CHAR(32) NOT NULL,
            namespace_name VARCHAR(256) NOT NULL,
            created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            info JSON,
            PRIMARY KEY (namespace_id),
            UNIQUE KEY uk_sdk_ns_coll_name (collection_id, namespace_name),
            KEY idx_sdk_ns_by_collection (collection_id)
        ) COMMENT='Namespace catalog' ORGANIZATION INDEX {scp};"""
        ns_ltables_sql = f"""CREATE TABLE IF NOT EXISTS {ns_ltables_q} (
            ltable_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            collection_id CHAR(32) NOT NULL,
            namespace_id BIGINT UNSIGNED NOT NULL,
            ltable_name VARCHAR(256) NOT NULL DEFAULT 'default',
            created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            info JSON,
            PRIMARY KEY (ltable_id),
            UNIQUE KEY uk_sdk_lt_coll_ns_name (collection_id, namespace_id, ltable_name),
            KEY idx_sdk_lt_by_ns (collection_id, namespace_id)
        ) COMMENT='LTable catalog' ORGANIZATION INDEX {scp};"""
        namespaces_stats_sql = f"""CREATE TABLE IF NOT EXISTS {self._qtable(NamespaceCollectionNames.sdk_namespaces_stats_table())} (
            collection_id CHAR(32) NOT NULL COMMENT 'collection id',
            namespace_id BIGINT UNSIGNED NOT NULL COMMENT 'namespace internal id',
            ltable_id BIGINT UNSIGNED NOT NULL COMMENT 'logic table internal id, 0 means namespace summary',
            estimated_rows BIGINT NOT NULL DEFAULT 0 COMMENT 'estimated row count',
            average_row_size BIGINT NOT NULL DEFAULT 0 COMMENT 'average row size in bytes',
            row_limit BIGINT NOT NULL DEFAULT -1 COMMENT 'row count limit, -1 means unlimited',
            size_limit BIGINT NOT NULL DEFAULT -1 COMMENT 'storage size limit in bytes, -1 means unlimited',
            last_estimate_time TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'last estimate time',
            included_index BOOL NOT NULL DEFAULT FALSE COMMENT 'whether stats include index data',
            PRIMARY KEY (namespace_id, ltable_id, included_index),
            KEY idx_sdk_ns_stat_by_collection (collection_id)
        ) COMMENT='Logic table row count and storage size statistics' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX
        PARTITION BY KEY(namespace_id) PARTITIONS 8;"""
        self._execute(ns_namespaces_sql)
        self._execute(ns_ltables_sql)
        self._execute(namespaces_stats_sql)
        try:
            self._execute(
                f"CREATE UNIQUE INDEX uk_sdk_ns_coll_name ON {ns_namespaces_q} (collection_id, namespace_name)"
            )
        except Exception as exc:
            _reraise_unless_unique_index_exists(exc)
        try:
            self._execute(
                f"CREATE UNIQUE INDEX uk_sdk_lt_coll_ns_name ON {ns_ltables_q} "
                f"(collection_id, namespace_id, ltable_name)"
            )
        except Exception as exc:
            _reraise_unless_unique_index_exists(exc)

    def _rollback_connection_if_supported(self) -> None:
        """Roll back the current connection transaction if the backend supports it."""
        conn_getter = getattr(self, "_ensure_connection", None)
        if conn_getter is not None:
            with contextlib.suppress(Exception):
                conn_getter().rollback()

    def _create_ns_collection_meta(self, collection_name: str, settings: dict) -> dict:
        """Insert namespace collection metadata, resolving unique-key conflicts idempotently."""
        self._create_sdk_collections_if_not_exists()
        settings_str = escape_string(json.dumps(settings))
        collection_name_escaped = escape_string(collection_name)
        sdk_coll = self._qtable(CollectionNames.sdk_collections_table_name())
        insert_sql = (
            f"INSERT INTO {sdk_coll} (collection_name, settings) VALUES ('{collection_name_escaped}', '{settings_str}')"
        )
        try:
            self._execute(insert_sql)
        except Exception as exc:
            if not _is_sdk_collection_catalog_conflict_error(exc):
                raise
            conn_getter = getattr(self, "_ensure_connection", None)
            if conn_getter is not None:
                with contextlib.suppress(Exception):
                    conn_getter().rollback()
        rows = self._execute(
            f"SELECT collection_id FROM {sdk_coll} WHERE collection_name = '{collection_name_escaped}'"
        )
        collection_id = str(rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["collection_id"])
        self._set_session_ns_context(collection_id=collection_id)
        return {"collection_id": collection_id, "collection_name": collection_name}

    def _get_ns_collection_meta(self, collection_name: str) -> dict | None:
        """Fetch namespace collection metadata from the catalog."""
        collection_name_escaped = escape_string(collection_name)
        try:
            rows = self._execute(
                f"SELECT collection_id, collection_name, settings "
                f"FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
                f"WHERE collection_name = '{collection_name_escaped}'"
            )
        except Exception:
            return None
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, (list, tuple)):
            settings = json.loads(row[2]) if row[2] else {}
        else:
            settings = json.loads(row["settings"]) if row.get("settings") else {}
        if not settings.get("use_namespace"):
            return None
        if isinstance(row, (list, tuple)):
            return {
                "collection_id": str(row[0]),
                "collection_name": row[1],
                "settings": settings,
            }
        return {
            "collection_id": str(row["collection_id"]),
            "collection_name": row["collection_name"],
            "settings": settings,
        }

    def _has_ns_collection(self, collection_name: str) -> bool:
        """Return whether a namespace collection with the given name exists."""
        try:
            return self._get_ns_collection_meta(collection_name) is not None
        except Exception:
            return False

    def _ns_collection_exists_by_id(self, collection_id: str) -> bool:
        """Whether a namespace-enabled collection with this id still exists.

        Used to reject namespace operations on a stale Collection handle whose
        underlying collection was deleted (the in-memory handle keeps its old id).
        """
        collection_id_escaped = escape_string(str(collection_id))
        rows = self._execute(
            f"SELECT collection_id FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            f"WHERE collection_id = '{collection_id_escaped}'"
        )
        return bool(rows)

    def _delete_ns_collection_meta(self, collection_name: str) -> None:
        """Delete namespace collection metadata from the catalog."""
        meta = self._get_ns_collection_meta(collection_name)
        if meta is None:
            raise ValueError(f"Namespace collection '{collection_name}' not found")
        collection_id = meta["collection_id"]
        collection_id_escaped = escape_string(collection_id)
        self._execute(
            f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` "
            f"WHERE collection_id = '{collection_id_escaped}'"
        )
        with contextlib.suppress(Exception):
            self._execute(
                f"DELETE FROM `{NamespaceCollectionNames.sdk_ltables_table()}` "
                f"WHERE collection_id = '{collection_id_escaped}'"
            )
        with contextlib.suppress(Exception):
            self._execute(
                f"DELETE FROM `{NamespaceCollectionNames.sdk_namespaces_table()}` "
                f"WHERE collection_id = '{collection_id_escaped}'"
            )
        with contextlib.suppress(Exception):
            self._execute(
                f"DELETE FROM `{NamespaceCollectionNames.sdk_namespaces_stats_table()}` "
                f"WHERE collection_id = '{collection_id_escaped}'"
            )
        self._cleanup_namespace_physical_tables(collection_id)

    def _create_namespace_physical_tables(
        self,
        collection_id: str,
        dimension: int,
        ivf_config=None,
        fulltext_config=None,
        is_shared_storage: bool = False,
        partition_count: int = _DEFAULT_PARTITION_COUNT,
        cleanup_on_error: bool = True,
    ) -> None:
        """Create the physical tables backing a namespace collection."""
        tg_name = NamespaceCollectionNames.tablegroup_name(collection_id)
        data_table = NamespaceCollectionNames.data_table_name(collection_id)
        kv_table = NamespaceCollectionNames.kv_data_table_name(collection_id)
        schema_table = NamespaceCollectionNames.logic_schema_table_name(collection_id)

        index_parts = ["SEARCH INDEX idx_json(data_content)"]
        if fulltext_config is not None:
            fulltext_clause = _get_fulltext_index_sql(fulltext_config)
            index_parts.insert(0, f"FULLTEXT INDEX idx_fts(document) {fulltext_clause}")
        if ivf_config is not None:
            vector_index_sql = _get_ivf_vector_index_sql(ivf_config)
            index_parts.append(f"VECTOR INDEX idx_vec(embedding) {vector_index_sql}")
        index_sql = ",\n                ".join(index_parts)
        partition_clause = f"PARTITION BY KEY(namespace_id) PARTITIONS {partition_count}"

        # All CREATEs use IF NOT EXISTS so this is idempotent: a fresh create builds
        # everything, while a resume (after a crash left some tables behind) skips the
        # existing ones and only fills in the gaps.
        try:
            self._execute(f"CREATE TABLEGROUP IF NOT EXISTS `{tg_name}` SHARDING='ADAPTIVE'")

            data_sql = f"""CREATE TABLE IF NOT EXISTS `{data_table}` (
                namespace_id BIGINT UNSIGNED NOT NULL,
                ltable_id BIGINT UNSIGNED NOT NULL,
                document LONGTEXT,
                embedding VECTOR({dimension}),
                data_content JSON NOT NULL,
                created_by VARCHAR(64) DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                {index_sql}
            ) TABLEGROUP=`{tg_name}` COMMENT='逻辑表主数据' DEFAULT CHARSET=utf8mb4 ORGANIZATION HEAP IS_LOGIC_TABLE = TRUE LOB_INROW_THRESHOLD={LOGIC_DATA_TABLE_LOB_INROW_THRESHOLD}
            {partition_clause}"""

            self._execute(data_sql)

            if is_shared_storage:
                hot_table = NamespaceCollectionNames.hot_table_name(collection_id)
                self._execute(f"""CREATE TABLE IF NOT EXISTS `{hot_table}` (
                    namespace_id BIGINT UNSIGNED NOT NULL,
                    last_access_time TIMESTAMP(6) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY(namespace_id)
                ) TABLEGROUP=`{tg_name}` COMMENT='热点/TTL附属表' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX
                {partition_clause}""")

            self._execute(f"""CREATE TABLE IF NOT EXISTS `{kv_table}` (
                namespace_id BIGINT UNSIGNED NOT NULL,
                kv_key VARBINARY(1024) NOT NULL,
                kv_value LONGBLOB NOT NULL,
                PRIMARY KEY(namespace_id, kv_key)
            ) TABLEGROUP=`{tg_name}` COMMENT='索引与映射KV表' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX LOB_INROW_THRESHOLD=786432
            {partition_clause}""")

            self._execute(f"""CREATE TABLE IF NOT EXISTS `{schema_table}` (
                namespace_id BIGINT UNSIGNED NOT NULL,
                ltable_id BIGINT UNSIGNED NOT NULL,
                schema_content JSON NOT NULL,
                created_by VARCHAR(64) DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY(namespace_id, ltable_id)
            ) TABLEGROUP=`{tg_name}` COMMENT='LTable schema定义' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX
            {partition_clause}""")

        except Exception:
            if cleanup_on_error:
                self._cleanup_namespace_physical_tables(collection_id)
            raise

    def _cleanup_namespace_physical_tables(self, collection_id: str) -> None:
        """Drop the physical tables backing a namespace collection."""
        for suffix_fn in [
            NamespaceCollectionNames.data_table_name,
            NamespaceCollectionNames.logic_schema_table_name,
            NamespaceCollectionNames.kv_data_table_name,
            NamespaceCollectionNames.hot_table_name,
        ]:
            with contextlib.suppress(Exception):
                self._execute(f"DROP TABLE IF EXISTS `{suffix_fn(collection_id)}`")
        with contextlib.suppress(Exception):
            self._execute(f"DROP TABLEGROUP IF EXISTS `{NamespaceCollectionNames.tablegroup_name(collection_id)}`")

    def _table_exists(self, table_name: str) -> bool:
        """Whether `table_name` exists in the current (catalog) database."""
        name_escaped = escape_string(table_name)
        try:
            rows = self._execute(
                "SELECT 1 FROM information_schema.TABLES "
                f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{name_escaped}'"
            )
            return bool(rows)
        except Exception:
            return False

    def _tablegroup_exists(self, tablegroup_name: str) -> bool:
        """Whether `tablegroup_name` exists in the current OceanBase tenant."""
        name_escaped = escape_string(tablegroup_name)
        try:
            rows = self._execute(f"SELECT 1 FROM oceanbase.DBA_OB_TABLEGROUPS WHERE TABLEGROUP_NAME = '{name_escaped}'")
            return bool(rows)
        except Exception:
            return False

    def _ns_missing_physical_resources(self, collection_id: str, is_shared_storage: bool) -> list[str]:
        """Return expected tablegroup/tables that are absent for this namespace collection."""
        resources: list[tuple[str, bool]] = [
            (NamespaceCollectionNames.tablegroup_name(collection_id), True),
            (NamespaceCollectionNames.data_table_name(collection_id), False),
            (NamespaceCollectionNames.kv_data_table_name(collection_id), False),
            (NamespaceCollectionNames.logic_schema_table_name(collection_id), False),
        ]
        if is_shared_storage:
            resources.append((NamespaceCollectionNames.hot_table_name(collection_id), False))
        missing: list[str] = []
        for resource_name, is_tablegroup in resources:
            exists = self._tablegroup_exists(resource_name) if is_tablegroup else self._table_exists(resource_name)
            if not exists:
                missing.append(resource_name)
        return missing

    def _is_incomplete_ns_collection(self, name: str) -> bool:
        """Whether `name` is a namespace collection whose catalog row exists but
        whose physical tables are not all present (e.g. creation was interrupted
        by a crash). Such a collection can be finished by re-running create.
        """
        meta = self._get_ns_collection_meta(name)
        if meta is None:
            return False
        is_ss = meta.get("settings", {}).get("storage_mode") == "ss"
        self._use_catalog_database()
        return len(self._ns_missing_physical_resources(meta["collection_id"], is_ss)) > 0

    def _purge_broken_ns_collection_if_incomplete(self, collection_name: str, meta: dict | None = None) -> bool:
        """Purge a namespace collection whose catalog row exists but physical resources are incomplete.

        Returns True when the collection was purged and should be treated as non-existent.
        """
        if meta is None:
            meta = self._get_ns_collection_meta(collection_name)
        if meta is None:
            return False
        self._use_catalog_database()
        is_ss = meta.get("settings", {}).get("storage_mode") == "ss"
        missing = self._ns_missing_physical_resources(meta["collection_id"], is_ss)
        if not missing:
            return False
        logger.warning(
            "Namespace collection '%s' (collection_id=%s) is missing physical resources %s; "
            "purging catalog metadata and cleaning up leftovers.",
            collection_name,
            meta["collection_id"],
            missing,
        )
        self._delete_ns_collection_meta(collection_name)
        return True

    def _resolve_namespace_ltable_id(self, collection_id: str, namespace_id: int) -> int:
        """Resolve the default ltable_id for (collection_id, namespace_id) from
        sdk_ltables. Cached per (collection_id, namespace_id) on the client
        instance to avoid the extra round-trip on every DML/DQL call.

        The cache is also seeded by `_create_ns_namespace_meta` and
        `_get_ns_namespace_meta` once they have read/created the row.
        """
        key = (str(collection_id), int(namespace_id))
        cache = getattr(self, "_ns_ltable_id_cache", None)
        if cache is None:
            cache = {}
            self._ns_ltable_id_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        coll_id_escaped = escape_string(str(collection_id))
        rows = self._execute(
            f"SELECT ltable_id FROM {self._qtable(NamespaceCollectionNames.sdk_ltables_table())} "
            f"WHERE collection_id = '{coll_id_escaped}' AND namespace_id = {int(namespace_id)} "
            f"AND ltable_name = 'default' LIMIT 1"
        )
        if not rows:
            raise ValueError(
                f"No default ltable found for collection_id={collection_id}, namespace_id={namespace_id} in sdk_ltables"
            )
        row = rows[0]
        lt_id = int(row[0] if isinstance(row, (list, tuple)) else row["ltable_id"])
        cache[key] = lt_id
        return lt_id

    def _cache_namespace_ltable_id(self, collection_id: str, namespace_id: int, ltable_id: int) -> None:
        """Cache the resolved logical-table id for a namespace."""
        key = (str(collection_id), int(namespace_id))
        cache = getattr(self, "_ns_ltable_id_cache", None)
        if cache is None:
            cache = {}
            self._ns_ltable_id_cache = cache
        cache[key] = int(ltable_id)

    def _set_session_ns_context(
        self,
        collection_id: str | None = None,
        namespace_id: int | None = None,
        ltable_id: int | None = None,
    ) -> None:
        """Set session variables identifying the active namespace context."""
        if collection_id is not None:
            self._execute(f"SET @collection_id = '{escape_string(str(collection_id))}'")
        if namespace_id is not None:
            self._execute(f"SET @namespace_id = {int(namespace_id)}")
        if ltable_id is not None:
            self._execute(f"SET @ltable_id = {int(ltable_id)}")

    def _fetch_ns_namespace_id(self, collection_id: str, namespace_name: str) -> int:
        """Fetch the namespace id for a collection/namespace pair from the catalog."""
        namespace_name_escaped = escape_string(namespace_name)
        collection_id_escaped = escape_string(collection_id)
        ns_table = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        rows = self._execute(
            f"SELECT namespace_id FROM {ns_table} "
            f"WHERE collection_id = '{collection_id_escaped}' AND namespace_name = '{namespace_name_escaped}'"
        )
        if not rows:
            raise ValueError(
                f"Namespace '{namespace_name}' not found for collection_id={collection_id} in sdk_namespaces"
            )
        return int(rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["namespace_id"])

    def _fetch_ns_ltable_id(
        self,
        collection_id: str,
        namespace_id: int,
        ltable_name: str = "default",
    ) -> int:
        """Fetch the logical-table id for a namespace from the catalog."""
        collection_id_escaped = escape_string(collection_id)
        ltable_name_escaped = escape_string(ltable_name)
        lt_table = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        lt_rows = self._execute(
            f"SELECT ltable_id FROM {lt_table} "
            f"WHERE collection_id = '{collection_id_escaped}' AND namespace_id = {int(namespace_id)} "
            f"AND ltable_name = '{ltable_name_escaped}'"
        )
        if not lt_rows:
            raise ValueError(
                f"LTable '{ltable_name}' not found for collection_id={collection_id}, "
                f"namespace_id={namespace_id} in sdk_ltables"
            )
        return int(lt_rows[0][0] if isinstance(lt_rows[0], (list, tuple)) else lt_rows[0]["ltable_id"])

    def _insert_ns_namespace_catalog_row(
        self,
        collection_id: str,
        namespace_name: str,
        *,
        idempotent: bool,
    ) -> int:
        """Insert a row into the sdk_namespaces catalog table."""
        namespace_name_escaped = escape_string(namespace_name)
        collection_id_escaped = escape_string(collection_id)
        ns_table = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        try:
            self._execute(
                f"INSERT INTO {ns_table} "
                f"(collection_id, namespace_name) VALUES ('{collection_id_escaped}', '{namespace_name_escaped}')"
            )
        except Exception as exc:
            if idempotent and _is_namespace_catalog_conflict_error(exc):
                self._rollback_connection_if_supported()
            else:
                raise
        return self._fetch_ns_namespace_id(collection_id, namespace_name)

    def _insert_ns_ltable_catalog_row(
        self,
        collection_id: str,
        namespace_id: int,
        *,
        idempotent: bool,
        ltable_name: str = "default",
    ) -> int:
        """Insert a row into the sdk_ltables catalog table."""
        collection_id_escaped = escape_string(collection_id)
        ltable_name_escaped = escape_string(ltable_name)
        lt_table = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        try:
            self._execute(
                f"INSERT INTO {lt_table} "
                f"(collection_id, namespace_id, ltable_name) "
                f"VALUES ('{collection_id_escaped}', {int(namespace_id)}, '{ltable_name_escaped}')"
            )
        except Exception as exc:
            if idempotent and _is_namespace_catalog_conflict_error(exc):
                self._rollback_connection_if_supported()
            else:
                raise
        return self._fetch_ns_ltable_id(collection_id, namespace_id, ltable_name)

    def _resolve_ns_ltable_index_layout(self, collection_id: str) -> tuple[bool, bool]:
        """Infer which optional indexes exist for a namespace collection."""
        collection_id_escaped = escape_string(collection_id)
        rows = self._execute(
            f"SELECT settings FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            f"WHERE collection_id = '{collection_id_escaped}'"
        )
        settings: dict[str, Any] = {}
        if rows:
            raw = rows[0]["settings"] if isinstance(rows[0], dict) else rows[0][0]
            settings = json.loads(raw) if raw else {}
        has_ivf = settings.get("dense_index_type") == "ivf"
        if "has_fulltext_index" in settings:
            has_fulltext = bool(settings["has_fulltext_index"])
        else:
            data_table = NamespaceCollectionNames.data_table_name(collection_id)
            if self._table_exists(data_table):
                index_rows = self._execute(f"SHOW INDEX FROM `{data_table}`")
                index_names = {(row.get("Key_name") if isinstance(row, dict) else row[2]) for row in (index_rows or [])}
                has_fulltext = "idx_fts" in index_names
            else:
                has_fulltext = False
        return has_fulltext, has_ivf

    def _finalize_ns_namespace_meta(
        self,
        collection_id: str,
        namespace_name: str,
        namespace_id: int,
        ltable_id: int,
    ) -> dict:
        """Finalize namespace metadata after catalog rows and physical tables are created."""
        self._cache_namespace_ltable_id(collection_id, namespace_id, ltable_id)
        schema_table = self._qtable(NamespaceCollectionNames.logic_schema_table_name(collection_id))
        has_fulltext, has_ivf = self._resolve_ns_ltable_index_layout(collection_id)
        schema_content = json.dumps(_build_default_ltable_schema(has_fulltext=has_fulltext, has_ivf=has_ivf))
        with contextlib.suppress(Exception):
            self._execute(
                f"INSERT INTO {schema_table} (namespace_id, ltable_id, schema_content) "
                f"VALUES ({namespace_id}, {ltable_id}, '{escape_string(schema_content)}')"
            )
        self._set_session_ns_context(namespace_id=namespace_id, ltable_id=ltable_id)
        return {"namespace_id": str(namespace_id), "namespace_name": namespace_name, "ltable_id": str(ltable_id)}

    def _create_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> dict:
        """Create namespace metadata, resolving unique-key conflicts idempotently."""
        if self._get_ns_namespace_meta(collection_id, namespace_name) is not None:
            raise ValueError(f"Namespace '{namespace_name}' already exists")
        try:
            ns_id = self._insert_ns_namespace_catalog_row(collection_id, namespace_name, idempotent=False)
            lt_id = self._insert_ns_ltable_catalog_row(collection_id, ns_id, idempotent=False)
        except Exception as exc:
            if _is_namespace_catalog_conflict_error(exc):
                self._rollback_connection_if_supported()
                raise ValueError(f"Namespace '{namespace_name}' already exists") from exc
            raise
        return self._finalize_ns_namespace_meta(collection_id, namespace_name, ns_id, lt_id)

    def _get_or_create_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> dict:
        """Get existing namespace metadata or create it if absent."""
        meta = self._get_ns_namespace_meta(collection_id, namespace_name)
        if meta is not None:
            if meta.get("ltable_id") is not None:
                return meta
            lt_id = self._insert_ns_ltable_catalog_row(collection_id, int(meta["namespace_id"]), idempotent=True)
            return self._finalize_ns_namespace_meta(collection_id, namespace_name, int(meta["namespace_id"]), lt_id)
        ns_id = self._insert_ns_namespace_catalog_row(collection_id, namespace_name, idempotent=True)
        lt_id = self._insert_ns_ltable_catalog_row(collection_id, ns_id, idempotent=True)
        return self._finalize_ns_namespace_meta(collection_id, namespace_name, ns_id, lt_id)

    def _get_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> dict | None:
        """Fetch namespace metadata from the catalog."""
        namespace_name_escaped = escape_string(namespace_name)
        collection_id_escaped = escape_string(collection_id)
        ns_table = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        lt_table = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        rows = self._execute(
            f"SELECT n.namespace_id AS namespace_id, n.namespace_name AS namespace_name, "
            f"l.ltable_id AS ltable_id "
            f"FROM {ns_table} n "
            f"LEFT JOIN {lt_table} l "
            f"ON l.collection_id = n.collection_id "
            f"AND l.namespace_id = n.namespace_id "
            f"AND l.ltable_name = 'default' "
            f"WHERE n.collection_id = '{collection_id_escaped}' "
            f"AND n.namespace_name = '{namespace_name_escaped}'"
        )
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, (list, tuple)):
            ns_id = str(row[0])
            ns_name = row[1]
            lt_raw = row[2] if len(row) > 2 else None
        else:
            ns_id = str(row["namespace_id"])
            ns_name = row["namespace_name"]
            lt_raw = row.get("ltable_id")
        lt_id = int(lt_raw) if lt_raw is not None else None
        if lt_id is not None:
            self._cache_namespace_ltable_id(collection_id, int(ns_id), lt_id)
        self._set_session_ns_context(namespace_id=int(ns_id), ltable_id=lt_id)
        meta: dict = {"namespace_id": ns_id, "namespace_name": ns_name}
        if lt_id is not None:
            meta["ltable_id"] = str(lt_id)
        return meta

    def _has_ns_namespace(self, collection_id: str, namespace_name: str) -> bool:
        """Return whether a namespace with the given name exists."""
        return self._get_ns_namespace_meta(collection_id, namespace_name) is not None

    def _ns_namespace_exists_by_id(self, collection_id: str, namespace_id: str) -> bool:
        """Whether a namespace with this id is still live in the catalog.

        Used to reject DML/DQL on a stale Namespace handle whose namespace (or
        whole collection) was deleted. delete_namespace soft-deletes by renaming
        the row to '__recyclebin_<name>_<id>' (kernel async cleanup follows), so
        a recyclebin-prefixed row counts as gone; a deleted collection removes
        the row outright. Underlying data may linger after either, so we trust
        the catalog, not the data table.
        """
        collection_id_escaped = escape_string(str(collection_id))
        rows = self._execute(
            f"SELECT namespace_id FROM {self._qtable(NamespaceCollectionNames.sdk_namespaces_table())} "
            f"WHERE collection_id = '{collection_id_escaped}' AND namespace_id = {int(namespace_id)} "
            f"AND LEFT(namespace_name, 13) <> '__recyclebin_'"
        )
        return bool(rows)

    def _delete_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> None:
        """Delete namespace metadata from the catalog."""
        meta = self._get_ns_namespace_meta(collection_id, namespace_name)
        if meta is None:
            raise ValueError(f"Namespace '{namespace_name}' not found")
        ns_id = meta["namespace_id"]
        lt_id_raw = meta.get("ltable_id")
        lt_id = int(lt_id_raw) if lt_id_raw is not None else None
        collection_id_escaped = escape_string(collection_id)
        # PL reads session database_name; must match where catalog tables live.
        self._use_catalog_database()
        self._set_session_ns_context(collection_id=collection_id, namespace_id=int(ns_id), ltable_id=lt_id)
        self._execute(f"CALL DBMS_LOGIC_TABLE.DROP_NAMESPACE('{collection_id_escaped}', {ns_id})")

    def _list_ns_namespaces(self, collection_id: str) -> list[dict]:
        """List namespaces registered for a collection."""
        collection_id_escaped = escape_string(collection_id)
        rows = self._execute(
            f"SELECT namespace_id, namespace_name FROM {self._qtable(NamespaceCollectionNames.sdk_namespaces_table())} "
            f"WHERE collection_id = '{collection_id_escaped}' "
            f"AND LEFT(namespace_name, 13) <> '__recyclebin_' "
            f"ORDER BY namespace_id"
        )
        results = []
        for row in rows:
            if isinstance(row, (list, tuple)):
                results.append({"namespace_id": str(row[0]), "namespace_name": row[1]})
            else:
                results.append({"namespace_id": str(row["namespace_id"]), "namespace_name": row["namespace_name"]})
        return results

    def _get_ns_namespace_id(self, collection_id: str, namespace_name: str) -> str:
        """Resolve the namespace id for a collection/namespace pair."""
        meta = self._get_ns_namespace_meta(collection_id, namespace_name)
        if meta is None:
            raise ValueError(f"Namespace '{namespace_name}' not found in collection {collection_id}")
        return meta["namespace_id"]

    # ==================== End Namespace Catalog Methods ====================

    def get_collection(self, name: str, embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED) -> "Collection":
        """Get an existing collection by name."""
        ns_meta = None
        with contextlib.suppress(Exception):
            ns_meta = self._get_ns_collection_meta(name)
        if ns_meta is not None:
            if self._purge_broken_ns_collection_if_incomplete(collection_name=name, meta=ns_meta):
                ns_meta = None
            else:
                return self._build_ns_collection_from_meta(ns_meta, embedding_function)
        try:
            collection = self._get_collection_v1(name, embedding_function)
        except ValueError as e:
            logger.debug(f"Failed to get collection v1: {e}, trying v2...")
            collection = self._get_collection_v2(name, embedding_function)
        return collection

    def _build_ns_collection_from_meta(self, meta: dict, embedding_function=_NOT_PROVIDED) -> "Collection":
        """Build a namespace Collection facade from catalog metadata."""
        settings = meta.get("settings", {})
        dimension = settings.get("dimension")
        distance = settings.get("distance", DEFAULT_DISTANCE_METRIC)
        partition_count = settings.get("partition_count")
        ef = None
        if embedding_function is not _NOT_PROVIDED:
            ef = embedding_function
        elif "embedding_function" in settings:
            ef_info = settings["embedding_function"]
            ef_class = EmbeddingFunctionRegistry.get_class(ef_info["name"])
            if ef_class is None:
                raise ValueError(f"Embedding function class '{ef_info['name']}' not found")
            ef = ef_class.build_from_config(ef_info.get("properties", {}))
        return Collection(
            client=self,
            name=meta["collection_name"],
            collection_id=meta["collection_id"],
            dimension=dimension,
            embedding_function=ef,
            distance=distance,
            use_namespace=True,
            partition_count=partition_count,
            has_vector_index=settings.get("dense_index_type") == "ivf",
        )

    def _resolve_collection_metadata_from_sdk_collections(self, collection_name: str) -> _CollectionMeta | None:
        """
        Resolve collection metadata infromation from sdk_collections table
        """
        try:
            query_sql = f"SELECT COLLECTION_ID, COLLECTION_NAME, SETTINGS FROM `{CollectionNames.sdk_collections_table_name()}` WHERE COLLECTION_NAME = '{collection_name}'"
            rows = self._execute(query_sql)
            if rows:
                return _CollectionMeta.from_row(rows[0])

            # not a v2 collection
            show_tables_sql = f"SHOW TABLES LIKE '{CollectionNames.table_name(collection_name)}'"
            result = self._execute(show_tables_sql)
            if result:
                return _CollectionMeta(collection_id=None, collection_name=collection_name, settings=None)
        except Exception as e:
            raise ValueError(f"Failed to resolve collection metadata from sdk_collections table: {e}") from e
        return None

    def _resolve_collection_metadata_from_table(self, table_name: str, collection_name: str) -> dict[str, Any]:
        """
        Resolve collection metadata infromation from collection table (not sdk_collections table)
        """
        metadata = {
            "dimension": None,
            "distance": None,
        }
        # Check if table exists by describing it
        try:
            table_info = self._execute(f"DESCRIBE `{table_name}`")
            if not table_info or len(table_info) == 0:
                raise ValueError(  # noqa: TRY301
                    f"Collection ('{collection_name}') not found: Table('{table_name}') not exists"
                )
        except Exception as e:
            # If DESCRIBE fails, check if it's because table doesn't exist
            error_msg = str(e).lower()
            if "doesn't exist" in error_msg or "not found" in error_msg or "table" in error_msg:
                raise ValueError(f"Collection ('{collection_name}') not found: Table('{table_name}') not exists") from e
            raise

        # Extract dimension from embedding column
        for row in table_info:
            # Handle both dict and tuple formats
            if isinstance(row, dict):
                field_name = row.get("Field", row.get("field", ""))
                field_type = row.get("Type", row.get("type", ""))
            elif isinstance(row, (tuple, list)):
                field_name = row[0] if len(row) > 0 else ""
                field_type = row[1] if len(row) > 1 else ""
            else:
                continue

            if field_name == "embedding" and "vector" in str(field_type).lower():
                # Extract dimension from vector(dimension) format
                match = re.search(r"vector\s*\(\s*(\d+)\s*\)", str(field_type), re.IGNORECASE)
                if match:
                    metadata["dimension"] = int(match.group(1))
                break

        # Extract distance from CREATE TABLE statement
        try:
            create_table_result = self._execute(f"SHOW CREATE TABLE `{table_name}`")
            if create_table_result and len(create_table_result) > 0:
                # Handle both dict and tuple formats
                if isinstance(create_table_result[0], dict):
                    create_stmt = create_table_result[0].get(
                        "Create Table", create_table_result[0].get("create table", "")
                    )
                elif isinstance(create_table_result[0], (tuple, list)):
                    # CREATE TABLE statement is usually in the second column
                    create_stmt = create_table_result[0][1] if len(create_table_result[0]) > 1 else ""
                else:
                    create_stmt = str(create_table_result[0])

                # Extract distance from VECTOR INDEX ... with(distance=..., ...)
                # Pattern: VECTOR INDEX ... with(distance=l2, ...) or with(distance='l2', ...)
                # Match: with(distance=value, ...) where value can be l2, cosine, inner_product, or ip
                distance_match = re.search(
                    r'with\s*\([^)]*distance\s*=\s*([\'"]?)(\w+)\1',
                    create_stmt,
                    re.IGNORECASE,
                )
                if distance_match:
                    distance = metadata["distance"] = distance_match.group(2).lower()
                    # Normalize distance values
                    if distance == "ip":
                        distance = "inner_product"
                    elif distance in ["l2", "cosine", "inner_product"]:
                        pass
                    else:
                        # Unknown distance, default to None
                        logger.warning(
                            f"Unknown distance value '{distance}' in CREATE TABLE statement, defaulting to None"
                        )
                        metadata["distance"] = None
        except Exception as e:
            # If SHOW CREATE TABLE fails, log warning but continue
            logger.warning(f"Failed to get CREATE TABLE statement for '{table_name}': {e}")

        return metadata

    def _resolve_embedding_function(self, settings: str | None) -> EmbeddingFunction[EmbeddingDocuments]:
        """Resolve the embedding function to use for a collection."""
        if not settings:
            return None
        settings_json = json.loads(settings)
        ef_settings = settings_json.get("embedding_function", {})
        ef_name = ef_settings.get("name", "")
        if not ef_name:
            return None
        embedding_function_class = EmbeddingFunctionRegistry.get_class(ef_name)
        if not embedding_function_class:
            raise ValueError(f"Embedding function class '{ef_name}' not found")
        return embedding_function_class.build_from_config(ef_settings.get("properties", {}))

    def _resolve_sparse_vector_index_config(self, settings: str | None) -> SparseVectorIndexConfig | None:
        """Restore SparseVectorIndexConfig from persisted settings JSON."""
        if not settings:
            return None
        settings_json = json.loads(settings)
        sparse_settings = settings_json.get("sparse_vector_index")
        if not sparse_settings:
            return None

        # Resolve sparse embedding function (required)
        sparse_ef = None
        ef_info = sparse_settings.get("embedding_function")
        if not ef_info:
            raise ValueError("Sparse vector index settings missing required embedding_function configuration")

        ef_name = ef_info.get("name", "")
        if not ef_name:
            raise ValueError("Sparse vector index settings missing embedding_function name")

        sparse_ef_class = SparseEmbeddingFunctionRegistry.get_class(ef_name)
        if sparse_ef_class is None:
            raise ValueError(f"Sparse embedding function class '{ef_name}' not found in registry")
        sparse_ef = sparse_ef_class.build_from_config(ef_info.get("properties", {}))

        # Resolve source_key
        source_key_str = sparse_settings.get("source_key")
        source_key = FieldKey.DOCUMENT
        if source_key_str:
            if source_key_str == "#document" or source_key_str == FieldKey.DOCUMENT.name:
                source_key = FieldKey.DOCUMENT
            else:
                source_key = source_key_str

        return SparseVectorIndexConfig(
            embedding_function=sparse_ef,
            source_key=source_key,
        )

    def _validate_embedding_function(
        self,
        embedding_function: EmbeddingFunction | None,
        embedding_function_persistence: EmbeddingFunction | None,
    ) -> EmbeddingFunction[EmbeddingDocuments]:
        """
        Validate embedding function

        Args:
            embedding_function: Embedding function user provided
            embedding_function_persistence: Embedding function restored from table metadata

        Returns:
            Embedding function
        """

        if embedding_function_persistence is not None and embedding_function is not _NOT_PROVIDED:
            if embedding_function is None or embedding_function_persistence.name() != embedding_function.name():
                raise ValueError(
                    "Both embedding function from parameter (not _NOT_PROVIDED, default value) and embedding function from persistence provided."
                )
            else:
                return embedding_function_persistence
        if embedding_function is _NOT_PROVIDED:
            return (
                embedding_function_persistence
                if embedding_function_persistence is not None
                else get_default_embedding_function()
            )
        else:
            return embedding_function

    def _get_collection_v2(self, name: str, embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED) -> "Collection":
        """Fetch a collection using the v2 catalog layout."""
        collection_meta = self._resolve_collection_metadata_from_sdk_collections(name)
        if not collection_meta or not collection_meta.collection_id:
            raise ValueError(f"Collection '{name}' does not exist")

        try:
            embedding_function_persistence = self._resolve_embedding_function(collection_meta.settings)
            embedding_function = self._validate_embedding_function(embedding_function, embedding_function_persistence)
            metadata = self._resolve_collection_metadata_from_table(
                CollectionNames.table_name_v2(collection_meta.collection_id), name
            )

            # Resolve sparse vector index config from persisted settings
            sparse_vector_index_config = self._resolve_sparse_vector_index_config(collection_meta.settings)

            return Collection(
                client=self,
                name=name,
                collection_id=collection_meta.collection_id,
                embedding_function=embedding_function,
                dimension=metadata["dimension"],
                distance=metadata["distance"],
                sparse_vector_index_config=sparse_vector_index_config,
            )
        except Exception as e:
            raise ValueError(f"Failed to get collection: {e}") from e

    def _get_collection_v1(self, name: str, embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED) -> "Collection":
        """
        Get a collection object (user-facing API)

        Args:
            name: Collection name
            embedding_function: Embedding function to convert documents to embeddings.
                               Defaults to DefaultEmbeddingFunction.
                               If explicitly set to None, collection will not have an embedding function.

        Returns:
            Collection object

        Raises:
            ValueError: If collection does not exist
        """
        # Construct table name
        table_name = CollectionNames.table_name(name)

        metadata = self._resolve_collection_metadata_from_table(table_name, name)

        # Handle embedding function
        # If not provided (sentinel), use default embedding function
        if embedding_function is _NOT_PROVIDED:
            embedding_function = get_default_embedding_function()

        # Create and return Collection object
        return Collection(client=self, name=name, embedding_function=embedding_function, **metadata)

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: The name of the collection to delete.

        Raises:
            ValueError: If the collection does not exist.

        Examples:
            >>> client.delete_collection("my_collection")
        """
        if self._has_ns_collection(name):
            self._delete_ns_collection_meta(name)
            logger.debug(f"Deleted namespace collection '{name}'")
            return
        try:
            self._delete_collection_v2(name)
            logger.debug(f"✅ Successfully deleted collection v2 '{name}' from sdk_collections table")
        except ValueError:
            self._delete_collection_v1(name)
            logger.debug(f"✅ Successfully deleted collection v1 '{name}' from table")

    def _delete_collection_v2(self, name: str) -> None:
        """
        Delete a collection (user-facing API)

        Args:
            name: Collection name
        """
        collection = self._get_collection_v2(name)
        if not collection:
            raise ValueError(f"Collection '{name}' does not exist")
        drop_table_sql = f"DROP TABLE `{CollectionNames.table_name_v2(collection.id)}`"
        query_sql = f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` WHERE COLLECTION_NAME = '{name}'"
        self._execute(drop_table_sql)
        self._execute(query_sql)
        logger.debug(f"✅ Successfully deleted collection '{name}' from sdk_collections table")

    def _delete_collection_v1(self, name: str) -> None:
        """
        Delete a collection (user-facing API)

        Args:
            name: Collection name

        Raises:
            ValueError: If collection does not exist
        """
        # Construct table name
        table_name = CollectionNames.table_name(name)

        # Check if table exists first
        if not self._has_collection_v1(name):
            raise ValueError(f"Collection '{name}' does not exist")

        # Execute DROP TABLE SQL
        self._execute(f"DROP TABLE IF EXISTS `{table_name}`")

    def list_collections(self) -> list["Collection"]:
        """List all collections in the database.

        Returns:
            A list of ``Collection`` objects.

        Examples:
            >>> collections = client.list_collections()
            >>> for col in collections:
            ...     print(col.name)
        """
        collections = self._list_ns_collections()
        collections.extend(self._list_collections_v1())
        collections.extend(self._list_collections_v2())
        return collections

    def _list_ns_collections(self) -> list["Collection"]:
        """List namespace-enabled collections."""
        result = []
        try:
            sdk_table = CollectionNames.sdk_collections_table_name()
            check_sql = f"SHOW TABLES LIKE '{sdk_table}'"
            check_result = self._execute(check_sql)
            if not check_result:
                return result
            rows = self._execute(f"SELECT collection_id, collection_name, settings FROM `{sdk_table}`")
            for row in rows:
                try:
                    if isinstance(row, dict):
                        settings = json.loads(row["settings"]) if row.get("settings") else {}
                    else:
                        settings = json.loads(row[2]) if row[2] else {}
                    if not settings.get("use_namespace"):
                        continue
                    if isinstance(row, dict):
                        meta = {
                            "collection_id": str(row["collection_id"]),
                            "collection_name": row["collection_name"],
                            "settings": settings,
                        }
                    else:
                        meta = {
                            "collection_id": str(row[0]),
                            "collection_name": row[1],
                            "settings": settings,
                        }
                    result.append(self._build_ns_collection_from_meta(meta))
                except Exception as e:
                    logger.warning(f"Failed to build namespace collection from row: {e}")
        except Exception:
            logger.debug("Failed to list namespace collections from catalog", exc_info=True)
        return result

    def _list_collections_v2(self) -> list["Collection"]:
        """List collections using the v2 catalog layout."""
        collections = []
        try:
            # Detect if the sdk_collections table exists before querying it
            sdk_collections_table = CollectionNames.sdk_collections_table_name()
            has_sdk_collections = False
            try:
                check_table_sql = f"SHOW TABLES LIKE '{sdk_collections_table}'"
                check_result = self._execute(check_table_sql)
                if check_result:
                    # Table exists (SHOW TABLES LIKE returns at least one row if exists)
                    has_sdk_collections = True
            except Exception:
                has_sdk_collections = False

            if has_sdk_collections:
                query_sql = f"SELECT COLLECTION_NAME, SETTINGS FROM {sdk_collections_table}"
                rows = self._execute(query_sql)
                for row in rows:
                    try:
                        if isinstance(row, dict):
                            collection_name = row.get("COLLECTION_NAME") or row.get("collection_name", "")
                            settings_raw = row.get("SETTINGS") or row.get("settings")
                        elif isinstance(row, (tuple, list)):
                            collection_name = row[0] if len(row) > 0 else ""
                            settings_raw = row[1] if len(row) > 1 else None
                        else:
                            collection_name = str(row)
                            settings_raw = None
                        if settings_raw:
                            try:
                                settings = json.loads(settings_raw) if isinstance(settings_raw, str) else settings_raw
                                if settings.get("use_namespace"):
                                    continue
                            except (json.JSONDecodeError, TypeError):
                                pass
                        collection = self.get_collection(collection_name)
                        collections.append(collection)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get collection. The data may be corrupted. The collection name: '{collection_name}': {e}"
                        )
                        continue

        except Exception as e:
            raise ValueError(f"Failed to list collections: {e}") from e
        return collections

    def _list_collections_v1(self) -> list["Collection"]:
        """
        List all collections from table names that start with collection prefix

        Returns:
            List of Collection objects
        """
        # List all tables that start with collection prefix
        # Use SHOW TABLES LIKE pattern to filter collection tables
        pattern = CollectionNames.table_pattern()
        try:
            tables = self._execute(f"SHOW TABLES LIKE '{pattern}'")
        except Exception:
            # Fallback: try to query information_schema
            try:
                # Get current database name
                db_result = self._execute("SELECT DATABASE()")
                if db_result and len(db_result) > 0:
                    db_name = (
                        db_result[0][0]
                        if isinstance(db_result[0], (tuple, list))
                        else db_result[0].get("DATABASE()", "")
                    )
                    tables = self._execute(
                        f"SELECT TABLE_NAME FROM information_schema.TABLES "
                        f"WHERE TABLE_SCHEMA = '{db_name}' AND TABLE_NAME LIKE '{pattern}'"
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
                table_name = next(iter(row.values()), "")
            elif isinstance(row, (tuple, list)):
                # Embedded client returns tuple, first element is table name
                table_name = row[0] if len(row) > 0 else ""
            else:
                table_name = str(row)

            # Extract collection name from table name
            if CollectionNames.is_collection_table(table_name):
                collection_name = CollectionNames.collection_name(table_name)

                # Get collection with dimension
                try:
                    collection = self.get_collection(collection_name)
                    collections.append(collection)
                except Exception as e:
                    logger.debug(f"Failed to get collection '{collection_name}': {e}")
                    continue

        return collections

    def count_collection(self) -> int:
        """Count the total number of collections.

        Returns:
            The number of collections.

        Examples:
            >>> count = client.count_collection()
            >>> print(f"Database has {count} collections")
        """
        collections = self.list_collections()
        return len(collections)

    def has_collection(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: The name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.

        Examples:
            >>> if client.has_collection("my_collection"):
            ...     print("Collection exists!")
        """
        return self._has_ns_collection(name) or self._has_collection_v2(name) or self._has_collection_v1(name)

    def _collection_table_exists(self, table_name: str) -> bool:
        """Return whether the physical table for a collection exists."""
        try:
            table_info = self._execute(f"DESCRIBE `{table_name}`")
            return table_info is not None and len(table_info) > 0
        except Exception:
            return False

    def _has_collection_v2(self, name: str) -> bool:
        """Return whether a collection exists using the v2 catalog layout."""
        try:
            name_escaped = escape_string(name)
            query_sql = (
                f"SELECT collection_id FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
                f"WHERE collection_name = '{name_escaped}' "
                f"ORDER BY created_at, collection_id LIMIT 1"
            )
            rows = self._execute(query_sql)
            if not rows or len(rows) == 0:
                return False

            collection_id = _extract_collection_id_from_sdk_row(rows[0])
            if not collection_id:
                return False

            return self._collection_table_exists(CollectionNames.table_name_v2(collection_id))
        except Exception:
            return False

    def _has_collection_v1(self, name: str) -> bool:
        """
        Check if a collection exists

        Args:
            name: Collection name

        Returns:
            True if exists, False otherwise
        """
        # Construct table name
        table_name = CollectionNames.table_name(name)

        # Check if table exists
        try:
            # Try to describe the table
            table_info = self._execute(f"DESCRIBE `{table_name}`")
            return table_info is not None and len(table_info) > 0
        except Exception:
            # If DESCRIBE fails, table doesn't exist
            return False

    def get_or_create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        use_namespace: bool = False,
        **kwargs,
    ) -> "Collection":
        """Get a collection if it exists, otherwise create it.

        Args:
            name: The name of the collection.
            schema: Schema configuration for fine-grained index control, including
                   sparse vector index support. When provided, ``configuration`` and
                   ``embedding_function`` parameters are ignored.
            configuration: Index configuration. Defaults to None (uses HNSW with
                Cosine distance and dimension 384). Can be a ``Configuration`` or
                ``HNSWConfiguration`` object. If set to None, the dimension will be
                inferred from the embedding function. Ignored if ``schema`` is provided.
            embedding_function: The embedding function to use for this collection.
                Defaults to ``DefaultEmbeddingFunction`` (all-MiniLM-L6-v2). If set to None,
                no embedding function will be used (embeddings must be provided manually).
                Ignored if ``schema`` is provided.
            use_namespace: If True, create a namespace-enabled collection. Defaults to False.
            **kwargs: Additional parameters passed to ``create_collection`` if the collection is created.

        Returns:
            The existing or newly created ``Collection`` object.

        Raises:
            ValueError: If the configuration/embedding function combination is invalid (e.g., dimension mismatch).

        Examples:
            >>> collection = client.get_or_create_collection("my_collection")
        """
        _validate_collection_name(name)

        if self.has_collection(name):
            if use_namespace and self._is_incomplete_ns_collection(name):
                return self.create_collection(
                    name=name,
                    schema=schema,
                    configuration=configuration,
                    embedding_function=embedding_function,
                    use_namespace=use_namespace,
                    **kwargs,
                )
            self._assert_get_or_create_namespace_mode_matches(name, use_namespace)
            return self.get_collection(name, embedding_function=embedding_function)

        try:
            return self.create_collection(
                name=name,
                schema=schema,
                configuration=configuration,
                embedding_function=embedding_function,
                use_namespace=use_namespace,
                **kwargs,
            )
        except Exception as exc:
            if _is_collection_conflict_error(exc):
                return self._get_or_resume_existing_collection(
                    name,
                    schema=schema,
                    configuration=configuration,
                    embedding_function=embedding_function,
                    use_namespace=use_namespace,
                    **kwargs,
                )
            raise

    def _get_or_resume_existing_collection(
        self,
        name: str,
        *,
        schema: Schema | None,
        configuration: ConfigurationParam,
        embedding_function: EmbeddingFunctionParam,
        use_namespace: bool,
        **kwargs,
    ) -> "Collection":
        """Return an existing collection or resume an incomplete namespace-enabled one."""
        if use_namespace:
            if self._get_ns_collection_meta(name) is None:
                raise ValueError(f"Collection '{name}' conflicted during create but namespace metadata is missing")
            if self._is_incomplete_ns_collection(name):
                return self.create_collection(
                    name=name,
                    schema=schema,
                    configuration=configuration,
                    embedding_function=embedding_function,
                    use_namespace=use_namespace,
                    **kwargs,
                )
        self._assert_get_or_create_namespace_mode_matches(name, use_namespace)
        return self.get_collection(name, embedding_function=embedding_function)

    def _assert_get_or_create_namespace_mode_matches(self, name: str, use_namespace: bool) -> None:
        """Reject get_or_create when an existing collection's namespace mode differs."""
        existing_use_namespace = self._get_ns_collection_meta(name) is not None
        if existing_use_namespace == use_namespace:
            return
        kind = "namespace-enabled" if existing_use_namespace else "standard"
        raise ValueError(
            f"Collection '{name}' already exists as a {kind} collection "
            f"(use_namespace={existing_use_namespace}), but get_or_create_collection was called with "
            f"use_namespace={use_namespace}. Delete the collection or use a different name."
        )

    def _get_collection_table_name(self, collection_id: str | None, collection_name: str) -> str:
        """
        Get collection table name
        """
        if collection_id:
            return CollectionNames.table_name_v2(collection_id)
        return CollectionNames.table_name(collection_name)

    def _fork_table_enabled(self) -> bool:
        """Return whether table fork is enabled on the backend."""
        db_type, version = self.detect_db_type_and_version()
        version_110 = Version("1.1.0.0")
        logger.debug(f"db_type: {db_type}, version: {version}")
        return db_type.lower() == "seekdb" and version >= version_110

    def _fork_database_enabled(self) -> bool:
        """Return whether database fork is enabled on the backend."""
        db_type, version = self.detect_db_type_and_version()
        version_120 = Version("1.2.0.0")
        logger.debug(f"db_type: {db_type}, version: {version}")
        return db_type.lower() == "seekdb" and version >= version_120

    def _refresh_enabled(self) -> bool:
        """Return whether index refresh is enabled on the backend."""
        db_type, version = self.detect_db_type_and_version()
        version_130 = Version("1.3.0.0")
        logger.debug(f"db_type: {db_type}, version: {version}")
        return db_type.lower() == "seekdb" and version >= version_130

    def refresh_index(self) -> None:
        """
        Flush async vector index build tasks when supported.

        For unsupported database versions, this method is a no-op to keep
        collection-level API calls backward compatible.
        """
        if not self._refresh_enabled():
            return

        self._execute("CALL dbms_index_manager.refresh();")

    def _get_collection_id(self, collection_name: str) -> str:
        """Resolve the collection id for a collection name."""
        collection_name_escaped = escape_string(collection_name)
        collection_id_query_sql = (
            f"SELECT collection_id FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            f"WHERE collection_name = '{collection_name_escaped}' "
            f"ORDER BY created_at, collection_id LIMIT 1"
        )
        collection_id_query_result = self._execute(collection_id_query_sql)
        if not collection_id_query_result or len(collection_id_query_result) == 0:
            raise ValueError(f"Collection not found: '{collection_name}'")
        collection_id = _extract_collection_id_from_sdk_row(collection_id_query_result[0])
        if not collection_id:
            raise ValueError(f"Collection not found: '{collection_name}'")
        return collection_id

    def _collection_fork(self, collection: Collection, forked_name: str) -> None:
        """
        Fork a collection

        Args:
            collection: Collection to fork
            forked_name: Forked collection name
        """
        if not self._fork_table_enabled():
            raise ValueError("Fork is not enabled for this database")

        _validate_collection_name(forked_name)

        if self.has_collection(forked_name):
            raise ValueError(f"Collection '{forked_name}' already exists")

        # Ensure sdk_collections exists (especially for v1-only databases)
        self._create_sdk_collections_if_not_exists()

        source_table_name = self._get_collection_table_name(collection.id, collection.name)
        collection_meta = self._resolve_collection_metadata_from_sdk_collections(collection.name)
        if not collection_meta:
            raise ValueError(f"Collection '{collection.name}' does not exist")
        forked_table_name = None
        try:
            settings_str = (
                f"'{escape_string(collection_meta.settings)}'" if collection_meta.settings is not None else "NULL"
            )
            insert_sql = f"INSERT INTO `{CollectionNames.sdk_collections_table_name()}` (COLLECTION_NAME, SETTINGS) VALUES ('{forked_name}', {settings_str})"
            self._execute(insert_sql)
            collection_id = self._get_collection_id(forked_name)
            forked_table_name = CollectionNames.table_name_v2(collection_id)

            fork_table_sql = f"FORK TABLE `{source_table_name}` TO `{forked_table_name}`"
            self._execute(fork_table_sql)
        except Exception as ex:
            try:
                if forked_table_name:
                    drop_table_sql = f"DROP TABLE IF EXISTS `{forked_table_name}`"
                    self._execute(drop_table_sql)
                delete_item_sql = f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` WHERE COLLECTION_NAME = '{forked_name}'"
                self._execute(delete_item_sql)
            except Exception as ex2:
                logger.warning(f"failed to clean data after failed to fork collection: {ex2}")

            raise ValueError(f"Failed to fork collection: {ex}") from ex
        logger.debug(f"✅ Successfully forked collection '{collection.name}' to '{forked_name}'")

    # ==================== Collection Internal Operations (Called by Collection) ====================
    # These methods are called by Collection objects, different clients implement different logic

    # -------------------- DML Operations --------------------

    def _generate_sparse_embeddings(
        self,
        sparse_config: SparseVectorIndexConfig,
        documents: list[str] | None,
        metadatas: list[dict] | None,
        num_items: int,
    ) -> list[SparseVector | None]:
        """
        Generate sparse embeddings based on SparseVectorIndexConfig.

        Returns a list of SparseVector (or None) for each item.
        """
        sparse_ef = sparse_config.embedding_function
        if sparse_ef is None:
            raise ValueError("Sparse embedding function is not provided")

        source_type, metadata_key = sparse_config.resolve_source_key()

        # Gather source texts
        source_texts = []
        for i in range(num_items):
            if source_type == "document":
                text = documents[i] if documents and i < len(documents) else None
                if text is None:
                    raise ValueError(
                        f"Sparse vector index is configured to generate from document field, "
                        f"but document at index {i} is None."
                    )
                if not isinstance(text, str):
                    raise TypeError(
                        f"Sparse vector index source_key refers to document field, "
                        f"but value at index {i} is not a string: {type(text).__name__}"
                    )
                source_texts.append(text)
            elif source_type == "metadata":
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                if meta is None:
                    raise ValueError(
                        f"Sparse vector index is configured to generate from metadata['{metadata_key}'], "
                        f"but metadata at index {i} is None."
                    )
                text = meta.get(metadata_key)
                if text is None:
                    raise ValueError(
                        f"Sparse vector index is configured to generate from metadata['{metadata_key}'], "
                        f"but metadata['{metadata_key}'] at index {i} is None."
                    )
                if not isinstance(text, str):
                    raise TypeError(
                        f"Sparse vector index source_key refers to metadata['{metadata_key}'], "
                        f"but value at index {i} is not a string: {type(text).__name__}"
                    )
                source_texts.append(text)
            else:
                raise ValueError(f"Invalid source type: {source_type}")

        # Generate sparse embeddings
        logger.debug(f"Generating sparse embeddings for {len(source_texts)} items")
        try:
            sparse_vectors = sparse_ef(source_texts)
        except Exception as e:
            raise ValueError(f"Failed to generate sparse embeddings: {e}") from e
        else:
            if len(sparse_vectors) != num_items:
                raise ValueError(
                    f"Sparse embedding function returned {len(sparse_vectors)} vectors, expected {num_items}."
                )
            logger.debug(f"✅ Successfully generated {len(sparse_vectors)} sparse embeddings")
            return sparse_vectors

    def _collection_add(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Add data to collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            embeddings: Single embedding or list of embeddings (optional)
            metadatas: Single metadata dict or list of metadata dicts (optional)
            documents: Single document or list of documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to embeddings.
                               Required if documents provided but embeddings not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.debug(f"Adding data to collection '{collection_name}'")

        explicit_embeddings = embeddings is not None
        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        self._warn_explicit_embeddings_override_embedding_function(
            operation="collection.add",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        # Handle vector generation logic:
        # 1. If embeddings are provided, use them directly without embedding
        # 2. If embeddings are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate embeddings from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither embeddings nor documents are provided, raise an error
        # NOTE: The embedding_function is passed through `get_collection` and `create_collection` parameters.
        # If embedding_function parameter passed in `get_collection` and `create_collection` is None,
        # then the embedding function is not provided. If developers passed through `_NOT_PROVIDED` (default value),
        # then the embedding function is the default embedding function.

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    embeddings = embedding_function(documents)
                except Exception as e:
                    logger.exception("Failed to generate embeddings")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}") from e
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling add(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        else:
            # Neither embeddings nor documents provided, raise an error
            raise ValueError(
                "Neither embeddings nor documents provided. "
                "Please provide either:\n"
                "  1. embeddings directly, or\n"
                "  2. documents with embedding_function to generate embeddings."
            )

        # Determine number of items
        num_items = 0
        if ids:
            num_items = len(ids)
        elif documents:
            num_items = len(documents)
        elif embeddings:
            num_items = len(embeddings)
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
        if embeddings and len(embeddings) != num_items:
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of items ({num_items})")

        # Get table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
            table_name = CollectionNames.table_name(collection_name)

        # Handle sparse embeddings generation
        sparse_config = kwargs.get("sparse_vector_index_config")
        sparse_embeddings = None
        has_sparse = sparse_config is not None
        if has_sparse:
            sparse_embeddings = self._generate_sparse_embeddings(sparse_config, documents, metadatas, num_items)

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
                # Use pymysql's escape_string for safe escaping
                doc_val_escaped = escape_string(doc_val)
                doc_sql = f"'{doc_val_escaped}'"
            else:
                doc_sql = "NULL"

            # Process metadata
            meta_val = metadatas[i] if metadatas else None
            if meta_val is not None:
                # Convert to JSON string and escape using pymysql's escape_string
                meta_json = json.dumps(meta_val, ensure_ascii=False)
                meta_json_escaped = escape_string(meta_json)
                meta_sql = f"'{meta_json_escaped}'"
            else:
                meta_sql = "NULL"

            # Process vector
            vec_val = embeddings[i] if embeddings else None
            vec_sql = "NULL" if vec_val is None else _embedding_to_hexstring(vec_val)

            # Process sparse vector
            if has_sparse:
                sparse_val = sparse_embeddings[i] if sparse_embeddings else None
                sparse_sql = "NULL" if sparse_val is None else _sparse_vector_to_sql(sparse_val)
                values_list.append(f"({id_sql}, {doc_sql}, {meta_sql}, {vec_sql}, {sparse_sql})")
            else:
                values_list.append(f"({id_sql}, {doc_sql}, {meta_sql}, {vec_sql})")

        # Build column list
        columns = f"{CollectionFieldNames.ID}, {CollectionFieldNames.DOCUMENT}, {CollectionFieldNames.METADATA}, {CollectionFieldNames.EMBEDDING}"
        if has_sparse:
            columns += f", {CollectionFieldNames.SPARSE_EMBEDDING}"

        # Build final SQL
        sql = f"""INSERT INTO `{table_name}` ({columns})
                 VALUES {",".join(values_list)}"""

        logger.debug(f"Executing SQL: {sql}")
        self._execute(sql)
        logger.debug(f"✅ Successfully added {num_items} item(s) to collection '{collection_name}'")

    def _collection_update(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Update data in collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to update
            embeddings: New embeddings (optional)
            metadatas: New metadata (optional)
            documents: New documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to embeddings.
                               Required if documents provided but embeddings not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.debug(f"Updating data in collection '{collection_name}'")

        explicit_embeddings = embeddings is not None
        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        self._warn_explicit_embeddings_override_embedding_function(
            operation="collection.update",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        # Handle vector generation logic:
        # 1. If embeddings are provided, use them directly without embedding
        # 2. If embeddings are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate embeddings from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither embeddings nor documents are provided:
        #    - If metadatas are provided, allow update (metadata-only update)
        #    - If metadatas are not provided, raise an error

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    embeddings = embedding_function(documents)
                except Exception as e:
                    logger.exception("Failed to generate embeddings")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}") from e
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling update(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif not metadatas:
            # Neither embeddings nor documents nor metadatas provided, raise an error
            raise ValueError(
                "Neither embeddings nor documents nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. embeddings directly, or\n"
                "  2. documents with embedding_function to generate embeddings, or\n"
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
        if embeddings and len(embeddings) != len(ids):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of ids ({len(ids)})")

        # Get table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
            table_name = CollectionNames.table_name(collection_name)

        # Handle sparse embeddings generation
        sparse_config = kwargs.get("sparse_vector_index_config")
        sparse_embeddings = None
        if sparse_config is not None:
            source_type, _ = sparse_config.resolve_source_key()
            should_generate = (source_type == "document" and documents) or (source_type == "metadata" and metadatas)
            if should_generate:
                sparse_embeddings = self._generate_sparse_embeddings(sparse_config, documents, metadatas, len(ids))

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
                    doc_val_escaped = escape_string(doc_val)
                    set_clauses.append(f"{CollectionFieldNames.DOCUMENT} = '{doc_val_escaped}'")

            if metadatas:
                meta_val = metadatas[i]
                if meta_val is not None:
                    meta_json = json.dumps(meta_val, ensure_ascii=False)
                    meta_json_escaped = escape_string(meta_json)
                    set_clauses.append(f"{CollectionFieldNames.METADATA} = '{meta_json_escaped}'")

            if embeddings:
                vec_val = embeddings[i]
                if vec_val is not None:
                    vec_str = "[" + ",".join(map(str, vec_val)) + "]"
                    set_clauses.append(f"{CollectionFieldNames.EMBEDDING} = '{vec_str}'")

            # Handle sparse embedding update
            if sparse_embeddings and sparse_embeddings[i] is not None:
                sparse_sql = _sparse_vector_to_sql(sparse_embeddings[i])
                set_clauses.append(f"{CollectionFieldNames.SPARSE_EMBEDDING} = {sparse_sql}")

            if not set_clauses:
                continue

            # Build UPDATE SQL
            sql = f"UPDATE `{table_name}` SET {', '.join(set_clauses)} WHERE {CollectionFieldNames.ID} = {id_sql}"

            logger.debug(f"Executing SQL: {sql}")
            self._execute(sql)

        logger.debug(f"✅ Successfully updated {len(ids)} item(s) in collection '{collection_name}'")

    def _collection_upsert(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Insert or update data in collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            embeddings: embeddings (optional)
            metadatas: Metadata (optional)
            documents: Documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to embeddings.
                               Required if documents provided but embeddings not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.debug(f"Upserting data in collection '{collection_name}'")

        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        # Handle vector generation logic:
        # 1. If embeddings are provided, use them directly without embedding
        # 2. If embeddings are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate embeddings from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither embeddings nor documents are provided:
        #    - If metadatas are provided, allow upsert (metadata-only upsert)
        #    - If metadatas are not provided, raise an error

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    embeddings = embedding_function(documents)
                except Exception as e:
                    logger.exception("Failed to generate embeddings")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}") from e
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling upsert(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif not metadatas:
            # Neither embeddings nor documents nor metadatas provided, raise an error
            raise ValueError(
                "Neither embeddings nor documents nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. embeddings directly, or\n"
                "  2. documents with embedding_function to generate embeddings, or\n"
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
        if embeddings and len(embeddings) != len(ids):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of ids ({len(ids)})")

        # Get table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
            table_name = CollectionNames.table_name(collection_name)

        # Handle sparse embeddings generation
        sparse_config = kwargs.get("sparse_vector_index_config")
        sparse_embeddings = None
        if sparse_config is not None:
            source_type, _ = sparse_config.resolve_source_key()
            should_generate = (source_type == "document" and documents) or (source_type == "metadata" and metadatas)
            if should_generate:
                sparse_embeddings = self._generate_sparse_embeddings(sparse_config, documents, metadatas, len(ids))

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
                include=["documents", "metadatas", "embeddings"],
            )

            # Get values for this item
            doc_val = documents[i] if documents else None
            meta_val = metadatas[i] if metadatas else None
            vec_val = embeddings[i] if embeddings else None

            if existing and len(existing.get("ids", [])) > 0:
                # Update existing record - only update provided fields
                existing_doc = existing.get("documents", [None])[0] if existing.get("documents") else None
                existing_meta = existing.get("metadatas", [None])[0] if existing.get("metadatas") else None
                existing_vec = existing.get("embeddings", [None])[0] if existing.get("embeddings") else None

                # Use provided values or keep existing values
                final_document = doc_val if doc_val is not None else existing_doc
                final_metadata = meta_val if meta_val is not None else existing_meta
                final_vector = vec_val if vec_val is not None else existing_vec

                # Build SET clause
                set_clauses = []

                if doc_val is not None:
                    if final_document is not None:
                        doc_val_escaped = escape_string(final_document)
                        set_clauses.append(f"{CollectionFieldNames.DOCUMENT} = '{doc_val_escaped}'")
                    else:
                        set_clauses.append(f"{CollectionFieldNames.DOCUMENT} = NULL")

                if meta_val is not None:
                    meta_json = json.dumps(final_metadata, ensure_ascii=False) if final_metadata else "{}"
                    meta_json_escaped = escape_string(meta_json)
                    set_clauses.append(f"{CollectionFieldNames.METADATA} = '{meta_json_escaped}'")

                if vec_val is not None:
                    vec_str = _embedding_to_hexstring(final_vector) if final_vector else "NULL"
                    set_clauses.append(f"{CollectionFieldNames.EMBEDDING} = {vec_str}")

                # Handle sparse embedding update
                if sparse_embeddings and sparse_embeddings[i] is not None:
                    sparse_sql = _sparse_vector_to_sql(sparse_embeddings[i])
                    set_clauses.append(f"{CollectionFieldNames.SPARSE_EMBEDDING} = {sparse_sql}")

                if set_clauses:
                    sql = (
                        f"UPDATE `{table_name}` SET {', '.join(set_clauses)} WHERE {CollectionFieldNames.ID} = {id_sql}"
                    )
                    logger.debug(f"Executing SQL: {sql}")
                    self._execute(sql)
            else:
                # Insert new record
                if doc_val is not None:
                    doc_val_escaped = escape_string(doc_val)
                    doc_sql = f"'{doc_val_escaped}'"
                else:
                    doc_sql = "NULL"

                if meta_val is not None:
                    meta_json = json.dumps(meta_val, ensure_ascii=False)
                    meta_json_escaped = escape_string(meta_json)
                    meta_sql = f"'{meta_json_escaped}'"
                else:
                    meta_sql = "NULL"

                vec_sql = "NULL" if vec_val is None else _embedding_to_hexstring(vec_val)

                # Build column list and values for insert
                columns = f"{CollectionFieldNames.ID}, {CollectionFieldNames.DOCUMENT}, {CollectionFieldNames.METADATA}, {CollectionFieldNames.EMBEDDING}"
                values = f"{id_sql}, {doc_sql}, {meta_sql}, {vec_sql}"

                if sparse_embeddings and sparse_embeddings[i] is not None:
                    sparse_sql = _sparse_vector_to_sql(sparse_embeddings[i])
                    columns += f", {CollectionFieldNames.SPARSE_EMBEDDING}"
                    values += f", {sparse_sql}"

                sql = f"""INSERT INTO `{table_name}` ({columns})
                         VALUES ({values})"""
                logger.debug(f"Executing SQL: {sql}")
                self._execute(sql)

        logger.debug(f"✅ Successfully upserted {len(ids)} item(s) in collection '{collection_name}'")

    def _collection_delete(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs,
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
        logger.debug(f"Deleting data from collection '{collection_name}'")

        # Validate that at least one filter is provided
        if not ids and not where and not where_document:
            raise ValueError("At least one of ids, where, or where_document must be provided")

        # Normalize ids to list
        id_list = None
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else ids

        # Get table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
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

        logger.debug(f"✅ Successfully deleted data from collection '{collection_name}'")

    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() are implemented below with common SQL-based logic

    def _normalize_query_embeddings(
        self, query_embeddings: list[float] | list[list[float]] | None
    ) -> list[list[float]]:
        """
        Normalize query embeddings to list of lists format

        Args:
            query_embeddings: Single vector or list of embeddings

        Returns:
            List of embeddings (each vector is a list of floats)
        """
        if query_embeddings is None:
            return []

        # Check if it's a single vector (list of numbers)
        if query_embeddings and isinstance(query_embeddings[0], (int, float)):
            return [query_embeddings]

        return query_embeddings

    def _normalize_include_fields(self, include: list[str] | None) -> dict[str, bool]:
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
        texts: str | list[str],
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """
        Embed text(s) to vector(s)

        Args:
            texts: Single text or list of texts
            embedding_function: EmbeddingFunction instance to convert texts to embeddings.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
                               If not provided, raises NotImplementedError.
            **kwargs: Additional parameters for embedding (unused for now)

        Returns:
            List of embeddings (List[List[float]]), where each inner list is an embedding vector

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

    def _normalize_row(self, row: Any, cursor_description: Any | None = None) -> dict[str, Any]:
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
        return dict(row) if hasattr(row, "_asdict") else row

    def _execute_query_with_cursor(
        self, conn: Any, sql: str, params: list[Any], use_context_manager: bool = True
    ) -> list[dict[str, Any]]:
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
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {sql}  -- params={params}", flush=True)
        try:
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    if not self._should_fetch_results(cursor, sql):
                        return []
                    rows = cursor.fetchall()
                    # Normalize rows
                    normalized_rows = []
                    for row in rows:
                        normalized_rows.append(self._normalize_row(row, cursor.description))
                    return normalized_rows
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                if not self._should_fetch_results(cursor, sql):
                    return []
                rows = cursor.fetchall()
                # Normalize rows
                normalized_rows = []
                for row in rows:
                    normalized_rows.append(self._normalize_row(row, cursor.description))
                return normalized_rows
            finally:
                cursor.close()
        except Exception as exc:
            maybe_reraise_friendly_kernel_error(exc)
            raise

    def _build_select_clause(self, include_fields: dict[str, bool]) -> str:
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
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        id_list: list[str] | None = None,
    ) -> tuple[str, list[Any]]:
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
                id_sql, id_param = self._convert_id_to_sql_with_paramters(id_val)
                processed_ids.append(id_sql)
                params.append(id_param)

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

        # Use pymysql's escape_string for safe escaping
        id_val_escaped = escape_string(id_val)
        # Use CAST to convert string to binary for varbinary(512) field
        return f"CAST('{id_val_escaped}' AS BINARY)"

    def _convert_id_to_sql_with_paramters(self, id_val: str) -> (str, str):
        """
        Convert ID to SQL format for varbinary(512) _id field with parameters
        """
        return "CAST(%s AS BINARY)", (id_val)

    def _convert_id_from_bytes(self, record_id: Any) -> str:
        """
        Convert _id from bytes to string format

        Args:
            record_id: Record ID from database (can be bytes, str, or other format)

        Returns:
            String ID
        """
        if record_id is None:
            return None

        # If it's already a string, return as is
        if isinstance(record_id, str):
            return record_id

        # Convert bytes to string (UTF-8 decode)
        if isinstance(record_id, bytes):
            try:
                return record_id.decode("utf-8")
            except UnicodeDecodeError:
                # If UTF-8 decode fails, return hex representation as fallback
                return record_id.hex()

        # For other formats, convert to string
        return str(record_id)

    def _process_query_row(self, row: dict[str, Any], include_fields: dict[str, bool]) -> dict[str, Any]:
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

    def _process_get_row(self, row: dict[str, Any], include_fields: dict[str, bool]) -> dict[str, Any]:
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
        if (include_fields.get("documents") or include_fields.get("document")) and "document" in row:
            document = row["document"]

        # Include metadata if requested
        if (include_fields.get("metadatas") or include_fields.get("metadata")) and row.get("metadata") is not None:
            metadata = self._parse_row_value(row["metadata"])

        # Include embedding if requested
        if (include_fields.get("embeddings") or include_fields.get("embedding")) and row.get("embedding") is not None:
            embedding = self._parse_row_value(row["embedding"])

        return {
            "id": record_id,
            "document": document,
            "embedding": embedding,
            "metadata": metadata,
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

    def _should_fetch_results(self, cursor: Any, sql: str) -> bool:
        """Return whether the given SQL statement is expected to yield result rows."""
        description = getattr(cursor, "description", None)
        if description is not None:
            return True
        return is_query_sql(sql)

    def _execute(self, sql: str) -> Any:
        """Execute a SQL statement against the connection and return any result rows."""
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {sql}", flush=True)
        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()

        try:
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    if self._should_fetch_results(cursor, sql):
                        return cursor.fetchall()
                    return None

            cursor = conn.cursor()
            try:
                cursor.execute(sql)
                if self._should_fetch_results(cursor, sql):
                    return cursor.fetchall()
                return None
            finally:
                cursor.close()
        except Exception as exc:
            maybe_reraise_friendly_kernel_error(exc)
            raise

    # -------------------- DQL Operations (Common Implementation) --------------------

    def _collection_query(
        self,
        collection_id: str | None,
        collection_name: str,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        query_key: FieldKey | None = None,
        query_hint: QueryHint | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
                embedding_function: EmbeddingFunction instance to convert query_texts to embeddings.
                                   Required if query_texts is provided and collection doesn't have
                                   an embedding_function set. Must implement __call__ method that
                                   accepts Documents and returns Embeddings (List[List[float]]).
                distance: Distance metric to use for similarity calculation (e.g., 'l2', 'cosine', 'inner_product').
                         Defaults to 'l2' if not provided.

        Returns:
            Dict with keys:
            - ids: List[List[str]] - List of ID lists, one list per query
            - documents: Optional[List[List[str]]] - List of document lists, one list per query
            - metadatas: Optional[List[List[Dict]]] - List of metadata lists, one list per query
            - embeddings: Optional[List[List[List[float]]]] - List of embedding lists, one list per query
            - distances: Optional[List[List[float]]] - List of distance lists, one list per query
        """
        logger.debug(f"Querying collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()

        # Convert collection name to table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
            table_name = CollectionNames.table_name(collection_name)

        # Check if this is a sparse vector query
        sparse_config = kwargs.get("sparse_vector_index_config")
        is_sparse_query = query_key is not None and (
            query_key is FieldKey.SPARSE_EMBEDDING
            or query_key == FieldKey.SPARSE_EMBEDDING.name
            or (hasattr(query_key, "name") and query_key.name == "#sparse_embedding")
        )

        if is_sparse_query:
            return self._collection_query_sparse(
                conn=conn,
                table_name=table_name,
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
                sparse_config=sparse_config,
                collection_name=collection_name,
                query_hint=query_hint,
                **kwargs,
            )

        # ===== Dense vector query path =====
        # Handle vector generation logic:
        # 1. If query_embeddings are provided, use them directly without embedding
        # 2. If query_embeddings are not provided but query_texts are provided:
        #    - If embedding_function is provided, use it to generate embeddings from query_texts
        #    - If embedding_function is not provided, raise an error
        # 3. If neither query_embeddings nor query_texts are provided, raise an error

        embedding_function = kwargs.get("embedding_function")

        if query_embeddings is not None:
            # Query embeddings provided, use them directly without embedding
            pass
        elif query_texts is not None:
            # Query embeddings not provided but query_texts are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug("Embedding query texts...")
                query_embeddings = self._embed_texts(query_texts, embedding_function=embedding_function)
            else:
                raise ValueError(
                    "query_texts provided but no query_embeddings and no embedding_function. "
                    "Either:\n"
                    "  1. Provide query_embeddings directly, or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from query_texts."
                )
        else:
            # Neither query_embeddings nor query_texts provided, raise an error
            raise ValueError(
                "Neither query_embeddings nor query_texts provided. "
                "Please provide either:\n"
                "  1. query_embeddings directly, or\n"
                "  2. query_texts with embedding_function to generate embeddings."
            )

        # Normalize query embeddings to list of lists
        query_embeddings = self._normalize_query_embeddings(query_embeddings)

        # Normalize include fields
        include_fields = self._normalize_include_fields(include)

        # Build SELECT clause
        select_clause = self._build_select_clause(include_fields)

        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document)

        # Get distance metric from kwargs, default to DEFAULT_DISTANCE_METRIC if not provided
        distance = kwargs.get("distance", DEFAULT_DISTANCE_METRIC)

        # Map distance metric to SQL function name
        distance_function_map = {
            "l2": "l2_distance",
            "cosine": "cosine_distance",
            "inner_product": "inner_product",
        }

        # Get the distance function name, default to 'l2_distance' if distance is not recognized
        distance_func = distance_function_map.get(distance, "l2_distance")

        if distance not in distance_function_map:
            logger.warning(f"Unknown distance metric '{distance}', defaulting to 'l2_distance'")

        use_context_manager = self._use_context_manager_for_cursor()

        # Collect results for each query vector separately
        all_ids = []
        all_documents = []
        all_metadatas = []
        all_embeddings = []
        all_distances = []

        for query_vector in query_embeddings:
            # Convert vector to string format for SQL
            vector_str = _embedding_to_hexstring(query_vector)

            # Build query hint
            hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)

            # Build SQL query with vector distance calculation
            # Reference: SELECT id, vec FROM t2 ORDER BY l2_distance(vec, '[0.1, 0.2, 0.3]') APPROXIMATE LIMIT 5;
            # Need to include distance in SELECT for result processing
            # Use the appropriate distance function based on the index configuration
            sql = f"""
                SELECT {hint_sql} {select_clause},
                       {distance_func}(embedding, {vector_str}) AS distance
                FROM `{table_name}`
                {where_clause}
                ORDER BY {distance_func}(embedding, {vector_str})
                APPROXIMATE
                LIMIT %s
            """

            # Execute query
            query_params = [*params, n_results]
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")

            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)

            # Collect results for this query vector
            query_ids = []
            query_documents = []
            query_metadatas = []
            query_embeddings = []
            query_distances = []

            for row in rows:
                result_item = self._process_query_row(row, include_fields)
                query_ids.append(result_item.get("_id"))

                if "documents" in include_fields or include is None:
                    query_documents.append(result_item.get("document"))

                if "metadatas" in include_fields or include is None:
                    query_metadatas.append(result_item.get("metadata") or {})

                if "embeddings" in include_fields:
                    query_embeddings.append(result_item.get("embedding"))

                query_distances.append(result_item.get("distance"))

            all_ids.append(query_ids)
            if "documents" in include_fields or include is None:
                all_documents.append(query_documents)
            if "metadatas" in include_fields or include is None:
                all_metadatas.append(query_metadatas)
            if "embeddings" in include_fields:
                all_embeddings.append(query_embeddings)
            all_distances.append(query_distances)

        # Build result dictionary in chromadb format
        result = {"ids": all_ids, "distances": all_distances}

        if "documents" in include_fields or include is None:
            result["documents"] = all_documents

        if "metadatas" in include_fields or include is None:
            result["metadatas"] = all_metadatas

        if "embeddings" in include_fields:
            result["embeddings"] = all_embeddings

        logger.debug(
            f"✅ Query completed for '{collection_name}' with {len(query_embeddings)} vectors, returning {len(all_ids)} result lists"
        )
        return result

    def _collection_query_sparse(
        self,
        conn,
        table_name: str,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        sparse_config=None,
        collection_name: str = "",
        query_hint=None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        [Internal] Query collection by sparse vector similarity.

        Supports:
        1. Text-based sparse vector query via query_texts + sparse embedding function

        Args:
            conn: Database connection
            table_name: Table name
            query_embeddings: Not supported for sparse query path. Use query_texts instead.
            query_texts: Query text(s) to be converted to sparse vectors
            n_results: Number of results
            where: Metadata filter
            where_document: Document filter
            include: Fields to include
            sparse_config: SparseVectorIndexConfig instance
            collection_name: Collection name (for logging)
        """
        logger.debug(f"Sparse vector query on collection '{collection_name}'")

        # Resolve sparse query vectors
        sparse_query_vectors: list[SparseVector] = []

        if query_embeddings is not None:
            raise ValueError(
                "For sparse vector queries, query_embeddings is not supported. "
                "Please provide query_texts and use the configured sparse embedding function."
            )
        elif query_texts is not None:
            # Generate sparse vectors from query texts using sparse embedding function
            if sparse_config is None or sparse_config.embedding_function is None:
                raise ValueError(
                    "query_texts provided for sparse vector query but no sparse embedding function is configured."
                )
            sparse_ef = sparse_config.embedding_function
            # Normalize query_texts to list
            if isinstance(query_texts, str):
                query_texts = [query_texts]
            logger.debug(f"Generating sparse embeddings for {len(query_texts)} query texts...")
            sparse_vectors = sparse_ef(query_texts)
            sparse_query_vectors = sparse_vectors
        else:
            raise ValueError(
                "Neither query_embeddings nor query_texts provided for sparse vector query. "
                "Please provide query_texts with a configured sparse embedding function."
            )

        if not sparse_query_vectors:
            raise ValueError("No sparse query vectors resolved.")

        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)

        # Normalize include fields
        include_fields = self._normalize_include_fields(include)

        # Build SELECT clause
        select_clause = self._build_select_clause(include_fields)

        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document)

        # Sparse vector queries always use inner_product distance
        distance_func = "inner_product"

        use_context_manager = self._use_context_manager_for_cursor()

        # Collect results for each sparse query vector separately
        all_ids = []
        all_documents = []
        all_metadatas = []
        all_embeddings = []
        all_distances = []

        for sv in sparse_query_vectors:
            # Convert sparse vector to SQL string format
            sv_sql = _sparse_vector_to_sql(sv)

            # Build SQL query with sparse vector distance calculation
            sql = f"""
                SELECT {hint_sql} {select_clause},
                       {distance_func}(sparse_embedding, {sv_sql}) AS distance
                FROM `{table_name}`
                {where_clause}
                ORDER BY {distance_func}(sparse_embedding, {sv_sql})
                APPROXIMATE
                LIMIT %s
            """.strip()

            # Execute query
            query_params = [*params, n_results]
            logger.debug(f"Executing sparse SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")

            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)

            # Collect results for this query vector
            query_ids = []
            query_documents = []
            query_metadatas = []
            query_embeddings_list = []
            query_distances = []

            for row in rows:
                result_item = self._process_query_row(row, include_fields)
                query_ids.append(result_item.get("_id"))

                if "documents" in include_fields or include is None:
                    query_documents.append(result_item.get("document"))

                if "metadatas" in include_fields or include is None:
                    query_metadatas.append(result_item.get("metadata") or {})

                if "embeddings" in include_fields:
                    query_embeddings_list.append(result_item.get("embedding"))

                query_distances.append(result_item.get("distance"))

            all_ids.append(query_ids)
            if "documents" in include_fields or include is None:
                all_documents.append(query_documents)
            if "metadatas" in include_fields or include is None:
                all_metadatas.append(query_metadatas)
            if "embeddings" in include_fields:
                all_embeddings.append(query_embeddings_list)
            all_distances.append(query_distances)

        # Build result dictionary in chromadb format
        result = {"ids": all_ids, "distances": all_distances}

        if "documents" in include_fields or include is None:
            result["documents"] = all_documents

        if "metadatas" in include_fields or include is None:
            result["metadatas"] = all_metadatas

        if "embeddings" in include_fields:
            result["embeddings"] = all_embeddings

        logger.debug(
            f"Sparse query completed for '{collection_name}' with {len(sparse_query_vectors)} vectors, "
            f"returning {len(all_ids)} result lists"
        )
        return result

    def _collection_get(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
            query_hint: Query optimization hints for database execution (optional)
            **kwargs: Additional parameters

        Returns:
            Dict with keys:
            - ids: List[str] - List of IDs
            - documents: Optional[List[str]] - List of documents
            - metadatas: Optional[List[Dict]] - List of metadata dictionaries
            - embeddings: Optional[List[List[float]]] - List of embeddings
        """
        logger.debug(f"Getting data from collection '{collection_name}'")
        conn = self._ensure_connection()

        # Convert collection name to table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
            table_name = CollectionNames.table_name(collection_name)

        # Set defaults
        if limit is None:
            limit = 100
        if offset is None:
            offset = 0

        # Normalize ids to list
        id_list = None
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else ids

        # Note: get() now returns dict format (not QueryResult)
        # Normalize include fields (default includes documents and metadatas)
        include_fields = self._normalize_include_fields(include)

        # Build SELECT clause - always include _id
        select_clause = self._build_select_clause(include_fields)

        use_context_manager = self._use_context_manager_for_cursor()

        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document, id_list)

        # Build query hint
        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)

        # Build SQL query
        sql = f"""
            SELECT {hint_sql} {select_clause}
            FROM `{table_name}`
            {where_clause}
            LIMIT %s OFFSET %s
        """

        # Execute query
        query_params = [*params, limit, offset]
        logger.debug(f"Executing SQL: {sql}")
        logger.debug(f"Parameters: {query_params}")

        rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)

        # Build result dictionary in chromadb format
        result_ids = []
        result_documents = []
        result_metadatas = []
        result_embeddings = []

        for row in rows:
            processed_row = self._process_get_row(row, include_fields)
            result_ids.append(processed_row["id"])

            if "documents" in include_fields or include is None:
                result_documents.append(processed_row["document"])

            if "metadatas" in include_fields or include is None:
                result_metadatas.append(processed_row["metadata"] or {})

            if "embeddings" in include_fields:
                result_embeddings.append(processed_row["embedding"])

        # Build result dictionary
        result = {"ids": result_ids}

        if "documents" in include_fields or include is None:
            result["documents"] = result_documents

        if "metadatas" in include_fields or include is None:
            result["metadatas"] = result_metadatas

        if "embeddings" in include_fields:
            result["embeddings"] = result_embeddings

        logger.debug(f"✅ Get completed for '{collection_name}', found {len(result_ids)} results")
        return result

    def _collection_hybrid_search(
        self,
        collection_id: str | None,
        collection_name: str,
        query: dict[str, Any] | None = None,
        knn: dict[str, Any] | None = None,
        rank: dict[str, Any] | None = None,
        n_results: int = 10,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        dimension: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
            dimension: Collection vector dimension for validating query_embeddings (optional)
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts in knn to embeddings.
                                   Required if knn.query_texts is provided and collection doesn't have
                                   an embedding_function set. Must implement __call__ method that
                                   accepts Documents and returns Embeddings (List[List[float]]).

        Returns:
            Dict with keys (query-compatible format):
            - ids: List[List[str]] - List of ID lists (one list for hybrid search result)
            - documents: Optional[List[List[str]]] - List of document lists (if included)
            - metadatas: Optional[List[List[Dict]]] - List of metadata lists (if included)
            - embeddings: Optional[List[List[List[float]]]] - List of embedding lists (if included)
            - distances: Optional[List[List[float]]] - List of distance lists
        """
        logger.debug(f"Hybrid search in collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()

        # Build table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
            table_name = CollectionNames.table_name(collection_name)

        # Build search_parm JSON
        search_parm = self._build_search_parm(
            query,
            knn,
            rank,
            n_results,
            include=include,
            dimension=dimension,
            **kwargs,
        )

        # Convert search_parm to JSON string
        search_parm_json = json.dumps(search_parm, ensure_ascii=False)

        # Use variable binding to avoid datatype issues
        use_context_manager = self._use_context_manager_for_cursor()

        # Set the search_parm variable first (use safe escaping)
        escaped_params = escape_string(search_parm_json)
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
            logger.warning("No SQL query returned from GET_SQL")
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }

        # Get the SQL query string
        query_sql = rows[0]["query_sql"]
        if isinstance(query_sql, str):
            # Remove any surrounding quotes if present
            query_sql = query_sql.strip().strip("'\"")

        # OB's GET_SQL can wrap JSON_EXTRACT expressions in backticks, which
        # turns them into literal column names. Unquote only those complete
        # expressions; ordinary identifiers around them must remain quoted.
        query_sql = _unquote_json_extract_expressions(query_sql)

        # Add query hint to the generated SQL
        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)
        if hint_sql and query_sql.upper().startswith("SELECT"):
            # Insert hint after SELECT keyword
            query_sql = f"SELECT {hint_sql} {query_sql[len('SELECT') :]}"

        logger.debug(f"Executing query SQL: {query_sql}")

        # Execute the returned SQL query
        result_rows = self._execute_query_with_cursor(conn, query_sql, [], use_context_manager)

        # Transform SQL query results to standard format
        return self._transform_sql_result(result_rows, include)

    def _build_search_parm(
        self,
        query: dict[str, Any] | list[dict[str, Any]] | None,
        knn: dict[str, Any] | list[dict[str, Any]] | None,
        rank: dict[str, Any] | None,
        n_results: int,
        include: list[str] | None = None,
        dimension: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Build search_parm JSON from query, knn, and rank parameters

        Args:
            query: Full-text search configuration dict or list of dicts
            knn: Vector search configuration dict or list of dicts
            rank: Ranking configuration dict
            n_results: Final number of results to return
            include: Fields requested by the SDK caller. Used to infer the minimal OceanBase GET_SQL
                `_source` allowlist to avoid returning large unused columns (e.g. `embedding`).
            dimension: Collection dimension for validating query_embeddings (optional)
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts in knn to embeddings.
                                   Required if knn.query_texts is provided. Must implement __call__
                                   method that accepts Documents and returns Embeddings (List[List[float]]).

        Returns:
            search_parm dictionary
        """
        search_parm = {}

        # Build query part (full-text search or scalar query)
        query_expr_list: list[dict[str, Any]] = []
        if query:
            query_items = query if isinstance(query, list) else [query]
            for query_item in query_items:
                query_expr = self._build_query_expression(query_item)
                if query_expr:
                    query_expr_list.append(query_expr)
        if query_expr_list:
            search_parm["query"] = query_expr_list if len(query_expr_list) > 1 else query_expr_list[0]

        # Build knn part (vector search)
        knn_expr_list: list[dict[str, Any]] = []
        if knn:
            knn_items = knn if isinstance(knn, list) else [knn]
            for knn_item in knn_items:
                knn_expr = self._build_knn_expression(knn_item, dimension=dimension, **kwargs)
                if not knn_expr:
                    continue
                if isinstance(knn_expr, list):
                    knn_expr_list.extend(knn_expr)
                else:
                    knn_expr_list.append(knn_expr)
        if knn_expr_list:
            search_parm["knn"] = knn_expr_list if len(knn_expr_list) > 1 else knn_expr_list[0]

        if n_results is not None:
            search_parm["size"] = n_results

        # Build rank part
        if rank:
            search_parm["rank"] = rank

        # Always infer a minimal `_source` allowlist from include to reduce response payload.
        search_parm["_source"] = self._build_source_fields(include)

        return search_parm

    @staticmethod
    def _pure_must_not_clauses(expr: dict[str, Any] | None) -> list[dict[str, Any]] | None:
        """Return inner must_not clauses when *expr* is ``{"bool": {"must_not": [...]}}`` only."""
        if not isinstance(expr, dict) or set(expr.keys()) != {"bool"}:
            return None
        bool_node = expr.get("bool")
        if not isinstance(bool_node, dict) or set(bool_node.keys()) != {"must_not"}:
            return None
        clauses = bool_node.get("must_not")
        if not isinstance(clauses, list):
            return None
        return clauses

    @staticmethod
    def _collect_scalar_fields_from_filter_clause(clause: dict[str, Any]) -> list[str]:
        """Collect scalar field names from term/terms leaves (including nested bool clauses)."""
        fields: list[str] = []
        if not isinstance(clause, dict):
            return fields
        term_body = clause.get("term")
        if isinstance(term_body, dict):
            fields.extend(term_body.keys())
        terms_body = clause.get("terms")
        if isinstance(terms_body, dict):
            fields.extend(terms_body.keys())
        range_body = clause.get("range")
        if isinstance(range_body, dict):
            fields.extend(range_body.keys())
        bool_node = clause.get("bool")
        if isinstance(bool_node, dict):
            for key in ("filter", "must", "should", "must_not"):
                sub = bool_node.get(key)
                if isinstance(sub, list):
                    for item in sub:
                        fields.extend(BaseClient._collect_scalar_fields_from_filter_clause(item))
                elif isinstance(sub, dict):
                    fields.extend(BaseClient._collect_scalar_fields_from_filter_clause(sub))
        return fields

    @staticmethod
    def _positive_clause_for_must_not(must_not_clauses: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a permissive positive filter leaf for must_not-only bools (OB rejects match_all)."""
        for clause in must_not_clauses:
            if not isinstance(clause, dict):
                continue
            for field in BaseClient._collect_scalar_fields_from_filter_clause(clause):
                return {"range": {field: {"gte": -9223372036854775808}}}
            qs_body = clause.get("query_string")
            if isinstance(qs_body, dict):
                fields = qs_body.get("fields") or ["document"]
                field = fields[0] if fields else "document"
                return {"exists": {"field": field}}
        return {"exists": {"field": "document"}}

    @staticmethod
    def _hoist_must_not_from_filters(
        filter_conditions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split a filter clause list into positive filters and hoisted must_not leaves."""
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        for cond in filter_conditions:
            clauses = BaseClient._pure_must_not_clauses(cond)
            if clauses is not None:
                negative.extend(clauses)
            else:
                positive.append(cond)
        return positive, negative

    def _build_query_expression(self, query: dict[str, Any]) -> dict[str, Any] | None:
        """
        Build query expression from query dict

        Supports:
        - Scalar query (metadata filtering only): query.range or query.term
        - Full-text search: query.query_string
        - Full-text search with metadata filtering: query.bool with must and filter
        """
        where_document = query.get("where_document")
        where = query.get("where")
        boost = query.get("boost")

        # Case 1: Scalar query (metadata filtering only, no full-text search)
        if not where_document and where:
            filter_conditions = self._build_metadata_filter_for_search_parm(where)
            if filter_conditions:
                # Wrap scalar conditions in a (non-scoring) `filter` clause: a
                # top-level bool is scoring by default and the kernel rejects scalar
                # term/range queries inside must/should of a scoring bool
                # (`scalar ... query in must/should clause not supported`). Negation
                # conditions ($not/$ne/$nin) come back as pure `{"bool": {"must_not"}}`
                # nodes; a bool with only must_not is rejected with `bool query ...
                # should have at least one positive clause`, so hoist their must_not
                # clauses onto the outer bool (which gains a positive `filter` once the
                # namespace filter is injected) instead of nesting them standalone.
                positive, negative = self._hoist_must_not_from_filters(filter_conditions)
                bool_q: dict[str, Any] = {}
                if positive:
                    bool_q["filter"] = positive
                if negative:
                    bool_q["must_not"] = negative
                return {"bool": bool_q}

        # Case 2: Full-text search (with or without metadata filtering)
        if where_document:
            # Build document query using query_string
            doc_query = self._build_document_query(where_document, boost=boost)
            if doc_query:
                filter_conditions = self._build_metadata_filter_for_search_parm(where)
                pos_filters, meta_must_not = self._hoist_must_not_from_filters(filter_conditions)
                doc_must_not = self._pure_must_not_clauses(doc_query)
                must_not_all = list(meta_must_not)
                if doc_must_not is not None:
                    must_not_all.extend(doc_must_not)

                if not filter_conditions and doc_must_not is None:
                    return doc_query

                bool_q: dict[str, Any] = {}
                if doc_must_not is None:
                    bool_q["must"] = [doc_query]
                if pos_filters:
                    bool_q["filter"] = pos_filters
                if must_not_all:
                    bool_q["must_not"] = must_not_all
                return {"bool": bool_q}

        return None

    def _build_document_query(
        self, where_document: dict[str, Any], boost: float | None = None
    ) -> dict[str, Any] | None:
        """
        Build document query from where_document condition using query_string

        Args:
            where_document: Document filter conditions
            boost: Optional weight for this document query

        Returns:
            query_string query dict
        """
        if not where_document:
            return None

        def _with_boost(expr: dict[str, Any] | None) -> dict[str, Any] | None:
            """Apply field boosting to a document query expression."""
            if boost is None or not expr:
                return expr

            def _apply_boost(target: Any) -> None:
                """Apply a boost factor to a single field expression."""
                if not isinstance(target, dict):
                    return
                if "query_string" in target and isinstance(target["query_string"], dict):
                    target["query_string"]["boost"] = boost
                    return
                bool_clause = target.get("bool")
                if isinstance(bool_clause, dict):
                    for key in ("must", "should", "must_not", "filter"):
                        clause = bool_clause.get(key)
                        if isinstance(clause, list):
                            for item in clause:
                                _apply_boost(item)
                        elif isinstance(clause, dict):
                            _apply_boost(clause)

            _apply_boost(expr)
            return expr

        return _with_boost(build_document_hybrid_expression(where_document, boost=boost))

    def _build_metadata_filter_for_search_parm(self, where: dict[str, Any] | None) -> list[dict[str, Any]]:
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

    def _build_search_parm_field_name(self, key: str) -> str:
        """
        Build field name used in search_parm filters.

        Uses ``JSON_EXTRACT``-wrapped keys for ``DBMS_HYBRID_SEARCH.GET_SQL`` (collection path).
        Namespace ``hybrid_search(TABLE ...)`` rewrites these to ``data_content.metadata.*`` DSL
        keys in ``_adapt_search_parm_for_ns``.
        """
        if key == "#id" or key == CollectionFieldNames.ID:
            return CollectionFieldNames.ID
        return f"(JSON_EXTRACT(metadata, '$.{key}'))"

    def _build_metadata_filter_conditions(self, condition: dict[str, Any]) -> list[dict[str, Any]]:
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
                # Scalar conditions must be ANDed via a (non-scoring) `filter`
                # clause, not `must`: the kernel rejects scalar term/range queries
                # inside must/should with `scalar ... query in must/should clause
                # not supported`. Hoist must_not-only leaves ($ne/$nin/$not) so they
                # are not nested as standalone bools inside `filter`.
                positive, negative = self._hoist_must_not_from_filters(must_conditions)
                bool_q: dict[str, Any] = {}
                if positive:
                    bool_q["filter"] = positive
                elif negative:
                    bool_q["filter"] = [self._positive_clause_for_must_not(negative)]
                if negative:
                    bool_q["must_not"] = negative
                result.append({"bool": bool_q})
            return result

        if "$or" in condition:
            should_conditions = []
            for sub_condition in condition["$or"]:
                sub_filters = self._build_metadata_filter_conditions(sub_condition)
                should_conditions.extend(sub_filters)
            if should_conditions:
                # `minimum_should_match: 1` makes this an explicit OR. In a
                # non-scoring (filter) context the kernel does not reliably apply the
                # implicit "at least one should" default, which otherwise yields an
                # intermittent `1210 Invalid argument`.
                result.append({"bool": {"should": should_conditions, "minimum_should_match": 1}})
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

            # Build field name with JSON_EXTRACT format (or _id for special key)
            field_name = self._build_search_parm_field_name(key)

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
                        # For $in, use terms query to match any value in list
                        if isinstance(op_value, (list, tuple)) and len(op_value) > 0:
                            result.append({"terms": {field_name: list(op_value)}})
                    elif op == "$nin" and isinstance(op_value, (list, tuple)) and len(op_value) > 0:
                        # For $nin, use must_not with terms query
                        result.append({"bool": {"must_not": [{"terms": {field_name: list(op_value)}}]}})

                if range_conditions:
                    result.append({"range": {field_name: range_conditions}})
                elif term_value is not None:
                    result.append({"term": {field_name: term_value}})
            else:
                # Direct equality
                result.append({"term": {field_name: value}})

        return result

    def _build_knn_expression(
        self, knn: dict[str, Any], dimension: int | None = None, **kwargs
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Build knn expression from knn dict

        Args:
            knn: Vector search configuration dict with:
                - query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
                - query_embeddings: Query vector(s) (optional if query_texts provided)
                - where: Metadata filter conditions (optional)
                - n_results: Number of results for vector search (optional)
                - boost: Optional weight for this knn search route
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts to embeddings.
                                   Required if query_texts is provided. Must implement __call__
                                   method that accepts Documents and returns Embeddings (List[List[float]]).
            dimension: Optional collection dimension for validating embeddings

        Returns:
            knn expression dict (or list of dicts when multiple query vectors) with optional filter
        """
        query_texts = knn.get("query_texts")
        query_embeddings = knn.get("query_embeddings")
        where = knn.get("where")
        where_document = knn.get("where_document")
        n_results = knn.get("n_results", 10)
        if not isinstance(n_results, int) or n_results < 1:
            raise ValueError(f"n_results must be an integer >= 1, got {n_results!r}")
        if n_results > _MAX_N_RESULTS:
            raise ValueError(
                f"n_results must be <= {_MAX_N_RESULTS}, got {n_results}. "
                "Use a smaller value or paginate with offset/limit."
            )
        boost = knn.get("boost")

        embedding_function = kwargs.get("embedding_function")

        self._warn_explicit_embeddings_override_embedding_function(
            operation="hybrid_search.knn",
            explicit_embeddings=query_embeddings is not None,
            has_documents=query_texts is not None,
            embedding_function=embedding_function,
        )

        def _normalize_vectors(raw_embeddings: Any) -> list[list[float]]:
            """Normalize input vectors to a consistent list-of-floats form."""
            if raw_embeddings is None:
                return []
            if isinstance(raw_embeddings, list) and raw_embeddings and isinstance(raw_embeddings[0], list):
                return raw_embeddings  # type: ignore[return-value]
            if isinstance(raw_embeddings, list):
                return [raw_embeddings]  # type: ignore[list-item]
            return []

        vectors: list[list[float]] = []
        if query_embeddings is not None:
            vectors = _normalize_vectors(query_embeddings)
        elif query_texts is not None:
            if embedding_function is not None:
                try:
                    texts = query_texts if isinstance(query_texts, list) else [query_texts]
                    embeddings = self._embed_texts(texts, embedding_function=embedding_function)
                    if embeddings and len(embeddings) > 0:
                        vectors = embeddings
                except Exception as e:
                    logger.exception("Failed to generate embeddings from query_texts")
                    raise ValueError(f"Failed to generate embeddings from query_texts: {e}") from e
            else:
                raise ValueError(
                    "knn.query_texts provided but no knn.query_embeddings and no embedding_function. "
                    "Either:\n"
                    "  1. Provide knn.query_embeddings directly, or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from knn.query_texts."
                )
        else:
            raise ValueError(
                "knn requires either query_embeddings or query_texts. "
                "Please provide either:\n"
                "  1. knn.query_embeddings directly, or\n"
                "  2. knn.query_texts with embedding_function to generate embeddings."
            )

        if not vectors:
            return None

        if dimension is not None:
            for vec in vectors:
                if len(vec) != dimension:
                    raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {len(vec)}")

        # Build knn expressions (one per vector)
        knn_exprs: list[dict[str, Any]] = []
        filter_conditions = self._build_metadata_filter_for_search_parm(where)
        pos_filters, must_not_clauses = self._hoist_must_not_from_filters(filter_conditions)
        knn_filter: list[dict[str, Any]] | None = None
        if must_not_clauses:
            bool_filter: dict[str, Any] = {
                "must_not": must_not_clauses,
                "filter": pos_filters or [self._positive_clause_for_must_not(must_not_clauses)],
            }
            knn_filter = [{"bool": bool_filter}]
        elif pos_filters:
            knn_filter = pos_filters

        if where_document is not None and where_document_knn_prefilterable(where_document):
            doc_filter = document_expr_as_knn_filter(build_document_hybrid_expression(where_document))
            if doc_filter is not None:
                knn_filter = [doc_filter] if knn_filter is None else [*knn_filter, doc_filter]

        for vector in vectors:
            expr = {"field": "embedding", "k": n_results, "query_vector": vector}
            if boost is not None:
                expr["boost"] = boost

            if knn_filter is not None:
                expr["filter"] = knn_filter

            knn_exprs.append(expr)

        return knn_exprs if len(knn_exprs) > 1 else knn_exprs[0]

    def _post_filter_namespace_query_result(
        self,
        result: dict[str, Any],
        where_document: dict[str, Any] | str,
        *,
        n_results: int,
    ) -> dict[str, Any]:
        """Drop hybrid_search rows that violate a where_document predicate."""
        ids_groups = result.get("ids") or []
        if not ids_groups:
            return result

        filtered: dict[str, Any] = {"ids": []}
        if result.get("distances") is not None:
            filtered["distances"] = []
        for optional_key in ("documents", "metadatas", "embeddings"):
            if optional_key in result:
                filtered[optional_key] = []

        for qi, ids in enumerate(ids_groups):
            kept_indices: list[int] = []
            docs_group = (result.get("documents") or [[]])[qi] if result.get("documents") else None
            for idx, _doc_id in enumerate(ids):
                if not docs_group or idx >= len(docs_group) or docs_group[idx] is None:
                    continue
                doc_text = str(docs_group[idx])
                if doc_matches_where_document(doc_text, where_document):
                    kept_indices.append(idx)
                if len(kept_indices) >= n_results:
                    break

            filtered["ids"].append([ids[i] for i in kept_indices])
            if result.get("distances"):
                dist_groups = result.get("distances")
                if dist_groups and qi < len(dist_groups):
                    group = dist_groups[qi]
                    filtered["distances"].append([group[i] for i in kept_indices if i < len(group)])
            for optional_key in ("documents", "metadatas", "embeddings"):
                if optional_key in filtered:
                    groups = result.get(optional_key)
                    if groups and qi < len(groups):
                        group = groups[qi]
                        filtered[optional_key].append([group[i] for i in kept_indices if i < len(group)])

        return filtered

    def _build_source_fields(self, include: list[str] | None) -> list[str]:
        """
        Infer OceanBase GET_SQL `_source` allowlist from include.
        """
        if include is None:
            requested = {"documents", "metadatas"}
        else:
            if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
                raise TypeError("include must be a List[str] or None")
            requested = {item.lower() for item in include}

        source = ["_id"]

        if {"documents", "document"} & requested:
            source.append("document")
        if {"metadatas", "metadata"} & requested:
            source.append("metadata")
        if {"embeddings", "embedding"} & requested:
            source.append("embedding")

        return source

    def _hybrid_row_score(self, row: dict[str, Any]) -> float:
        """Extract relevance score from a hybrid_search SQL row (OB uses ``__score``)."""
        for key in (
            "_distance",
            "distance",
            "_score",
            "score",
            "__score",
            "DISTANCE",
            "_DISTANCE",
            "SCORE",
            "__SCORE",
        ):
            val = row.get(key)
            if val is not None:
                return float(val)
        return 0.0

    def _transform_sql_result(self, result_rows: list[dict[str, Any]], include: list[str] | None) -> dict[str, Any]:
        """
        Transform SQL query results to standard format (query-compatible format)

        Args:
            result_rows: List of row dictionaries from SQL query
            include: Fields to include in results (optional)

        Returns:
            Standard format dictionary with ids, distances, metadatas, documents, embeddings
            in query-compatible format (List[List[...]] for consistency with query method)
        """

        ids = []
        distances = []
        metadatas = []
        documents = []
        embeddings = []

        for row in result_rows:
            # Extract id (handle different column names and fallbacks)
            row_id = None
            for key in ("id", "_id", "ID", "Id", "_ID"):
                if key in row and row.get(key) is not None:
                    row_id = row.get(key)
                    break
            if row_id is None:
                for key in row:
                    if isinstance(key, str) and key.lower().endswith("id") and row.get(key) is not None:
                        row_id = row.get(key)
                        break
            row_id = self._convert_id_from_bytes(row_id)
            ids.append(row_id)

            distances.append(self._hybrid_row_score(row))

            # Extract metadata
            if include is None or "metadatas" in include or "metadata" in include:
                metadata = row.get("metadata") or row.get("METADATA")
                # Parse JSON string if needed
                if isinstance(metadata, str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        metadata = json.loads(metadata)
                metadatas.append(metadata or {})
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
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        embedding = json.loads(embedding)
                embeddings.append(embedding)
            else:
                embeddings.append(None)

        # Return in query-compatible format (List[List[...]])
        result = {"ids": [ids], "distances": [distances]}

        if include is None or "documents" in include or "document" in include:
            result["documents"] = [documents]

        if include is None or "metadatas" in include or "metadata" in include:
            result["metadatas"] = [metadatas]

        if include and ("embeddings" in include or "embedding" in include):
            result["embeddings"] = [embeddings]

        return result

    def _transform_search_result(self, search_result: dict[str, Any], include: list[str] | None) -> dict[str, Any]:
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
            "embeddings": embeddings,
        }

    # -------------------- Collection Info --------------------

    def _collection_count(self, collection_id: str | None, collection_name: str) -> int:
        """
        [Internal] Get the number of items in collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name

        Returns:
            Item count
        """
        logger.debug(f"Counting items in collection '{collection_name}'")
        conn = self._ensure_connection()

        # Convert collection name to table name
        if collection_id:
            table_name = CollectionNames.table_name_v2(collection_id)
        else:
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
                count = row.get("cnt", 0)
            elif isinstance(row, (tuple, list)):
                count = row[0] if len(row) > 0 else 0
            else:
                count = int(row) if row else 0

        logger.debug(f"✅ Collection '{collection_name}' has {count} items")
        return count

    # ==================== Namespace DML/DQL Methods ====================

    @staticmethod
    def _rewrite_where_for_ns(where: dict[str, Any] | None) -> dict[str, Any] | None:
        """Rewrite a WHERE clause to target namespace-scoped columns."""
        if where is None:
            return None
        rewritten = {}
        for k, v in where.items():
            if k in ("$and", "$or"):
                rewritten[k] = [BaseClient._rewrite_where_for_ns(sub) for sub in v]
            elif k == "$not":
                rewritten[k] = BaseClient._rewrite_where_for_ns(v)
            else:
                rewritten[f"metadata.{k}"] = v
        return rewritten

    @staticmethod
    def _append_namespace_filter(
        where_clause: str,
        params: list[Any],
        namespace_id: int,
        ltable_id: int,
    ) -> tuple[str, list[Any]]:
        """Append the namespace id filter to a WHERE clause."""
        ns_cond = f"namespace_id = {namespace_id} AND ltable_id = {ltable_id}"
        if not where_clause:
            return f"WHERE {ns_cond}", params
        if where_clause.strip().upper().startswith("WHERE"):
            inner = where_clause.strip()[5:].strip()
            return f"WHERE {ns_cond} AND ({inner})", params
        return f"WHERE {ns_cond} AND ({where_clause})", params

    @staticmethod
    def _validate_namespace_explicit_embeddings_if_needed(
        embeddings: list[list[float]] | None,
        *,
        explicit_embeddings: bool,
        has_vector_index: bool,
        collection_dimension: int | None,
    ) -> None:
        """Validate user-supplied embeddings match the collection VECTOR column dimension."""
        if not explicit_embeddings or not embeddings:
            return
        expected = collection_dimension if collection_dimension is not None else DEFAULT_VECTOR_DIMENSION
        _validate_namespace_explicit_embedding_dimensions(
            embeddings,
            expected_dimension=expected,
            has_vector_index=has_vector_index,
        )

    @staticmethod
    def _warn_explicit_embeddings_override_embedding_function(
        *,
        operation: str,
        explicit_embeddings: bool,
        has_documents: bool,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None,
    ) -> None:
        """Log when explicit embeddings take priority over embedding_function."""
        if explicit_embeddings and has_documents and embedding_function is not None:
            logger.warning(
                "%s: explicit embeddings provided together with documents while an "
                "embedding_function is configured; using explicit embeddings and "
                "not calling embedding_function.",
                operation,
            )

    def _count_namespace_records_by_id(
        self,
        table_name: str,
        namespace_id: int,
        ltable_id: int,
        record_id: str,
    ) -> int:
        """Count namespace records matching the given id."""
        id_expr = _NS_DATA_CONTENT_ID_EXPR
        sql = (
            f"SELECT COUNT(*) AS cnt FROM `{table_name}` "
            f"WHERE namespace_id = {int(namespace_id)} AND ltable_id = {int(ltable_id)} "
            f"AND {id_expr} = %s"
        )
        conn = self._ensure_connection()
        use_ctx = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, [record_id], use_ctx)
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            return int(row.get("cnt", 0))
        if isinstance(row, (tuple, list)):
            return int(row[0])
        return int(row)

    def _delete_namespace_records_by_id(
        self,
        table_name: str,
        namespace_id: int,
        ltable_id: int,
        record_id: str,
    ) -> None:
        """Delete namespace records matching the given id."""
        id_expr = _NS_DATA_CONTENT_ID_EXPR
        sql = (
            f"DELETE FROM `{table_name}` "
            f"WHERE namespace_id = {int(namespace_id)} AND ltable_id = {int(ltable_id)} "
            f"AND {id_expr} = %s"
        )
        conn = self._ensure_connection()
        use_ctx = self._use_context_manager_for_cursor()
        if use_ctx:
            with conn.cursor() as cursor:
                cursor.execute(sql, [record_id])
        else:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, [record_id])
            finally:
                cursor.close()

    def _reconcile_namespace_duplicate_records(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ltable_id: int,
        table_name: str,
        ids: list[str],
        documents: list[str] | None,
        metadatas: list[dict] | None,
        embeddings: list[list[float]] | None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None,
        **kwargs: Any,
    ) -> None:
        """Collapse concurrent upsert races to a single row per business id."""
        if not ids or collection_id is None:
            return
        ns_id = int(namespace_id)
        for i, record_id in enumerate(ids):
            doc_val = documents[i] if documents and i < len(documents) else None
            meta_val = metadatas[i] if metadatas and i < len(metadatas) else None
            emb_val = embeddings[i] if embeddings and i < len(embeddings) else None
            for attempt in range(120):
                duplicate_count = self._count_namespace_records_by_id(table_name, ns_id, ltable_id, record_id)
                if duplicate_count <= 1:
                    break
                self._delete_namespace_records_by_id(table_name, ns_id, ltable_id, record_id)
                if self._count_namespace_records_by_id(table_name, ns_id, ltable_id, record_id) == 0:
                    self._namespace_add(
                        collection_id=collection_id,
                        collection_name=collection_name,
                        namespace_id=namespace_id,
                        namespace_name=namespace_name,
                        ids=[record_id],
                        embeddings=[emb_val] if emb_val is not None else None,
                        metadatas=[meta_val] if meta_val is not None else None,
                        documents=[doc_val] if doc_val is not None else None,
                        embedding_function=embedding_function,
                        **kwargs,
                    )
                if attempt < 119:
                    time.sleep(0.05 * min(attempt + 1, 10))
            else:
                raise ValueError(f"Failed to reconcile duplicate namespace rows for record_id={record_id!r}")

    @namespace_kernel_error_guard
    def _namespace_add(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """Add records to a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", None)
        explicit_embeddings = embeddings is not None
        if isinstance(ids, str):
            ids = [ids]
        _validate_record_ids(ids)
        if len(ids) > _MAX_NAMESPACE_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(ids)} exceeds maximum allowed {_MAX_NAMESPACE_BATCH_SIZE} records per request."
            )
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        self._warn_explicit_embeddings_override_embedding_function(
            operation="namespace.add",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        if embeddings:
            pass
        elif documents:
            if embedding_function is not None:
                embeddings = embedding_function(documents)
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling add(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif metadatas:
            pass
        else:
            raise ValueError(
                "Neither embeddings, documents, nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. embeddings directly,\n"
                "  2. documents with embedding_function to generate embeddings, or\n"
                "  3. metadatas for metadata-only add."
            )

        num_items = len(ids)
        if documents and len(documents) != num_items:
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of ids ({num_items})")
        if metadatas and len(metadatas) != num_items:
            raise ValueError(f"Number of metadatas ({len(metadatas)}) does not match number of ids ({num_items})")
        if embeddings and len(embeddings) != num_items:
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of ids ({num_items})")

        self._validate_namespace_explicit_embeddings_if_needed(
            embeddings,
            explicit_embeddings=explicit_embeddings,
            has_vector_index=has_vector_index,
            collection_dimension=collection_dimension,
        )

        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        values_list = []
        for i in range(num_items):
            doc_val = documents[i] if documents else None
            doc_sql = f"'{escape_string(doc_val)}'" if doc_val is not None else "NULL"

            vec_val = embeddings[i] if embeddings else None
            vec_sql = "NULL" if vec_val is None else _embedding_to_hexstring(vec_val)

            meta_val = metadatas[i] if metadatas else None
            data_content = {"id": ids[i]}
            if meta_val is not None:
                data_content["metadata"] = meta_val
            dc_json = json.dumps(data_content, ensure_ascii=False)
            dc_sql = f"'{escape_string(dc_json)}'"

            values_list.append(f"({ns_id}, {ltable_id}, {doc_sql}, {vec_sql}, {dc_sql})")

        columns = (
            f"{NamespaceFieldNames.NAMESPACE_ID}, {NamespaceFieldNames.LTABLE_ID}, "
            f"{NamespaceFieldNames.DOCUMENT}, {NamespaceFieldNames.EMBEDDING}, {NamespaceFieldNames.DATA_CONTENT}"
        )
        sql = f"INSERT INTO `{table_name}` ({columns}) VALUES {','.join(values_list)}"
        self._execute(sql)

    @namespace_kernel_error_guard
    def _namespace_update(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """Update existing records in a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", None)
        explicit_embeddings = embeddings is not None
        if isinstance(ids, str):
            ids = [ids]
        _validate_record_ids(ids)
        if len(ids) > _MAX_NAMESPACE_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(ids)} exceeds maximum allowed {_MAX_NAMESPACE_BATCH_SIZE} records per request."
            )
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        self._warn_explicit_embeddings_override_embedding_function(
            operation="namespace.update",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                embeddings = embedding_function(documents)
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling update(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )

        self._validate_namespace_explicit_embeddings_if_needed(
            embeddings,
            explicit_embeddings=explicit_embeddings,
            has_vector_index=has_vector_index,
            collection_dimension=collection_dimension,
        )

        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        id_expr = _NS_DATA_CONTENT_ID_EXPR
        active_ids = []
        for i, record_id in enumerate(ids):
            has_update = (
                (documents and i < len(documents) and documents[i] is not None)
                or (embeddings and i < len(embeddings) and embeddings[i] is not None)
                or (metadatas and i < len(metadatas) and metadatas[i] is not None)
            )
            if has_update:
                active_ids.append((i, record_id))

        if not active_ids:
            return

        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()

        # VECTOR columns do not support CASE-WHEN in OceanBase, so embedding
        # updates are executed per-row while document/metadata use batch CASE-WHEN.
        emb_ids = []
        for i, record_id in active_ids:
            if embeddings and i < len(embeddings) and embeddings[i] is not None:
                emb_ids.append((i, record_id))

        if emb_ids:
            for i, record_id in emb_ids:
                vec_sql = _embedding_to_hexstring(embeddings[i])
                sql = (
                    f"UPDATE `{table_name}` SET embedding = {vec_sql} "
                    f"WHERE namespace_id = {ns_id} AND ltable_id = {ltable_id} "
                    f"AND {id_expr} = %s"
                )
                if use_context_manager:
                    with conn.cursor() as cursor:
                        cursor.execute(sql, [record_id])
                else:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(sql, [record_id])
                    finally:
                        cursor.close()

        doc_case_parts = []
        meta_case_parts = []
        params = []
        has_doc = False
        has_meta = False
        batch_ids = []

        for i, record_id in active_ids:
            if documents and i < len(documents) and documents[i] is not None:
                has_doc = True
                doc_case_parts.append(f"WHEN {id_expr} = %s THEN %s")
                params.extend([record_id, documents[i]])
                if record_id not in list(batch_ids):
                    batch_ids.append(record_id)
            if metadatas and i < len(metadatas) and metadatas[i] is not None:
                has_meta = True
                meta_json = json.dumps(metadatas[i], ensure_ascii=False)
                meta_case_parts.append(
                    f"WHEN {id_expr} = %s THEN JSON_SET(data_content, '$.metadata', CAST(%s AS JSON))"
                )
                params.extend([record_id, meta_json])
                if record_id not in batch_ids:
                    batch_ids.append(record_id)

        if has_doc or has_meta:
            set_clauses = []
            if has_doc:
                set_clauses.append(f"document = CASE {' '.join(doc_case_parts)} ELSE document END")
            if has_meta:
                set_clauses.append(f"data_content = CASE {' '.join(meta_case_parts)} ELSE data_content END")

            id_placeholders = ", ".join(["%s"] * len(batch_ids))
            params.extend(batch_ids)

            sql = (
                f"UPDATE `{table_name}` SET {', '.join(set_clauses)} "
                f"WHERE namespace_id = {ns_id} AND ltable_id = {ltable_id} "
                f"AND {id_expr} IN ({id_placeholders})"
            )
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
            else:
                cursor = conn.cursor()
                try:
                    cursor.execute(sql, params)
                finally:
                    cursor.close()

    @namespace_kernel_error_guard
    def _namespace_upsert(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """Insert or update records in a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", None)
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        if isinstance(ids, str):
            ids = [ids]
        _validate_record_ids(ids)
        if len(ids) > _MAX_NAMESPACE_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(ids)} exceeds maximum allowed {_MAX_NAMESPACE_BATCH_SIZE} records per request."
            )
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        existing_ids = set()
        id_expr = _NS_DATA_CONTENT_ID_EXPR
        id_placeholders = ", ".join(["%s"] * len(ids))
        check_sql = (
            f"SELECT JSON_EXTRACT(data_content, '$.id') AS rid FROM `{table_name}` "
            f"WHERE namespace_id = {ns_id} AND ltable_id = {ltable_id} "
            f"AND {id_expr} IN ({id_placeholders})"
        )
        conn = self._ensure_connection()
        use_ctx = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, check_sql, list(ids), use_ctx)
        for row in rows:
            rid_raw = row.get("rid") if isinstance(row, dict) else row[0]
            rid = json.loads(rid_raw) if isinstance(rid_raw, str) else rid_raw
            if rid is not None:
                existing_ids.add(str(rid))

        add_indices = []
        update_indices = []
        for i, rid in enumerate(ids):
            if rid in existing_ids:
                update_indices.append(i)
            else:
                add_indices.append(i)

        if add_indices:
            add_ids = [ids[i] for i in add_indices]
            add_docs = [documents[i] for i in add_indices] if documents else None
            add_metas = [metadatas[i] for i in add_indices] if metadatas else None
            add_embs = [embeddings[i] for i in add_indices] if embeddings else None
            self._namespace_add(
                collection_id=collection_id,
                collection_name=collection_name,
                namespace_id=namespace_id,
                namespace_name=namespace_name,
                ids=add_ids,
                embeddings=add_embs,
                metadatas=add_metas,
                documents=add_docs,
                embedding_function=embedding_function,
                has_vector_index=has_vector_index,
                collection_dimension=collection_dimension,
                **kwargs,
            )

        if update_indices:
            upd_ids = [ids[i] for i in update_indices]
            upd_docs = [documents[i] for i in update_indices] if documents else None
            upd_metas = [metadatas[i] for i in update_indices] if metadatas else None
            upd_embs = [embeddings[i] for i in update_indices] if embeddings else None
            self._namespace_update(
                collection_id=collection_id,
                collection_name=collection_name,
                namespace_id=namespace_id,
                namespace_name=namespace_name,
                ids=upd_ids,
                embeddings=upd_embs,
                metadatas=upd_metas,
                documents=upd_docs,
                embedding_function=embedding_function,
                has_vector_index=has_vector_index,
                collection_dimension=collection_dimension,
                **kwargs,
            )

        self._reconcile_namespace_duplicate_records(
            collection_id=collection_id,
            collection_name=collection_name,
            namespace_id=namespace_id,
            namespace_name=namespace_name,
            ltable_id=ltable_id,
            table_name=table_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            embedding_function=embedding_function,
            **kwargs,
        )

    @namespace_kernel_error_guard
    def _namespace_delete(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Delete records from a namespace collection."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        if ids is None and where is None and where_document is None:
            raise ValueError("At least one of ids, where, or where_document must be provided")

        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        conditions = []
        params = []

        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            _validate_record_ids(ids)
            id_placeholders = " OR ".join([f"{_NS_DATA_CONTENT_ID_EXPR} = %s"] * len(ids))
            conditions.append(f"({id_placeholders})")
            params.extend(ids)

        if where is not None:
            rewritten = self._rewrite_where_for_ns(where)
            meta_clause, meta_params = FilterBuilder.build_metadata_filter(rewritten, "data_content")
            if meta_clause:
                conditions.append(meta_clause)
                params.extend(meta_params)

        if where_document is not None:
            doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
            if doc_clause:
                conditions.append(doc_clause)
                params.extend(doc_params)

        user_where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        where_clause, params = self._append_namespace_filter(user_where, params, ns_id, ltable_id)
        sql = f"DELETE FROM `{table_name}` {where_clause}"
        if params:
            conn = self._ensure_connection()
            use_context_manager = self._use_context_manager_for_cursor()
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
            else:
                cursor = conn.cursor()
                try:
                    cursor.execute(sql, params)
                finally:
                    cursor.close()
        else:
            self._execute(sql)

    @namespace_kernel_error_guard
    def _namespace_query(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run a vector query against a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", kwargs.pop("dimension", None))
        explicit_query_embeddings = query_embeddings is not None
        embedding_function = kwargs.get("embedding_function")
        distance = kwargs.get("distance", DEFAULT_DISTANCE_METRIC)

        self._warn_explicit_embeddings_override_embedding_function(
            operation="namespace.query",
            explicit_embeddings=explicit_query_embeddings,
            has_documents=query_texts is not None,
            embedding_function=embedding_function,
        )

        if query_embeddings is not None:
            pass
        elif query_texts is not None:
            if embedding_function is not None:
                query_embeddings = self._embed_texts(query_texts, embedding_function=embedding_function)
            else:
                raise ValueError("query_texts provided but no embedding_function.")
        else:
            raise ValueError("Neither query_embeddings nor query_texts provided.")

        query_embeddings = self._normalize_query_embeddings(query_embeddings)
        self._validate_namespace_explicit_embeddings_if_needed(
            query_embeddings,
            explicit_embeddings=explicit_query_embeddings,
            has_vector_index=has_vector_index,
            collection_dimension=collection_dimension,
        )

        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        include_fields = self._normalize_include_fields(include)

        # Logical-table vector search must use hybrid_search DSL; direct SQL vector
        # index scans on namespace partitions are not supported on OceanBase.
        all_ids: list[list[Any]] = []
        all_documents: list[list[Any]] = []
        all_metadatas: list[list[Any]] = []
        all_embeddings: list[list[Any]] = []
        all_distances: list[list[float]] = []

        hybrid_kwargs = {k: v for k, v in kwargs.items() if k not in ("embedding_function", "distance", "dimension")}

        for query_vector in query_embeddings:
            knn_cfg: dict[str, Any] = {
                "query_embeddings": query_vector,
                "n_results": n_results,
            }
            if where is not None:
                knn_cfg["where"] = where

            post_filter_wd: dict[str, Any] | str | None = None
            hybrid_include = include
            if where_document is not None:
                if where_document_knn_prefilterable(where_document):
                    knn_cfg["where_document"] = where_document
                else:
                    post_filter_wd = where_document
                    knn_cfg["n_results"] = min(max(n_results * 5, n_results), _MAX_N_RESULTS)
                    if include is None:
                        hybrid_include = ["documents", "metadatas"]
                    elif not {"documents", "document"} & {*(include or [])}:
                        hybrid_include = ["documents", *include]

            batch = self._namespace_hybrid_search(
                collection_id=collection_id,
                collection_name=collection_name,
                namespace_id=namespace_id,
                namespace_name=namespace_name,
                query=None,
                knn=knn_cfg,
                n_results=knn_cfg["n_results"],
                include=hybrid_include,
                embedding_function=embedding_function,
                distance=distance,
                dimension=collection_dimension,
                **hybrid_kwargs,
            )

            if post_filter_wd is not None:
                batch = self._post_filter_namespace_query_result(
                    batch,
                    post_filter_wd,
                    n_results=n_results,
                )

            batch_ids = batch.get("ids") or [[]]
            all_ids.append(batch_ids[0] if batch_ids else [])
            all_distances.append((batch.get("distances") or [[]])[0])

            if "documents" in include_fields or "document" in include_fields or include is None:
                batch_docs = batch.get("documents") or [[]]
                all_documents.append(batch_docs[0] if batch_docs else [])
            if "metadatas" in include_fields or "metadata" in include_fields or include is None:
                batch_meta = batch.get("metadatas") or [[]]
                all_metadatas.append(batch_meta[0] if batch_meta else [])
            if "embeddings" in include_fields or "embedding" in include_fields:
                batch_emb = batch.get("embeddings") or [[]]
                all_embeddings.append(batch_emb[0] if batch_emb else [])

        result: dict[str, Any] = {"ids": all_ids, "distances": all_distances}
        if "documents" in include_fields or "document" in include_fields or include is None:
            result["documents"] = all_documents
        if "metadatas" in include_fields or "metadata" in include_fields or include is None:
            result["metadatas"] = all_metadatas
        if "embeddings" in include_fields or "embedding" in include_fields:
            result["embeddings"] = all_embeddings
        return result

    @namespace_kernel_error_guard
    def _namespace_get(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch records from a namespace collection by id/where/pagination."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        include_fields = self._normalize_include_fields(include)
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        select_parts = ["JSON_EXTRACT(data_content, '$.id') AS record_id"]
        if include_fields.get("documents") or include_fields.get("document") or include is None:
            select_parts.append("document")
        if include_fields.get("metadatas") or include_fields.get("metadata") or include is None:
            select_parts.append("JSON_EXTRACT(data_content, '$.metadata') AS metadata")
        if include_fields.get("embeddings") or include_fields.get("embedding"):
            select_parts.append("embedding")

        user_conditions = []
        params = []

        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            id_conds = []
            for rid in ids:
                id_escaped = escape_string(rid)
                id_conds.append(f"{_NS_DATA_CONTENT_ID_EXPR} = '{id_escaped}'")
            user_conditions.append(f"({' OR '.join(id_conds)})")

        if where is not None:
            rewritten = self._rewrite_where_for_ns(where)
            meta_clause, meta_params = FilterBuilder.build_metadata_filter(rewritten, "data_content")
            if meta_clause:
                user_conditions.append(meta_clause)
                params.extend(meta_params)

        if where_document is not None:
            doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
            if doc_clause:
                user_conditions.append(doc_clause)
                params.extend(doc_params)

        user_where = f"WHERE {' AND '.join(user_conditions)}" if user_conditions else ""
        where_str, params = self._append_namespace_filter(user_where, params, ns_id, ltable_id)
        select_clause = ", ".join(select_parts)
        sql = f"SELECT {select_clause} FROM `{table_name}` {where_str}"
        if limit is None and offset is not None:
            limit = 100
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
            if offset is not None:
                sql += f" OFFSET {int(offset)}"

        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, params if params else [], use_context_manager)

        result_ids = []
        result_documents = []
        result_metadatas = []
        result_embeddings = []

        for row in rows:
            if isinstance(row, dict):
                rid_raw = row.get("record_id")
                rid = json.loads(rid_raw) if isinstance(rid_raw, str) else rid_raw
                result_ids.append(rid)
                if "documents" in include_fields or "document" in include_fields or include is None:
                    result_documents.append(row.get("document"))
                if "metadatas" in include_fields or "metadata" in include_fields or include is None:
                    meta_raw = row.get("metadata")
                    if isinstance(meta_raw, str):
                        meta_raw = json.loads(meta_raw)
                    result_metadatas.append(meta_raw or {})
                if "embeddings" in include_fields or "embedding" in include_fields:
                    emb = row.get("embedding")
                    if isinstance(emb, bytes):
                        emb = self._parse_embedding_from_bytes(emb)
                    elif isinstance(emb, str):
                        emb = json.loads(emb)
                    result_embeddings.append(emb)
            elif isinstance(row, (list, tuple)):
                idx = 0
                rid_raw = row[idx]
                idx += 1
                rid = json.loads(rid_raw) if isinstance(rid_raw, str) else rid_raw
                result_ids.append(rid)
                if "documents" in include_fields or "document" in include_fields or include is None:
                    result_documents.append(row[idx])
                    idx += 1
                if "metadatas" in include_fields or "metadata" in include_fields or include is None:
                    meta_raw = row[idx]
                    idx += 1
                    if isinstance(meta_raw, str):
                        meta_raw = json.loads(meta_raw)
                    result_metadatas.append(meta_raw or {})
                if "embeddings" in include_fields or "embedding" in include_fields:
                    emb = row[idx]
                    idx += 1
                    if isinstance(emb, bytes):
                        emb = self._parse_embedding_from_bytes(emb)
                    elif isinstance(emb, str):
                        emb = json.loads(emb)
                    result_embeddings.append(emb)

        result = {"ids": result_ids}
        if "documents" in include_fields or "document" in include_fields or include is None:
            result["documents"] = result_documents
        if "metadatas" in include_fields or "metadata" in include_fields or include is None:
            result["metadatas"] = result_metadatas
        if "embeddings" in include_fields or "embedding" in include_fields:
            result["embeddings"] = result_embeddings
        return result

    @namespace_kernel_error_guard
    def _namespace_count(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        **kwargs,
    ) -> int:
        """Count records in a namespace collection."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)
        where_clause, _ = self._append_namespace_filter("", [], ns_id, ltable_id)
        sql = f"SELECT COUNT(*) AS cnt FROM `{table_name}` {where_clause}"
        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, [], use_context_manager)
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            return row.get("cnt", 0)
        elif isinstance(row, (tuple, list)):
            return row[0] if len(row) > 0 else 0
        return int(row) if row else 0

    @namespace_kernel_error_guard
    def _namespace_peek(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        limit: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Return a small sample of records from a namespace collection."""
        return self._namespace_get(
            collection_id=collection_id,
            collection_name=collection_name,
            namespace_id=namespace_id,
            namespace_name=namespace_name,
            limit=limit,
            offset=0,
            include=["documents", "metadatas", "embeddings"],
        )

    def _adapt_search_parm_for_ns(self, search_parm: dict[str, Any], ns_id: int, lt_id: int) -> dict[str, Any]:
        """Adapt search parameters to the namespace-scoped table layout."""
        ns_filter = [
            {"term": {"namespace_id": ns_id}},
            {"term": {"ltable_id": int(lt_id)}},
        ]

        def _rewrite_field_refs(obj):
            """Rewrite field references to their namespace-scoped equivalents."""
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    new_key = k
                    if k == "_id":
                        new_key = "data_content.id"
                    elif isinstance(k, str):
                        prefix = "(JSON_EXTRACT(metadata, '$."
                        suffix = "'))"
                        if k.startswith(prefix) and k.endswith(suffix) and len(k) > len(prefix) + len(suffix):
                            inner = k[len(prefix) : -len(suffix)]
                            new_key = f"data_content.metadata.{inner}"
                    new_dict[new_key] = _rewrite_field_refs(v)
                return new_dict
            elif isinstance(obj, list):
                return [_rewrite_field_refs(item) for item in obj]
            return obj

        search_parm = _rewrite_field_refs(search_parm)

        if "_source" in search_parm:
            needs_data_content = False
            new_source = []
            for field in search_parm["_source"]:
                if field in ("_id", "metadata"):
                    needs_data_content = True
                else:
                    new_source.append(field)
            if needs_data_content:
                new_source.insert(0, "data_content")
            search_parm["_source"] = new_source

        def _inject_filter_into_knn(node):
            """Inject the namespace filter into a KNN search expression."""
            if isinstance(node, dict):
                if "filter" in node:
                    existing = node["filter"]
                    if isinstance(existing, list):
                        node["filter"] = existing + ns_filter
                    else:
                        node["filter"] = [existing, *ns_filter]
                else:
                    node["filter"] = list(ns_filter)
            return node

        # Scoring (full-text) leaf queries may live in a `must` clause; scalar
        # leaf queries (term/terms/range/json/array) MUST go into `filter`. The
        # kernel rejects a scalar query inside must/should of a scoring bool with
        # `OB_NOT_SUPPORTED: scalar term query in must/should clause`, and a
        # top-level bool query is treated as scoring by default.
        _scoring_leaf_keys = {"query_string", "match", "multi_match", "match_phrase"}

        def _inject_filter_into_query(node):
            """Inject the namespace filter into a query expression."""
            if not isinstance(node, dict):
                return node
            if "bool" in node:
                bool_node = node["bool"]
                if "filter" in bool_node:
                    existing = bool_node["filter"]
                    if isinstance(existing, list):
                        bool_node["filter"] = existing + ns_filter
                    else:
                        bool_node["filter"] = [existing, *ns_filter]
                else:
                    bool_node["filter"] = list(ns_filter)
                return node
            if _scoring_leaf_keys & node.keys():
                return {"bool": {"must": [node], "filter": list(ns_filter)}}
            return {"bool": {"filter": [node, *ns_filter]}}

        if "query" in search_parm:
            q = search_parm["query"]
            if isinstance(q, list):
                search_parm["query"] = [_inject_filter_into_query(item) for item in q]
            else:
                search_parm["query"] = _inject_filter_into_query(q)

        if "knn" in search_parm:
            knn = search_parm["knn"]
            if isinstance(knn, list):
                for item in knn:
                    _inject_filter_into_knn(item)
            else:
                _inject_filter_into_knn(knn)

        if "query" not in search_parm and "knn" not in search_parm:
            search_parm["query"] = {"bool": {"filter": list(ns_filter)}}

        return search_parm

    @namespace_kernel_error_guard
    def _namespace_hybrid_search(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        query: dict[str, Any] | None = None,
        knn: dict[str, Any] | None = None,
        rank: dict[str, Any] | None = None,
        n_results: int = 10,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run a hybrid (vector + fulltext) search against a namespace collection."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        conn = self._ensure_connection()
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        search_parm = self._build_search_parm(
            query,
            knn,
            rank,
            n_results,
            include=include,
            dimension=kwargs.get("dimension"),
            **{k: v for k, v in kwargs.items() if k != "dimension"},
        )
        search_parm = self._adapt_search_parm_for_ns(search_parm, ns_id, ltable_id)

        search_parm.pop("_source", None)

        if "knn" in search_parm:
            if query_hint is None:
                query_hint = QueryHint(vector_index=True)
            elif query_hint.vector_index is None:
                query_hint = QueryHint(
                    parallel=query_hint.parallel,
                    query_timeout=query_hint.query_timeout,
                    vector_index=True,
                )

        search_parm_json = json.dumps(search_parm, ensure_ascii=False)
        use_context_manager = self._use_context_manager_for_cursor()

        escaped_params = search_parm_json.replace("'", "''")

        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name) or ""
        hybrid_sql = (
            f"SELECT {hint_sql + ' ' if hint_sql else ''}* FROM hybrid_search(TABLE `{table_name}`, '{escaped_params}')"
        )
        result_rows = self._execute_query_with_cursor(conn, hybrid_sql, [], use_context_manager)
        if not result_rows:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }
        return self._transform_ns_hybrid_result(result_rows, include)

    def _transform_ns_hybrid_result(
        self, result_rows: list[dict[str, Any]], include: list[str] | None
    ) -> dict[str, Any]:
        """Transform raw hybrid-search rows into the public result shape."""
        if not result_rows:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }

        ids = []
        distances = []
        metadatas = []
        documents = []
        embeddings = []

        for row in result_rows:
            dc_raw = row.get("data_content") or row.get("DATA_CONTENT")
            dc = {}
            if isinstance(dc_raw, str):
                with contextlib.suppress(json.JSONDecodeError):
                    dc = json.loads(dc_raw)
            elif isinstance(dc_raw, dict):
                dc = dc_raw

            row_id = dc.get("id")
            if row_id is None:
                for key in ("id", "_id", "ID"):
                    if key in row and row[key] is not None:
                        row_id = row[key]
                        break
            row_id = self._convert_id_from_bytes(row_id)
            ids.append(row_id)

            distances.append(self._hybrid_row_score(row))

            if include is None or "metadatas" in include or "metadata" in include:
                meta = dc.get("metadata")
                if meta is None:
                    meta = row.get("metadata") or row.get("METADATA")
                if isinstance(meta, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        meta = json.loads(meta)
                metadatas.append(meta or {})
            else:
                metadatas.append(None)

            if include is None or "documents" in include or "document" in include:
                documents.append(row.get("document") or row.get("DOCUMENT"))
            else:
                documents.append(None)

            if include and ("embeddings" in include or "embedding" in include):
                emb = row.get("embedding") or row.get("EMBEDDING")
                if isinstance(emb, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        emb = json.loads(emb)
                elif isinstance(emb, bytes):
                    emb = self._parse_embedding_from_bytes(emb)
                embeddings.append(emb)
            else:
                embeddings.append(None)

        result = {"ids": [ids], "distances": [distances]}
        if include is None or "documents" in include or "document" in include:
            result["documents"] = [documents]
        if include is None or "metadatas" in include or "metadata" in include:
            result["metadatas"] = [metadatas]
        if include and ("embeddings" in include or "embedding" in include):
            result["embeddings"] = [embeddings]
        return result

    def _namespace_prewarm(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        **kwargs,
    ) -> None:
        """Prewarm the namespace logical table to reduce first-query latency."""
        raise NotImplementedError("prewarm is not supported in this client mode")
