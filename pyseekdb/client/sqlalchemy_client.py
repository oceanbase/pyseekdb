from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete as sa_delete,
    func,
    insert as sa_insert,
    literal,
    not_,
    or_,
    select,
    text as sa_text,
    update as sa_update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError
from sqlalchemy.sql import ColumnElement, quoted_name
from sqlalchemy.sql.expression import cast

from .client_base import (
    BaseClient,
    ConfigurationParam,
    EmbeddingFunction,
    EmbeddingFunctionParam,
    EmbeddingDocuments,
    HNSWConfiguration,
    _NOT_PROVIDED,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_VECTOR_DIMENSION,
)
from .collection import Collection
from .database import Database
from .meta_info import CollectionFieldNames, CollectionNames

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CollectionMetadata:
    """Lightweight holder for collection configuration."""

    name: str
    dimension: int
    distance: str


class _SqlalchemyFilterBuilder:
    """Convert dictionary-based filters into SQLAlchemy expressions."""

    def __init__(self, metadata_column: ColumnElement[Any], document_column: ColumnElement[Any]) -> None:
        self._metadata_column = metadata_column
        self._document_column = document_column

    def build_metadata(self, where: Optional[dict[str, Any]]) -> Optional[ColumnElement[Any]]:
        if not where:
            return None
        return self._build_condition(where)

    def build_document(self, where_document: Optional[dict[str, Any]]) -> Optional[ColumnElement[Any]]:
        if not where_document:
            return None
        return self._build_document_condition(where_document)

    def _build_condition(self, condition: dict[str, Any]) -> ColumnElement[Any]:
        clauses: list[ColumnElement[Any]] = []

        for key, value in condition.items():
            if key == "$and":
                clauses.append(and_(*[self._build_condition(item) for item in value]))
            elif key == "$or":
                clauses.append(or_(*[self._build_condition(item) for item in value]))
            elif key == "$not":
                clauses.append(not_(self._build_condition(value)))
            else:
                clauses.append(self._build_field_clause(key, value))

        if not clauses:
            return literal(True)
        if len(clauses) == 1:
            return clauses[0]
        return and_(*clauses)

    def _build_field_clause(self, field: str, value: Any) -> ColumnElement[Any]:
        target = self._metadata_column[field].astext

        if isinstance(value, dict):
            parts: list[ColumnElement[Any]] = []
            for operator, operand in value.items():
                parts.append(self._build_operator_clause(target, operator, operand))
            if not parts:
                return literal(True)
            if len(parts) == 1:
                return parts[0]
            return and_(*parts)

        return target == value

    def _build_operator_clause(self, target: ColumnElement[Any], operator: str, operand: Any) -> ColumnElement[Any]:
        if operator == "$eq":
            return target == operand
        if operator == "$ne":
            return target != operand
        if operator in ("$lt", "$lte", "$gt", "$gte"):
            comparable = self._cast_for_numeric(target, operand)
            if operator == "$lt":
                return comparable < operand
            if operator == "$lte":
                return comparable <= operand
            if operator == "$gt":
                return comparable > operand
            return comparable >= operand
        if operator == "$in":
            values = list(operand if isinstance(operand, list) else [operand])
            return target.in_(values)
        if operator == "$nin":
            values = list(operand if isinstance(operand, list) else [operand])
            return ~target.in_(values)

        raise ValueError(f"Unsupported metadata operator '{operator}'")

    def _cast_for_numeric(self, target: ColumnElement[Any], operand: Any) -> ColumnElement[Any]:
        if isinstance(operand, (int, float)):
            return cast(target, Float)
        return target

    def _build_document_condition(self, condition: dict[str, Any]) -> ColumnElement[Any]:
        clauses: list[ColumnElement[Any]] = []

        for key, value in condition.items():
            if key == "$and":
                clauses.append(and_(*[self._build_document_condition(item) for item in value]))
            elif key == "$or":
                clauses.append(or_(*[self._build_document_condition(item) for item in value]))
            elif key == "$contains":
                clauses.append(self._document_column.ilike(f"%{value}%"))
            elif key == "$regex":
                clauses.append(self._document_column.op("REGEXP")(value))
            else:
                raise ValueError(f"Unsupported document operator '{key}'")

        if not clauses:
            return literal(True)
        if len(clauses) == 1:
            return clauses[0]
        return and_(*clauses)


class SQLAlchemyClient(BaseClient):
    """
    SQLAlchemy-based client implementation.

    This implementation targets compatibility with SQLAlchemy-exposed engines without relying
    on pgvector-specific operators. Vector similarity is performed in Python to keep behaviour
    consistent across supported databases.
    """

    _engine: Engine
    _metadata: MetaData
    _schema: str | None
    _collection_meta_table: Table
    _table_cache: dict[str, Table]

    def __init__(
        self,
        engine: Engine | str,
        *,
        schema: Optional[str] = None,
        metadata_table: str = "seekdb_collection_metadata",
        engine_kwargs: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if isinstance(engine, str):
            self._engine = create_engine(engine, **(engine_kwargs or {}))
        else:
            self._engine = engine

        self._schema = schema
        self._metadata = MetaData(schema=schema)
        self._table_cache = {}

        self._collection_meta_table = Table(
            metadata_table,
            self._metadata,
            Column("name", String(255), primary_key=True),
            Column("dimension", Integer, nullable=False),
            Column("distance", String(32), nullable=False),
            extend_existing=True,
        )

        self._metadata.create_all(self._engine, tables=[self._collection_meta_table])

    # ==================== Connection Management ====================

    def _ensure_connection(self) -> Engine:
        return self._engine

    def is_connected(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(sa_text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def _cleanup(self) -> None:
        self._engine.dispose()

    def execute(self, sql: str) -> Any:
        with self._engine.begin() as connection:
            result: Result = connection.execute(sa_text(sql))
            if result.returns_rows:
                return result.fetchall()
            return result.rowcount

    def get_raw_connection(self) -> Engine:
        return self._engine

    @property
    def mode(self) -> str:
        return "SQLAlchemyClient"

    # ==================== Collection Management ====================

    def create_collection(
        self,
        name: str,
        configuration: ConfigurationParam = _NOT_PROVIDED,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
        **kwargs: Any,
    ) -> Collection:
        embedding_function_resolved = self._resolve_embedding_function(embedding_function)
        dimension = self._resolve_dimension(configuration, embedding_function_resolved)
        distance = self._resolve_distance(configuration)

        self._create_collection_table(name)
        self._persist_collection_metadata(_CollectionMetadata(name=name, dimension=dimension, distance=distance))

        return Collection(
            client=self,
            name=name,
            dimension=dimension,
            embedding_function=embedding_function_resolved,
            distance=distance,
            **kwargs,
        )

    def get_collection(
        self,
        name: str,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
    ) -> Collection:
        metadata = self._load_collection_metadata(name)
        if metadata is None:
            raise ValueError(f"Collection '{name}' does not exist")

        try:
            self._get_table(name)
        except NoSuchTableError as exc:
            raise ValueError(f"Collection '{name}' table not found") from exc

        embedding_function_resolved = self._resolve_embedding_function(embedding_function)

        return Collection(
            client=self,
            name=name,
            dimension=int(metadata.dimension),
            embedding_function=embedding_function_resolved,
            distance=metadata.distance,
        )

    def delete_collection(self, name: str) -> None:
        table = self._get_table(name, reflect_only=True)
        table.drop(self._engine, checkfirst=True)
        self._table_cache.pop(name, None)

        with self._engine.begin() as connection:
            connection.execute(
                sa_delete(self._collection_meta_table).where(self._collection_meta_table.c.name == name)
            )

    def list_collections(self) -> list[Collection]:
        with self._engine.connect() as connection:
            rows = connection.execute(select(self._collection_meta_table)).all()

        collections: list[Collection] = []
        for row in rows:
            collections.append(
                Collection(
                    client=self,
                    name=row.name,
                    dimension=int(row.dimension),
                    distance=row.distance,
                    embedding_function=None,
                )
            )
        return collections

    def has_collection(self, name: str) -> bool:
        if self._load_collection_metadata(name) is None:
            return False
        try:
            self._get_table(name)
        except NoSuchTableError:
            return False
        return True

    # ==================== Database Management ====================

    def create_database(self, name: str, tenant: str = "public") -> None:
        with self._engine.begin() as connection:
            connection.execute(sa_text(f'CREATE SCHEMA IF NOT EXISTS "{name}"'))

    def get_database(self, name: str, tenant: str = "public") -> Database:
        with self._engine.connect() as connection:
            result = connection.execute(
                sa_text(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema"
                ),
                {"schema": name},
            ).first()
        if result is None:
            raise ValueError(f"Database (schema) '{name}' not found")
        return Database(name=name, tenant=tenant, charset="UTF8", collation="en_US.UTF-8")

    def delete_database(self, name: str, tenant: str = "public") -> None:
        with self._engine.begin() as connection:
            connection.execute(sa_text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))

    def list_databases(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tenant: str = "public",
    ) -> Sequence[Database]:
        query = "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"
        params: dict[str, Any] = {}
        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit
        if offset is not None:
            query += " OFFSET :offset"
            params["offset"] = offset

        with self._engine.connect() as connection:
            rows = connection.execute(sa_text(query), params).fetchall()

        return [Database(name=row.schema_name, tenant=tenant, charset="UTF8", collation="en_US.UTF-8") for row in rows]

    # ==================== Collection Internal Operations ====================

    def _collection_add(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: str | list[str],
        embeddings: Optional[list[float] | list[list[float]]] = None,
        metadatas: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
        documents: Optional[str | list[str]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs: Any,
    ) -> None:
        records = self._prepare_records(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            embedding_function=embedding_function,
        )
        table = self._get_table(collection_name)
        with self._engine.begin() as connection:
            connection.execute(sa_insert(table), records)

    def _collection_update(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: str | list[str],
        embeddings: Optional[list[float] | list[list[float]]] = None,
        metadatas: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
        documents: Optional[str | list[str]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs: Any,
    ) -> None:
        records = self._prepare_records(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            embedding_function=embedding_function,
            require_embeddings=False,
        )
        table = self._get_table(collection_name)

        with self._engine.begin() as connection:
            for record in records:
                update_values = {
                    key: value
                    for key, value in record.items()
                    if key in {CollectionFieldNames.DOCUMENT, CollectionFieldNames.METADATA, CollectionFieldNames.EMBEDDING}
                }
                if not update_values:
                    continue
                connection.execute(
                    sa_update(table)
                    .where(table.c._id == record[CollectionFieldNames.ID])
                    .values(**update_values)
                )

    def _collection_upsert(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: str | list[str],
        embeddings: Optional[list[float] | list[list[float]]] = None,
        metadatas: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
        documents: Optional[str | list[str]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs: Any,
    ) -> None:
        records = self._prepare_records(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            embedding_function=embedding_function,
        )
        table = self._get_table(collection_name)

        with self._engine.begin() as connection:
            for record in records:
                update_values = {
                    key: record[key]
                    for key in (CollectionFieldNames.DOCUMENT, CollectionFieldNames.METADATA, CollectionFieldNames.EMBEDDING)
                    if key in record
                }
                result = connection.execute(
                    sa_update(table)
                    .where(table.c._id == record[CollectionFieldNames.ID])
                    .values(**update_values)
                )
                if result.rowcount == 0:
                    connection.execute(sa_insert(table).values(record))

    def _collection_delete(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Optional[str | list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not ids and not where and not where_document:
            raise ValueError("At least one of ids, where, or where_document must be provided")

        table = self._get_table(collection_name)
        builder = _SqlalchemyFilterBuilder(table.c.metadata, table.c.document)

        filters: list[ColumnElement[Any]] = []
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else ids
            filters.append(table.c._id.in_(id_list))

        metadata_filter = builder.build_metadata(where)
        if metadata_filter is not None:
            filters.append(metadata_filter)

        document_filter = builder.build_document(where_document)
        if document_filter is not None:
            filters.append(document_filter)

        condition = and_(*filters) if filters else literal(True)

        with self._engine.begin() as connection:
            connection.execute(sa_delete(table).where(condition))

    def _collection_query(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query_embeddings: Optional[list[float] | list[list[float]]] = None,
        query_texts: Optional[str | list[str]] = None,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        include: Optional[list[str]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        distance: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if query_embeddings is None and query_texts is None:
            raise ValueError("Either query_embeddings or query_texts must be provided")

        if query_embeddings is None:
            query_embeddings = self._embed_texts(query_texts, embedding_function=embedding_function)

        vectors = self._normalize_query_embeddings(query_embeddings)
        include_fields = self._normalize_include_fields(include)
        metadata = self._load_collection_metadata(collection_name)
        if metadata is None:
            raise ValueError(f"Collection '{collection_name}' metadata not found")

        table = self._get_table(collection_name)
        builder = _SqlalchemyFilterBuilder(table.c.metadata, table.c.document)

        filters: list[ColumnElement[Any]] = []
        metadata_filter = builder.build_metadata(where)
        if metadata_filter is not None:
            filters.append(metadata_filter)

        document_filter = builder.build_document(where_document)
        if document_filter is not None:
            filters.append(document_filter)

        condition = and_(*filters) if filters else literal(True)

        stmt = select(
            table.c._id,
            table.c.document,
            table.c.metadata,
            table.c.embedding,
        ).where(condition)

        with self._engine.connect() as connection:
            rows = list(connection.execute(stmt))

        ids_result: list[list[str]] = []
        documents_result: list[list[str]] = []
        metadatas_result: list[list[dict[str, Any]]] = []
        embeddings_result: list[list[list[float]]] = []
        distances_result: list[list[float]] = []

        metric = distance or metadata.distance

        for vector in vectors:
            normalized_vector = self._normalize_vector(vector)
            scored: list[tuple[float, Any, list[float]]] = []
            for row in rows:
                candidate = self._normalize_embedding_data(row.embedding)
                if candidate is None:
                    continue
                if len(candidate) != len(normalized_vector):
                    raise ValueError("Embedding dimension mismatch during query")
                distance_value = self._compute_distance(metric, normalized_vector, candidate)
                scored.append((distance_value, row, candidate))

            scored.sort(key=lambda item: item[0])
            top_items = scored[:n_results]

            ids_row: list[str] = []
            documents_row: list[str] = []
            metadatas_row: list[dict[str, Any]] = []
            embeddings_row: list[list[float]] = []
            distances_row: list[float] = []

            for distance_value, row, candidate in top_items:
                ids_row.append(row._id)
                if include_fields.get("documents", False) and row.document is not None:
                    documents_row.append(row.document)
                if include_fields.get("metadatas", False) and row.metadata is not None:
                    metadatas_row.append(row.metadata)
                if include_fields.get("embeddings", False):
                    embeddings_row.append(candidate)
                distances_row.append(distance_value)

            ids_result.append(ids_row)
            if include_fields.get("documents", False):
                documents_result.append(documents_row)
            if include_fields.get("metadatas", False):
                metadatas_result.append(metadatas_row)
            if include_fields.get("embeddings", False):
                embeddings_result.append(embeddings_row)
            distances_result.append(distances_row)

        result: dict[str, Any] = {"ids": ids_result, "distances": distances_result}
        if include_fields.get("documents", False):
            result["documents"] = documents_result
        if include_fields.get("metadatas", False):
            result["metadatas"] = metadatas_result
        if include_fields.get("embeddings", False):
            result["embeddings"] = embeddings_result

        return result

    def _collection_get(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Optional[str | list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        table = self._get_table(collection_name)
        builder = _SqlalchemyFilterBuilder(table.c.metadata, table.c.document)
        include_fields = self._normalize_include_fields(include)

        filters: list[ColumnElement[Any]] = []
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else ids
            filters.append(table.c._id.in_(id_list))

        metadata_filter = builder.build_metadata(where)
        if metadata_filter is not None:
            filters.append(metadata_filter)

        document_filter = builder.build_document(where_document)
        if document_filter is not None:
            filters.append(document_filter)

        condition = and_(*filters) if filters else literal(True)

        stmt = select(
            table.c._id,
            table.c.document,
            table.c.metadata,
            table.c.embedding,
        ).where(condition)

        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)

        with self._engine.connect() as connection:
            rows = connection.execute(stmt).all()

        ids_result: list[str] = []
        documents_result: list[str] = []
        metadatas_result: list[dict[str, Any]] = []
        embeddings_result: list[list[float]] = []

        for row in rows:
            ids_result.append(row._id)
            if include_fields.get("documents", False) and row.document is not None:
                documents_result.append(row.document)
            if include_fields.get("metadatas", False) and row.metadata is not None:
                metadatas_result.append(row.metadata)
            if include_fields.get("embeddings", False):
                candidate = self._normalize_embedding_data(row.embedding)
                if candidate is not None:
                    embeddings_result.append(candidate)

        result: dict[str, Any] = {"ids": ids_result}
        if include_fields.get("documents", False):
            result["documents"] = documents_result
        if include_fields.get("metadatas", False):
            result["metadatas"] = metadatas_result
        if include_fields.get("embeddings", False):
            result["embeddings"] = embeddings_result
        return result

    def _collection_count(
        self,
        collection_id: Optional[str],
        collection_name: str,
        **kwargs: Any,
    ) -> int:
        table = self._get_table(collection_name)
        with self._engine.connect() as connection:
            result = connection.execute(select(func.count()).select_from(table)).scalar()
        return int(result or 0)

    def _collection_hybrid_search(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query: Optional[dict[str, Any]] = None,
        knn: Optional[dict[str, Any]] = None,
        rank: Optional[dict[str, Any]] = None,
        n_results: int = 10,
        include: Optional[list[str]] = None,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError("Hybrid search is not supported by SQLAlchemyClient")

    # ==================== Helper Methods ====================

    def _create_collection_table(self, name: str) -> None:
        table_name = quoted_name(CollectionNames.table_name(name), quote=True)
        table = Table(
            table_name,
            self._metadata,
            Column(CollectionFieldNames.ID, String(512), primary_key=True),
            Column(CollectionFieldNames.DOCUMENT, Text, nullable=True),
            Column(
                CollectionFieldNames.EMBEDDING,
                JSONB().with_variant(JSON(), "sqlite"),
                nullable=True,
            ),
            Column(
                CollectionFieldNames.METADATA,
                JSONB().with_variant(JSON(), "sqlite"),
                nullable=True,
            ),
            schema=self._schema,
        )
        table.create(self._engine, checkfirst=True)
        self._table_cache[name] = table

    def _persist_collection_metadata(self, metadata: _CollectionMetadata) -> None:
        with self._engine.begin() as connection:
            result = connection.execute(
                sa_update(self._collection_meta_table)
                .where(self._collection_meta_table.c.name == metadata.name)
                .values(dimension=metadata.dimension, distance=metadata.distance)
            )
            if result.rowcount == 0:
                connection.execute(
                    sa_insert(self._collection_meta_table).values(
                        name=metadata.name,
                        dimension=metadata.dimension,
                        distance=metadata.distance,
                    )
                )

    def _load_collection_metadata(self, name: str) -> Optional[_CollectionMetadata]:
        with self._engine.connect() as connection:
            row = connection.execute(
                select(self._collection_meta_table).where(self._collection_meta_table.c.name == name)
            ).first()
        if row is None:
            return None
        return _CollectionMetadata(name=row.name, dimension=row.dimension, distance=row.distance)

    def _resolve_embedding_function(
        self,
        embedding_function: EmbeddingFunctionParam,
    ) -> Optional[EmbeddingFunction[EmbeddingDocuments]]:
        if embedding_function is _NOT_PROVIDED:
            return self._get_default_embedding_function()
        return embedding_function

    def _resolve_dimension(
        self,
        configuration: ConfigurationParam,
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]],
    ) -> int:
        if configuration is _NOT_PROVIDED:
            if embedding_function is not None and hasattr(embedding_function, "dimension"):
                return int(embedding_function.dimension)
            return DEFAULT_VECTOR_DIMENSION
        if configuration is None:
            if embedding_function is None:
                raise ValueError("Cannot determine dimension without configuration or embedding function")
            return self._calculate_embedding_dimension(embedding_function)
        if not isinstance(configuration, HNSWConfiguration):
            raise TypeError(f"configuration must be HNSWConfiguration, got {type(configuration)}")
        if embedding_function is not None and hasattr(embedding_function, "dimension"):
            actual = int(embedding_function.dimension)
            if configuration.dimension != actual:
                raise ValueError(
                    f"Configuration dimension ({configuration.dimension}) does not match embedding function dimension ({actual})"
                )
        return configuration.dimension

    def _resolve_distance(self, configuration: ConfigurationParam) -> str:
        if isinstance(configuration, HNSWConfiguration):
            return configuration.distance
        return DEFAULT_DISTANCE_METRIC

    def _calculate_embedding_dimension(self, embedding_function: EmbeddingFunction[EmbeddingDocuments]) -> int:
        generated = embedding_function.__call__("seekdb")
        if not generated or not generated[0]:
            raise ValueError("Embedding function returned empty result when called with 'seekdb'")
        return len(generated[0])

    def _get_default_embedding_function(self) -> Optional[EmbeddingFunction[EmbeddingDocuments]]:
        from .embedding_function import get_default_embedding_function

        return get_default_embedding_function()

    def _get_table(self, name: str, *, reflect_only: bool = False) -> Table:
        if name in self._table_cache and not reflect_only:
            return self._table_cache[name]
        table_name = quoted_name(CollectionNames.table_name(name), quote=True)
        table = Table(table_name, self._metadata, schema=self._schema, autoload_with=self._engine)
        if not reflect_only:
            self._table_cache[name] = table
        return table

    def _prepare_records(
        self,
        *,
        ids: str | list[str],
        embeddings: Optional[list[float] | list[list[float]]],
        metadatas: Optional[dict[str, Any] | list[dict[str, Any]]],
        documents: Optional[str | list[str]],
        embedding_function: Optional[EmbeddingFunction[EmbeddingDocuments]],
        require_embeddings: bool = True,
    ) -> list[dict[str, Any]]:
        id_list = [ids] if isinstance(ids, str) else list(ids)
        doc_list: Optional[list[Optional[str]]] = None
        if documents is not None:
            doc_list = [documents] if isinstance(documents, str) else list(documents)

        metadata_list: Optional[list[Optional[dict[str, Any]]]] = None
        if metadatas is not None:
            if isinstance(metadatas, dict):
                metadata_list = [metadatas]
            else:
                metadata_list = list(metadatas)

        embedding_list: Optional[list[list[float]]] = None
        if embeddings is not None:
            if embeddings and isinstance(embeddings[0], (int, float)):
                embedding_list = [self._normalize_vector(embeddings)]  # type: ignore[arg-type]
            else:
                embedding_list = [self._normalize_vector(item) for item in embeddings]  # type: ignore[arg-type]
        elif doc_list is not None and embedding_function is not None:
            generated = embedding_function(doc_list)
            embedding_list = [self._normalize_vector(item) for item in generated]
        elif require_embeddings:
            raise ValueError("Embeddings or documents must be provided for this operation")

        num_items = len(id_list)
        if doc_list is not None and len(doc_list) != num_items:
            raise ValueError("documents length does not match ids length")
        if metadata_list is not None and len(metadata_list) != num_items:
            raise ValueError("metadatas length does not match ids length")
        if embedding_list is not None and len(embedding_list) != num_items:
            raise ValueError("embeddings length does not match ids length")

        records: list[dict[str, Any]] = []
        for index in range(num_items):
            record: dict[str, Any] = {CollectionFieldNames.ID: id_list[index]}
            if doc_list is not None:
                record[CollectionFieldNames.DOCUMENT] = doc_list[index]
            if metadata_list is not None:
                metadata_value = metadata_list[index]
                if metadata_value is not None and not isinstance(metadata_value, dict):
                    raise ValueError("metadata entries must be dictionaries")
                record[CollectionFieldNames.METADATA] = metadata_value
            if embedding_list is not None:
                record[CollectionFieldNames.EMBEDDING] = embedding_list[index]
            records.append(record)
        return records

    def _normalize_embedding_data(self, value: Any) -> Optional[list[float]]:
        if value is None:
            return None
        if isinstance(value, list):
            return [float(item) for item in value]
        if isinstance(value, tuple):
            return [float(item) for item in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [float(item) for item in parsed]
            except json.JSONDecodeError:
                return None
        return None

    def _normalize_vector(self, vector: Sequence[float]) -> list[float]:
        return [float(item) for item in vector]

    def _compute_distance(self, metric: str, query: Sequence[float], candidate: Sequence[float]) -> float:
        if metric == "l2":
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(query, candidate)))
        if metric == "cosine":
            dot_product = sum(a * b for a, b in zip(query, candidate))
            query_norm = math.sqrt(sum(a * a for a in query))
            candidate_norm = math.sqrt(sum(b * b for b in candidate))
            if query_norm == 0 or candidate_norm == 0:
                return 1.0
            cosine_similarity = dot_product / (query_norm * candidate_norm)
            return 1.0 - cosine_similarity
        if metric == "inner_product":
            dot_product = sum(a * b for a, b in zip(query, candidate))
            return -dot_product
        raise ValueError(f"Unsupported distance metric '{metric}'")

