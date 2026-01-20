"""
Hybrid search builder with a fluent interface.

Provides a user-friendly way to construct hybrid_search parameters by chaining
query/knn/rank/select/limit methods. Supports basic DSL helpers:
- DOCUMENT.contains()/not_contains() for document filters
- K("field") comparison operators for metadata filters
- TEXT(...) for knn query_texts
- EMBEDDINGS(...) for knn query_embeddings (callable and usable in select)

The builder keeps payloads close to the existing hybrid_search() dict schema
so the server-side SQL generation remains unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

__all__ = [
    "DOCUMENT",
    "DOCUMENTS",
    "EMBEDDINGS",
    "EMBEDDINGS_FIELD",
    "IDS",
    "METADATAS",
    "SCORES",
    "TEXT",
    "HybridSearch",
    "K",
]


class DocumentExpression:
    """Wrapper for document filter expressions."""

    def __init__(self, expression: dict[str, Any]):
        self._expression = expression

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._expression)

    def __and__(self, other: Any) -> DocumentExpression:
        other_expr = _as_document_expression(other)
        return DocumentExpression({"$and": [self.to_dict(), other_expr.to_dict()]})

    def __or__(self, other: Any) -> DocumentExpression:
        other_expr = _as_document_expression(other)
        return DocumentExpression({"$or": [self.to_dict(), other_expr.to_dict()]})


def _as_document_expression(value: Any) -> DocumentExpression:
    if isinstance(value, DocumentExpression):
        return value
    if isinstance(value, dict):
        return DocumentExpression(value)
    if isinstance(value, str):
        return DocumentExpression({"$contains": value})
    raise TypeError(f"Unsupported document expression type: {type(value)}")


class _DocumentBuilder:
    """Entry point for building document filter expressions."""

    def contains(self, text: str | list[str]) -> DocumentExpression:
        if isinstance(text, list):
            if len(text) == 1:
                return DocumentExpression({"$contains": text[0]})
            return DocumentExpression({"$and": [{"$contains": t} for t in text]})
        return DocumentExpression({"$contains": text})

    def not_contains(self, text: str) -> DocumentExpression:
        return DocumentExpression({"$not_contains": text})


DOCUMENT = _DocumentBuilder()


class MetadataExpression:
    """Wrapper for metadata filter expressions."""

    def __init__(self, expression: dict[str, Any]):
        self._expression = expression

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._expression)

    def __and__(self, other: Any) -> MetadataExpression:
        other_expr = _as_metadata_expression(other)
        return MetadataExpression({"$and": [self.to_dict(), other_expr.to_dict()]})

    def __or__(self, other: Any) -> MetadataExpression:
        other_expr = _as_metadata_expression(other)
        return MetadataExpression({"$or": [self.to_dict(), other_expr.to_dict()]})

    def __invert__(self) -> MetadataExpression:
        return MetadataExpression({"$not": self.to_dict()})


def _as_metadata_expression(value: Any) -> MetadataExpression:
    if isinstance(value, MetadataExpression):
        return value
    if isinstance(value, dict):
        return MetadataExpression(value)
    raise TypeError(f"Unsupported metadata expression type: {type(value)}")


class MetadataField:
    """Metadata field helper supporting comparison operators."""

    def __init__(self, key: str):
        self._key = key

    def _wrap(self, op: str, value: Any) -> MetadataExpression:
        if op == "$eq":
            return MetadataExpression({self._key: value})
        return MetadataExpression({self._key: {op: value}})

    def __eq__(self, other: Any) -> MetadataExpression:  # type: ignore[override]
        return self._wrap("$eq", other)

    def __ne__(self, other: Any) -> MetadataExpression:  # type: ignore[override]
        return self._wrap("$ne", other)

    def __lt__(self, other: Any) -> MetadataExpression:
        return self._wrap("$lt", other)

    def __le__(self, other: Any) -> MetadataExpression:
        return self._wrap("$lte", other)

    def __gt__(self, other: Any) -> MetadataExpression:
        return self._wrap("$gt", other)

    def __ge__(self, other: Any) -> MetadataExpression:
        return self._wrap("$gte", other)

    def is_in(self, values: Iterable[Any]) -> MetadataExpression:
        """Membership check: field in given values."""
        return self._wrap("$in", list(values))

    def not_in(self, values: Iterable[Any]) -> MetadataExpression:
        """Negated membership check: field not in given values."""
        return self._wrap("$nin", list(values))


def K(key: str) -> MetadataField:
    """Metadata field alias."""
    return MetadataField(key)


class TextQuery:
    """Container for knn query_texts."""

    def __init__(self, texts: str | list[str]):
        if isinstance(texts, str):
            self.texts = [texts]
        else:
            self.texts = list(texts)


class EmbeddingsQuery:
    """Container for knn query_embeddings."""

    def __init__(self, embeddings: list[float] | list[list[float]]):
        if embeddings is None:
            raise ValueError("query_embeddings cannot be None")
        vectors: list[list[float]] = []
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            vectors = [list(vec) for vec in embeddings]  # type: ignore[arg-type]
        elif isinstance(embeddings, list):
            vectors = [list(embeddings)]  # type: ignore[arg-type]
        else:
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")
        if not vectors:
            raise ValueError("query_embeddings must not be empty")
        self.vectors = vectors


class _TextBuilder:
    def __call__(self, texts: str | list[str]) -> TextQuery:
        return TextQuery(texts)


class _EmbeddingsBuilder:
    field_name = "embeddings"

    def __call__(self, embeddings: list[float] | list[list[float]]) -> EmbeddingsQuery:
        return EmbeddingsQuery(embeddings)


TEXT = _TextBuilder()
EMBEDDINGS = _EmbeddingsBuilder()

# Select field constants
IDS = "ids"
DOCUMENTS = "documents"
METADATAS = "metadatas"
EMBEDDINGS_FIELD = "embeddings"
SCORES = "scores"


def _pop_n_results(kwargs: dict[str, Any]) -> int | None:
    for key in ("n_results", "n_result", "nresult"):
        if key in kwargs:
            return kwargs.pop(key)
    return None


def _combine_filters(base: dict[str, Any] | None, extras: list[dict[str, Any]]) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if base:
        filters.append(copy.deepcopy(base))
    filters.extend(copy.deepcopy(item) for item in extras if item)
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


class HybridSearch:
    """
    Fluent builder for collection.hybrid_search().

    Example:
        search = (
            HybridSearch()
            .query(DOCUMENT.contains("machine learning"), K("category") == "AI", n_results=10, boost=0.5)
            .knn(TEXT("AI research"), K("year") > 2020, n_results=10, boost=0.6)
            .limit(5)
            .select(IDS, DOCUMENTS, METADATAS, EMBEDDINGS, SCORES)
        )
    """

    def __init__(
        self,
        query: dict[str, Any] | list[dict[str, Any]] | None = None,
        knn: dict[str, Any] | list[dict[str, Any]] | None = None,
        rank: dict[str, Any] | None = None,
        n_results: int | None = None,
        nresult: int | None = None,
        include: list[str] | None = None,
    ):
        self._queries: list[dict[str, Any]] = []
        self._knns: list[dict[str, Any]] = []
        self._rank = copy.deepcopy(rank) if rank else None
        self._n_results = n_results if n_results is not None else nresult
        self._include = copy.deepcopy(include) if include is not None else None

        if query:
            self._append_query_payload(query)
        if knn:
            self._append_knn_payload(knn)

    def _append_query_payload(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        if isinstance(payload, list):
            self._queries.extend(copy.deepcopy(payload))
        else:
            self._queries.append(copy.deepcopy(payload))

    def _append_knn_payload(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        if isinstance(payload, list):
            self._knns.extend(copy.deepcopy(payload))
        else:
            self._knns.append(copy.deepcopy(payload))

    def query(self, *filters: Any, **kwargs) -> HybridSearch:  # noqa: C901
        n_results = _pop_n_results(kwargs)
        boost = kwargs.pop("boost", None)
        where = kwargs.pop("where", None)
        where_document = kwargs.pop("where_document", None)

        doc_filters: list[dict[str, Any]] = []
        meta_filters: list[dict[str, Any]] = []
        for item in filters:
            if isinstance(item, DocumentExpression):
                doc_filters.append(item.to_dict())
            elif isinstance(item, MetadataExpression):
                meta_filters.append(item.to_dict())
            elif isinstance(item, dict):
                # Heuristic: document filter keys contain $contains/$not_contains/$and/$or
                if any(key in item for key in ("$contains", "$not_contains", "$and", "$or")):
                    doc_filters.append(copy.deepcopy(item))
                else:
                    meta_filters.append(copy.deepcopy(item))
            elif isinstance(item, str):
                doc_filters.append(DocumentExpression({"$contains": item}).to_dict())

        final_where_document = _combine_filters(where_document, doc_filters)
        final_where = _combine_filters(where, meta_filters)

        query_payload: dict[str, Any] = {}
        if final_where_document is not None:
            query_payload["where_document"] = final_where_document
        if final_where is not None:
            query_payload["where"] = final_where
        if n_results is not None:
            query_payload["n_results"] = n_results
        if boost is not None:
            query_payload["boost"] = boost

        if query_payload:
            self._queries.append(query_payload)
        return self

    def knn(self, *filters: Any, **kwargs) -> HybridSearch:  # noqa: C901
        n_results = _pop_n_results(kwargs)
        boost = kwargs.pop("boost", None)
        where = kwargs.pop("where", None)
        query_texts = kwargs.pop("query_texts", None)
        query_embeddings = kwargs.pop("query_embeddings", None)

        text_arg: TextQuery | None = None
        emb_arg: EmbeddingsQuery | None = None
        meta_filters: list[dict[str, Any]] = []

        if len(filters) == 1 and isinstance(filters[0], dict) and not any([query_texts, query_embeddings, where]):
            # Direct knn payload
            knn_payload = copy.deepcopy(filters[0])
            if n_results is not None:
                knn_payload["n_results"] = n_results
            if boost is not None:
                knn_payload["boost"] = boost
            self._knns.append(knn_payload)
            return self

        for item in filters:
            if isinstance(item, TextQuery):
                text_arg = item
            elif isinstance(item, EmbeddingsQuery):
                emb_arg = item
            elif isinstance(item, MetadataExpression):
                meta_filters.append(item.to_dict())
            elif isinstance(item, dict):
                meta_filters.append(copy.deepcopy(item))
            elif isinstance(item, str) and text_arg is None and query_texts is None:
                text_arg = TextQuery(item)

        if emb_arg is not None:
            query_embeddings = emb_arg.vectors
        if text_arg is not None and query_embeddings is None:
            query_texts = text_arg.texts

        final_where = _combine_filters(where, meta_filters)

        knn_payload: dict[str, Any] = {}
        if query_texts is not None:
            knn_payload["query_texts"] = query_texts
        if query_embeddings is not None:
            knn_payload["query_embeddings"] = query_embeddings
        if final_where is not None:
            knn_payload["where"] = final_where
        if n_results is not None:
            knn_payload["n_results"] = n_results
        if boost is not None:
            knn_payload["boost"] = boost

        if not knn_payload.get("query_texts") and not knn_payload.get("query_embeddings"):
            raise ValueError("knn requires either query_texts or query_embeddings")

        self._knns.append(knn_payload)
        return self

    def rank(self, rank: str | dict[str, Any] | None = None, **kwargs) -> HybridSearch:
        """
        Configure the ranking strategy.

        Supported usages:
            rank()                               -> defaults to rrf with no params
            rank("rrf")                          -> explicit rrf
            rank("rrf", rank_window_size=60)     -> rrf with parameters
            rank({"rrf": {"rank_window_size": 60}}) -> legacy dict style
        """

        def _validate_and_build(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if method != "rrf":
                raise ValueError("Only 'rrf' rank method is supported")
            if not isinstance(params, dict):
                raise TypeError("Rank parameters must be provided as a dict")
            allowed_keys = {"rank_window_size", "rank_constant"}
            unsupported = set(params) - allowed_keys
            if unsupported:
                raise ValueError(f"Unsupported parameters for rrf rank: {sorted(unsupported)}")
            # Drop None values to avoid sending unset params
            return {k: copy.deepcopy(v) for k, v in params.items() if v is not None}

        if isinstance(rank, dict):
            if kwargs:
                raise ValueError("Do not mix dict rank input with keyword parameters")
            if len(rank) != 1:
                raise ValueError("Rank dict must contain exactly one rank method")
            method, params = next(iter(rank.items()))
            params = params or {}
            validated = _validate_and_build(method, params)
            self._rank = {method: validated}
            return self

        method = "rrf" if rank is None else rank
        if not isinstance(method, str):
            raise TypeError("Rank must be provided as a string, dict, or None")

        validated = _validate_and_build(method, kwargs)
        self._rank = {method: validated}
        return self

    def limit(self, n_results: int) -> HybridSearch:
        self._n_results = n_results
        return self

    def select(self, *fields: Any) -> HybridSearch:
        include: list[str] = []
        for field in fields:
            if field is None:
                continue
            if field is EMBEDDINGS or isinstance(field, _EmbeddingsBuilder):
                if EMBEDDINGS_FIELD not in include:
                    include.append(EMBEDDINGS_FIELD)
                continue
            if isinstance(field, str):
                name = field.lower()
                if name in ("documents", "metadatas", "embeddings") and name not in include:
                    include.append(name)
        self._include = include
        return self

    def to_params(self, dimension: int | None = None) -> dict[str, Any]:
        """Return a dict containing query/knn/rank/include/n_results for hybrid_search."""
        query_param: dict[str, Any] | list[dict[str, Any]] | None = None
        if self._queries:
            query_param = self._queries[0] if len(self._queries) == 1 else copy.deepcopy(self._queries)

        knn_param: dict[str, Any] | list[dict[str, Any]] | None = None
        if self._knns:
            knn_param = self._knns[0] if len(self._knns) == 1 else copy.deepcopy(self._knns)

        self._validate_knn_embeddings(knn_param, dimension)

        return {
            "query": query_param,
            "knn": knn_param,
            "rank": copy.deepcopy(self._rank) if self._rank else None,
            "n_results": self._n_results,
            "include": copy.deepcopy(self._include) if self._include is not None else None,
        }

    @staticmethod
    def _validate_knn_embeddings(
        knn_param: dict[str, Any] | list[dict[str, Any]] | None,
        dimension: int | None,
    ) -> None:
        if dimension is None or not knn_param:
            return

        def _validate_vector(vec: list[Any]) -> None:
            if len(vec) != dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {len(vec)}")

        def _extract_vectors(payload: dict[str, Any]) -> list[list[Any]]:
            embeddings = payload.get("query_embeddings")
            if embeddings is None:
                return []
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return embeddings  # type: ignore[return-value]
            if isinstance(embeddings, list):
                return [embeddings]  # type: ignore[list-item]
            return []

        payloads = knn_param if isinstance(knn_param, list) else [knn_param]
        for payload in payloads:
            for vec in _extract_vectors(payload):
                _validate_vector(vec)
