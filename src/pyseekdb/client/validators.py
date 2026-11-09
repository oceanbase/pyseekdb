"""Input validators for namespace names and record identifiers."""

import re

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_MAX_NAME_LENGTH = 512
_MAX_NAMESPACE_NAME_LENGTH = 256
_MAX_NAMESPACE_BATCH_SIZE = 100
_MAX_N_RESULTS = 16384  # OceanBase vector-search k upper bound
_VALID_INCLUDE_FIELDS = frozenset({
    "documents",
    "document",
    "metadatas",
    "metadata",
    "embeddings",
    "embedding",
    "distances",
    "distance",
})


def _validate_include(include: list[str] | None) -> None:
    """Validate ``include`` is a list of supported result field names."""
    if include is None:
        return
    if not isinstance(include, list):
        raise TypeError(f"include must be a list[str] or None, got {type(include).__name__}")
    invalid = [field for field in include if field not in _VALID_INCLUDE_FIELDS]
    if invalid:
        allowed = "documents, metadatas, embeddings, distances"
        raise ValueError(
            f"Invalid include field(s): {invalid!r}. "
            f"Allowed values: {allowed} "
            f"(singular forms document, metadata, embedding, distance are also accepted)."
        )


def _validate_namespace_name(name: str) -> None:
    """Validate a namespace name for type, length, and allowed characters."""
    if not isinstance(name, str):
        raise TypeError(f"Invalid namespace name: '{name}'. Namespace name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError(f"Invalid namespace name: '{name}'. Namespace name must not be empty")
    if len(name) > _MAX_NAMESPACE_NAME_LENGTH:
        raise ValueError(
            f"Invalid namespace name: '{name}'. Namespace name too long: {len(name)} characters; maximum allowed is {_MAX_NAMESPACE_NAME_LENGTH}."
        )
    if _NAME_PATTERN.match(name) is None:
        raise ValueError(
            f"Invalid namespace name: '{name}'. Namespace name contains invalid characters. "
            "Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]"
        )


def _validate_database_name(name: str) -> None:
    """Validate a SQL database identifier for catalog table qualification."""
    if not isinstance(name, str):
        raise TypeError(f"Invalid database name: '{name}'. Database name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError(f"Invalid database name: '{name}'. Database name must not be empty")
    if _NAME_PATTERN.match(name) is None:
        raise ValueError(
            f"Invalid database name: '{name}'. Database name contains invalid characters. "
            "Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]"
        )


def _quote_sql_identifier(identifier: str) -> str:
    """Quote a SQL identifier and escape embedded backticks."""
    return "`" + identifier.replace("`", "``") + "`"


def _validate_n_results(n_results: int, *, max_results: int = _MAX_N_RESULTS) -> None:
    """Validate ``n_results`` is a positive integer within the engine limit."""
    if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results < 1:
        raise ValueError(f"n_results must be an integer >= 1, got {n_results!r}")
    if n_results > max_results:
        raise ValueError(
            f"n_results must be <= {max_results}, got {n_results}. Use a smaller value or paginate with offset/limit."
        )


def _validate_record_ids(ids: list[str]) -> None:
    """Validate namespace/collection record id list shape and per-id constraints."""
    if not isinstance(ids, list):
        raise TypeError(f"Invalid record ids: expected list[str], got {type(ids).__name__}")
    for rid in ids:
        if not isinstance(rid, str):
            raise TypeError(f"Invalid record id: '{rid}'. Record id must be a string, got {type(rid).__name__}")
        if not rid:
            raise ValueError("Invalid record id: Record id must not be empty")
        if len(rid) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"Invalid record id: '{rid[:50]}...'. Record id too long: {len(rid)} characters; maximum allowed is {_MAX_NAME_LENGTH}."
            )
        if _NAME_PATTERN.match(rid) is None:
            raise ValueError(
                f"Invalid record id: '{rid}'. Record id contains invalid characters. "
                "Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]"
            )


def _validate_namespace_explicit_embedding_dimensions(
    embeddings: list[list[float]],
    *,
    expected_dimension: int,
    has_vector_index: bool,
) -> None:
    """Reject explicit embeddings whose dimension does not match the collection VECTOR column."""
    for i, vec in enumerate(embeddings):
        if vec is None:
            continue
        actual = len(vec)
        if actual != expected_dimension:
            if has_vector_index:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dimension}, got {actual} at index {i}."
                )
            raise ValueError(
                f"Collections without a vector index store embeddings as "
                f"VECTOR({expected_dimension}); explicit embeddings must be "
                f"{expected_dimension}-dimensional, got {actual} at index {i}."
            )


def _validate_namespace_no_index_explicit_embeddings(
    embeddings: list[list[float]],
    *,
    expected_dimension: int,
) -> None:
    """Reject explicit embeddings whose dimension does not match a no-index VECTOR column."""
    _validate_namespace_explicit_embedding_dimensions(
        embeddings,
        expected_dimension=expected_dimension,
        has_vector_index=False,
    )
