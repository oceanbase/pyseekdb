import warnings
from dataclasses import dataclass
from enum import Enum
from warnings import deprecated

from pyseekdb.client.embedding_function import DefaultEmbeddingFunction, EmbeddingFunction
from pyseekdb.client.sparse_embedding_function import SparseEmbeddingFunction
from pyseekdb.client.types import K

# Default configuration constants
# Note: Default embedding function (DefaultEmbeddingFunction) produces 384-dim embeddings
# So we use 384 as the default dimension to match
DEFAULT_VECTOR_DIMENSION = 384  # Matches DefaultEmbeddingFunction dimension
DEFAULT_DISTANCE_METRIC = "cosine"


class DistanceMetric(str, Enum):
    """
    Distance metric constants for vector similarity calculation.

    Values can be used as strings (e.g., DistanceMetric.L2 == 'l2').
    """

    L2 = "l2"
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"


@dataclass
class FulltextIndexConfig:
    """
    Fulltext analyzer configuration for fulltext indexing.

    Args:
        analyzer: Analyzer name, can be 'space', 'ngram', 'ngram2', 'beng', 'ik' and so on (default: 'ik')
        properties: Optional dictionary of parser-specific parameters (key: string, value: primitive type)
    """

    analyzer: str = "ik"
    properties: dict[str, str | int | float | bool] | None = None


@dataclass
class HNSWConfiguration:
    """
    HNSW (Hierarchical Navigable Small World) index configuration

    Args:
        dimension: Vector dimension (number of elements in each vector)
        distance: Distance metric for similarity calculation (e.g., 'l2', 'cosine', 'inner_product')
        properties: Optional dictionary of properties for the HNSW index (key: string, value: primitive type)
        Please refer to [HNSW configuration](https://en.oceanbase.com/docs/common-oceanbase-database-10000000003351043) for detailed information.
    """

    dimension: int
    distance: str = DistanceMetric.L2.value
    properties: dict[str, str | int | float | bool] | None = None

    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")
        valid_distances = [e.value for e in DistanceMetric]
        if self.distance not in valid_distances:
            raise ValueError(f"distance must be one of {valid_distances}, got {self.distance}")
        if self.properties:
            for value in self.properties.values():
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(f"properties must be a dictionary of string, int, float, or bool, got {value}")
            distance_keys = [key for key in self.properties if key.lower() == "distance"]
            for key in distance_keys:
                warnings.warn(f"{key} is a reserved keyword in properties, it will be ignored", stacklevel=2)
                self.properties.pop(key)


@dataclass
class VectorIndexConfig:
    hnsw: HNSWConfiguration | None = None
    embedding_function: EmbeddingFunction | None = None

    def __post_init__(self):
        if self.hnsw is not None:
            self.hnsw.__post_init__()
        if not self.embedding_function:
            self.embedding_function = DefaultEmbeddingFunction()


@dataclass
class SparseVectorIndexConfig:
    """
    Sparse vector index configuration.

    Sparse vectors are suitable for keyword-based retrieval (e.g., BM25, SPLADE).
    They complement dense vectors and can be used for hybrid search.

    Args:
        embedding_function: Sparse embedding function (e.g., BM25EmbeddingFunction, SpladeEmbeddingFunction).
            If None, users must provide sparse vectors directly when adding data.
        source_key: Source field key specifying which field to generate sparse vectors from.
            - ``K.DOCUMENT`` or ``"#document"``: use the document field (default)
            - A plain string like ``"title"``: use ``metadata["title"]``
            - ``None``: users must provide sparse vectors directly
        lib: Vector index library (default: "vsag")
        distance: Distance metric (default: "inner_product"). Only inner_product is supported
            for sparse vectors.
        type: Index type (default: "sindi")
        prune: Whether to enable pruning (default: False)
        refine: Whether to enable refining (default: False)
        drop_ratio_build: Drop ratio for index building (default: 0.0)
        drop_ratio_search: Drop ratio for search (default: 0.0)
        refine_k: Refine K factor (default: 4.0)

    Note:
        - Each collection can have at most one sparse vector index.
        - Sparse vectors are stored in the ``sparse_embedding`` column.
        - ``embedding_function`` is required and must support persistence.
        - Sparse vectors are always generated from ``source_key`` by ``embedding_function``.

    Example:
        >>> # Auto-generate from document field
        >>> config = SparseVectorIndexConfig(
        ...     embedding_function=BM25EmbeddingFunction(),
        ...     source_key=K.DOCUMENT
        ... )
        >>>
        >>> # Auto-generate from metadata field
        >>> config = SparseVectorIndexConfig(
        ...     embedding_function=BM25EmbeddingFunction(),
        ...     source_key="title"
        ... )
        >>>
        >>> # Generate from metadata field
        >>> config = SparseVectorIndexConfig(
        ...     embedding_function=BM25EmbeddingFunction(),
        ...     source_key="title"
        ... )
    """

    embedding_function: SparseEmbeddingFunction | None = None
    source_key: str | K | None = K.DOCUMENT  # Default: generate from document field
    lib: str = "vsag"
    distance: str = DistanceMetric.INNER_PRODUCT.value
    type: str = "sindi"
    prune: bool | None = None
    refine: bool | None = None
    drop_ratio_build: float | None = None
    drop_ratio_search: float | None = None
    refine_k: float | None = None
    properties: dict[str, str | int | float | bool] | None = None

    def __post_init__(self):  # noqa: C901
        if self.distance != DistanceMetric.INNER_PRODUCT.value:
            raise ValueError(
                f"Sparse vector index only supports {DistanceMetric.INNER_PRODUCT.value} distance, got '{self.distance}'"
            )
        if self.lib.lower() != "vsag":
            raise ValueError(f"Sparse vector index only supports 'vsag' library, got '{self.lib}'")
        if self.type.lower() != "sindi":
            raise ValueError(f"Sparse vector index only supports 'sindi' type, got '{self.type}'")
        if self.drop_ratio_build is not None and (self.drop_ratio_build < 0.0 or self.drop_ratio_build > 0.9):
            raise ValueError(f"drop_ratio_build must be between 0.0 and 0.9, got '{self.drop_ratio_build}'")
        if self.drop_ratio_search is not None and (self.drop_ratio_search < 0.0 or self.drop_ratio_search > 0.9):
            raise ValueError(f"drop_ratio_search must be between 0.0 and 0.9, got '{self.drop_ratio_search}'")
        if self.refine_k is not None and (self.refine_k < 1.0 or self.refine_k > 1000.0):
            raise ValueError(f"refine_k must be between 1.0 and 1000.0, got '{self.refine_k}'")
        if self.properties:
            for _, value in self.properties.items():
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(f"properties must be a dictionary of string, int, float, or bool, got {value}")

        self._validate_source_key()
        if self.embedding_function is None:
            raise ValueError(
                "embedding_function is None. Please provide an embedding_function to generate sparse vectors."
            )
        if not SparseEmbeddingFunction.support_persistence(self.embedding_function):
            raise ValueError(
                "Sparse embedding function must support persistence. "
                "Please implement name(), get_config(), and build_from_config()."
            )

    def _validate_source_key(self) -> None:
        if self.source_key is None:
            self.source_key = K.DOCUMENT
            return

        key = self.source_key.name if hasattr(self.source_key, "name") else self.source_key
        if key == K.DOCUMENT.name:
            self.source_key = K.DOCUMENT
            return

        if not isinstance(key, str):
            raise TypeError(f"source_key must be a string, FieldKey, or None, got {type(key).__name__}")

        if key.startswith("#"):
            raise ValueError(f"source_key must not start with '#' except '#document', got '{self.source_key}'")
        self.source_key = key

    def resolve_source_key(self) -> tuple[str, str | None]:
        """
        Resolve the source_key to determine data source.

        Returns:
            Tuple of (source_type, metadata_key) where:
            - source_type is "document" or "metadata"
            - metadata_key is the metadata field name (only for "metadata" source_type)
        """
        if self.source_key is None:
            raise ValueError("source_key is None. Please provide a source_key to generate sparse vectors.")
        if self.source_key is K.DOCUMENT or self.source_key == K.DOCUMENT.name:
            return ("document", None)
        # Plain string refers to metadata field
        return ("metadata", self.source_key)


@deprecated("Configuration is deprecated. Please use Schema instead.")
class Configuration:
    """
    Configuration for collection creation

    Args:
        hnsw: HNSWConfiguration or None
        fulltext_config: FulltextIndexConfig or None. If None, defaults to FulltextIndexConfig(analyzer='ik')
    """

    def __init__(
        self,
        hnsw: HNSWConfiguration | None = None,
        fulltext_config: FulltextIndexConfig | None = None,
    ):
        self.hnsw = hnsw
        self.fulltext_config = fulltext_config


# Type alias for configuration parameter that can be HNSWConfiguration, None, or sentinel
ConfigurationParam = Configuration | HNSWConfiguration | None
