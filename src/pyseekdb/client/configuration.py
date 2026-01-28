from dataclasses import dataclass
from enum import Enum

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
    """

    dimension: int
    distance: str = DistanceMetric.L2.value

    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")
        valid_distances = [e.value for e in DistanceMetric]
        if self.distance not in valid_distances:
            raise ValueError(f"distance must be one of {valid_distances}, got {self.distance}")


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
