from dataclasses import dataclass
from typing import Optional, Union, Any

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


# Type alias for embedding_function parameter that can be EmbeddingFunction, None, or sentinel
EmbeddingFunctionParam = Union[EmbeddingFunction[EmbeddingDocuments], None, Any]

class Configuration:
    """
    Configuration for collection creation

    Args:
        hnsw: HNSWConfiguration or None
        ef: EmbeddingFunction or None
    """
    def __init__(self,
                 hnsw: Optional[HNSWConfiguration] = None,
                 ef: Optional[EmbeddingFunction[EmbeddingDocuments]] = None,
                 ):
        self.hnsw = hnsw
        self.ef = ef

    @staticmethod
    def of(configuration):
        pass

# Type alias for configuration parameter that can be HNSWConfiguration, None, or sentinel
ConfigurationParam = Union[Configuration, HNSWConfiguration, None, Any]
