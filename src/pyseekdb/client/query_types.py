"""
Query-related type definitions and data classes for pyseekdb client.

This module contains types and classes used for query operations,
such as query hints for database optimization.
"""

from dataclasses import dataclass


@dataclass
class QueryHint:
    """
    Query hint for database optimization

    Args:
        parallel: Number of parallel execution threads (optional)
        query_timeout: Query timeout in seconds (optional, converted to microseconds for OceanBase)
    """

    parallel: int | None = None
    query_timeout: float | None = None

    def __post_init__(self):
        if self.parallel is not None and self.parallel <= 0:
            raise ValueError(f"parallel must be positive, got {self.parallel}")
        if self.query_timeout is not None and self.query_timeout <= 0:
            raise ValueError(f"query_timeout must be positive, got {self.query_timeout}")
