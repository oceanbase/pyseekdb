"""
Unit tests for concurrent-safe get_or_create_collection helpers.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import (  # noqa: E402
    BaseClient,
    _is_collection_conflict_error,
)
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402


class TestCollectionConflictDetection:
    def test_detects_value_error_for_existing_collection(self):
        assert _is_collection_conflict_error(ValueError("Collection 'items' already exists"))

    def test_detects_seekdb_table_exists_error(self):
        class SeekdbError(Exception):
            pass

        exc = SeekdbError("Table 'c$v2$abc' already exists failed: code=1050")
        assert _is_collection_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        assert not _is_collection_conflict_error(ValueError("invalid dimension"))

    def test_ignores_metadata_failure_without_conflict_cause(self):
        assert not _is_collection_conflict_error(
            ValueError("Failed to create collection metadata: Collection not found: 'items'")
        )

    def test_detects_conflict_in_cause_chain(self):
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        assert _is_collection_conflict_error(outer)


class TestGetOrCreateCollectionRecovery:
    def test_returns_existing_collection_after_create_conflict(self):
        client = MagicMock(spec=BaseClient)
        client.has_collection.side_effect = [False, True]
        existing = object()
        client.get_collection.return_value = existing
        client.create_collection.side_effect = ValueError("Collection 'items' already exists")

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_retries_get_after_wrapped_table_conflict(self):
        client = MagicMock(spec=BaseClient)
        client.has_collection.side_effect = [False, True]
        existing = object()
        client.get_collection.return_value = existing
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        client.create_collection.side_effect = outer

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
