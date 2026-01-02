"""
Unit tests for collection name validation.
"""
import pytest
import sys
from pathlib import Path

# Ensure local src/ is on sys.path so we import the in-repo pyseekdb,
# not an already-installed version in the virtualenv.
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import _validate_collection_name  # type: ignore
from pyseekdb.client.meta_info import CollectionNames


class TestCollectionNameValidation:
    """Tests for collection name constraints."""

    @property
    def _effective_max_length(self) -> int:
        """
        Calculate effective maximum name length based on current prefix and
        database table name limit used in client_base.
        """
        from pyseekdb.client.client_base import (  # type: ignore
            _MAX_COLLECTION_NAME_LENGTH,
            _MAX_TABLE_NAME_LENGTH,
        )

        prefix = CollectionNames.table_name("")
        available = max(0, _MAX_TABLE_NAME_LENGTH - len(prefix))
        return min(_MAX_COLLECTION_NAME_LENGTH, available)

    def test_valid_names(self):
        """Names with allowed characters and length should pass."""
        max_len = self._effective_max_length
        valid_names = [
            "a",
            "A",
            "0",
            "collection_1",
            "MyCollection_123",
            "A" * max_len,
        ]
        for name in valid_names:
            _validate_collection_name(name)

    def test_invalid_type(self):
        """Non-string names should raise TypeError."""
        with pytest.raises(TypeError):
            _validate_collection_name(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            _validate_collection_name(None)  # type: ignore[arg-type]

    def test_empty_name(self):
        """Empty string should be rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_collection_name("")

    def test_name_too_long(self):
        """Names longer than effective maximum should be rejected."""
        max_len = self._effective_max_length
        long_name = "a" * (max_len + 1)
        with pytest.raises(ValueError, match="maximum allowed is"):
            _validate_collection_name(long_name)

    def test_invalid_characters(self):
        """Names with characters outside [a-zA-Z0-9_] should be rejected."""
        invalid_names = [
            "name-with-dash",
            "name.with.dot",
            "name with space",
            "name$",
            "名字",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Only letters, digits, and underscore"):
                _validate_collection_name(name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
