"""
Unit tests for collection name validation.
"""
import pytest
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import pyseekdb from source
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyseekdb.client.client_base import _validate_collection_name  # type: ignore


class TestCollectionNameValidation:
    """Tests for collection name constraints."""

    def test_valid_names(self):
        """Names with allowed characters and length should pass."""
        valid_names = [
            "a",
            "A",
            "0",
            "collection_1",
            "MyCollection_123",
            "A" * 512,
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
        """Names longer than 512 characters should be rejected."""
        long_name = "a" * 513
        with pytest.raises(ValueError, match="maximum allowed is 512"):
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

