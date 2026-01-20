"""
Unit tests for Version class
Tests version comparison and parsing functionality
"""

import pytest

from pyseekdb.client.version import Version


class TestVersion:
    """Unit tests for Version class"""

    def test_version_comparison(self):
        """Test Version class comparison functionality"""
        # Test version comparison
        version1 = Version("1.0.1.0")
        version2 = Version("1.0.0.1")

        # version1 should be greater than version2
        assert version1 > version2, f"Expected {version1} > {version2}"
        assert version1 >= version2, f"Expected {version1} >= {version2}"
        assert version2 < version1, f"Expected {version2} < {version1}"
        assert version2 <= version1, f"Expected {version2} <= {version1}"
        assert version1 != version2, f"Expected {version1} != {version2}"

        # Test equality
        version3 = Version("1.0.1.0")
        assert version1 == version3, f"Expected {version1} == {version3}"

        # Test 3-part version (should be normalized to 4 parts)
        version4 = Version("1.2.3")
        version5 = Version("1.2.3.0")
        assert version4 == version5, f"Expected {version4} == {version5}"

        # Test string representation (preserve all parts including trailing .0)
        assert str(version1) == "1.0.1.0"
        assert str(version4) == "1.2.3.0"  # Full version preserved

        print("\n✅ Version comparison tests passed")
        print(f"   version1={version1}, version2={version2}")
        print(f"   version1 > version2: {version1 > version2}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
