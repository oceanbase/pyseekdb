"""
Tests for detect_db_type_and_version method
Tests database type and version detection functionality for server and oceanbase client modes using client fixtures
"""

import pytest

import pyseekdb
from pyseekdb.client.version import Version


class TestDetectDbTypeAndVersion:
    """Tests for detect_db_type_and_version method"""

    def test_version_comparison(self):
        """Test Version class comparison functionality (pure unit test, no database connection required)"""
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

    @pytest.mark.parametrize("db_client", ["server"], indirect=True)
    def test_seekdb_type_detection(self, db_client):
        """Test: detect seekdb Server type and version"""
        # Verify client type
        assert db_client is not None
        assert hasattr(db_client, "_server")
        assert isinstance(db_client._server, pyseekdb.RemoteServerClient)

        # Test detect_db_type_and_version
        db_type, version = db_client._server.detect_db_type_and_version()

        # Verify results
        assert db_type == "seekdb"
        assert version is not None
        assert isinstance(version, Version)
        # Test version comparison
        assert version > Version("0.0.0.0"), (  # noqa: S104
            f"Version should be greater than 0.0.0.0, got: {version}"
        )

        print("\n✅ Successfully detected seekdb Server")
        print(f"   Database type: {db_type}")
        print(f"   Version: {version}")

    @pytest.mark.parametrize("db_client", ["oceanbase"], indirect=True)
    def test_ob_type_detection(self, db_client):
        """Test: detect OceanBase Server type and version"""
        # Verify client type
        assert db_client is not None
        assert hasattr(db_client, "_server")
        assert isinstance(db_client._server, pyseekdb.RemoteServerClient)

        # Test detect_db_type_and_version
        db_type, version = db_client._server.detect_db_type_and_version()

        # Verify results
        assert db_type == "oceanbase"
        assert version is not None
        assert isinstance(version, Version)
        # Test version comparison
        assert version > Version("0.0.0.0"), (  # noqa: S104
            f"Version should be greater than 0.0.0.0, got: {version}"
        )

        print("\n✅ Successfully detected OceanBase Server")
        print(f"   Database type: {db_type}")
        print(f"   Version: {version}")

    def test_connection_establishment(self, db_client):
        """Test: verify detect_db_type_and_version establishes connection automatically"""
        # Note: db_client from fixture may already be connected due to connection test
        # We test that the method works correctly

        # Call detect_db_type_and_version
        db_type, version = db_client._server.detect_db_type_and_version()

        # Verify connection is established
        assert db_client._server.is_connected()

        # Verify results
        assert db_type in ["seekdb", "oceanbase"]
        assert version is not None

        print("\n✅ detect_db_type_and_version successfully works with connection")
        print(f"   Database type: {db_type}")
        print(f"   Version: {version}")

    def test_return_format(self, db_client):
        """Test: verify detect_db_type_and_version returns correct tuple format"""
        # Test detect_db_type_and_version
        result = db_client._server.detect_db_type_and_version()

        # Verify return type is tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        db_type, version = result

        # Verify tuple elements
        assert isinstance(db_type, str)
        assert isinstance(version, Version)
        assert db_type in ["seekdb", "oceanbase"]
        assert version > Version("0.0.0.0")  # noqa: S104

        print("\n✅ detect_db_type_and_version returns correct tuple format")
        print(f"   Result: {result}")
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
