"""
Tests for detect_db_type_and_version method
Tests database type and version detection functionality for all client modes (embedded, server, oceanbase)
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb
from pyseekdb.client.version import Version


# ==================== Environment Variable Configuration ====================
# Server mode (seekdb Server)
SERVER_HOST = os.environ.get('SERVER_HOST', '127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))
SERVER_DATABASE = os.environ.get('SERVER_DATABASE', 'test')
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SERVER_PASSWORD', '')

# OceanBase mode
OB_HOST = os.environ.get('OB_HOST', 'localhost')
OB_PORT = int(os.environ.get('OB_PORT', '11202'))
OB_TENANT = os.environ.get('OB_TENANT', 'mysql')
OB_DATABASE = os.environ.get('OB_DATABASE', 'test')
OB_USER = os.environ.get('OB_USER', 'root')
OB_PASSWORD = os.environ.get('OB_PASSWORD', '')


class TestDetectDbTypeAndVersion:
    """Tests for detect_db_type_and_version method"""
    
    def test_server_detect_seekdb(self):
        """Basic test: detect seekdb Server type and version"""
        # Create server client (returns _ClientProxy)
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",  # Default tenant for seekdb Server
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Verify client type
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        
        # Test connection and detection
        try:
            # Ensure connection is established
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
            
            # Test detect_db_type_and_version
            db_type, version = client._server.detect_db_type_and_version()
            
            # Verify results
            assert db_type == "seekdb"
            assert version is not None
            assert isinstance(version, Version)
            # Test version comparison
            assert version > Version("0.0.0.0"), f"Version should be greater than 0.0.0.0, got: {version}"
            
            print(f"\n✅ Successfully detected seekdb Server")
            print(f"   Database type: {db_type}")
            print(f"   Version: {version}")
            
        except Exception as e:
            pytest.fail(f"seekdb Server detection failed ({SERVER_HOST}:{SERVER_PORT}): {e}\n"
                       f"Hint: Please ensure seekdb Server is running on port {SERVER_PORT}")
    
    def test_oceanbase_detect_oceanbase(self):
        """Basic test: detect OceanBase Server type and version"""
        # Create OceanBase client (returns _ClientProxy)
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Verify client type
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        
        # Test connection and detection
        try:
            # Ensure connection is established
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
            
            # Test detect_db_type_and_version
            db_type, version = client._server.detect_db_type_and_version()
            
            # Verify results
            assert db_type == "oceanbase"
            assert version is not None
            assert isinstance(version, Version)
            # Test version comparison
            assert version > Version("0.0.0.0"), f"Version should be greater than 0.0.0.0, got: {version}"
            
            print(f"\n✅ Successfully detected OceanBase Server")
            print(f"   Database type: {db_type}")
            print(f"   Version: {version}")
            
        except Exception as e:
            pytest.fail(f"OceanBase Server detection failed ({OB_HOST}:{OB_PORT}): {e}\n"
                       f"Hint: Please ensure OceanBase is running and tenant '{OB_TENANT}' is created")
    
    def test_server_connection_establishment(self):
        """Basic test: verify detect_db_type_and_version establishes connection automatically"""
        # Create server client (returns _ClientProxy)
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Verify client is not connected initially (lazy loading)
        assert not client._server.is_connected()
        
        try:
            # Call detect_db_type_and_version (should establish connection)
            db_type, version = client._server.detect_db_type_and_version()
            
            # Verify connection was established
            assert client._server.is_connected()
            
            # Verify results
            assert db_type in ["seekdb", "oceanbase"]
            assert version is not None
            
            print(f"\n✅ detect_db_type_and_version successfully established connection")
            print(f"   Database type: {db_type}")
            print(f"   Version: {version}")
            
        except Exception as e:
            pytest.fail(f"Connection establishment test failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
    
    def test_server_return_format(self):
        """Basic test: verify detect_db_type_and_version returns correct tuple format"""
        # Create server client (returns _ClientProxy)
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        try:
            # Test detect_db_type_and_version
            result = client._server.detect_db_type_and_version()
            
            # Verify return type is tuple
            assert isinstance(result, tuple)
            assert len(result) == 2
            
            db_type, version = result
            
            # Verify tuple elements
            assert isinstance(db_type, str)
            assert isinstance(version, Version)
            assert db_type in ["seekdb", "oceanbase"]
            assert version > Version("0.0.0.0")
            
            print(f"\n✅ detect_db_type_and_version returns correct tuple format")
            print(f"   Result: {result}")
            print(f"   Type: {type(result)}")
            print(f"   Length: {len(result)}")
            
        except Exception as e:
            pytest.fail(f"Return format test failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
    
    def test_server_version_comparison(self):
        """Test Version class comparison functionality (pure unit test, no database connection required)"""
        # Test version comparison as mentioned in code review
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
        
        print(f"\n✅ Version comparison tests passed")
        print(f"   version1={version1}, version2={version2}")
        print(f"   version1 > version2: {version1 > version2}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("pyseekdb - Tests for detect_db_type_and_version")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

