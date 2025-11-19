"""
AdminClient database management tests - testing all database CRUD operations
Tests create, get, list, and delete database operations for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb_store"))

# Server mode (seekdb Server)
SERVER_HOST = os.environ.get('SERVER_HOST', '127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))  # seekdb Server port
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SERVER_PASSWORD', '')

# OceanBase mode
OB_HOST = os.environ.get('OB_HOST', '127.0.0.1')
OB_PORT = int(os.environ.get('OB_PORT', '11202'))
OB_TENANT = os.environ.get('OB_TENANT', 'mysql')
OB_USER = os.environ.get('OB_USER', 'root')
OB_PASSWORD = os.environ.get('OB_PASSWORD', '')


class TestAdminDatabaseManagement:
    """Test AdminClient database management operations for all three modes"""
    
    def test_embedded_admin_database_operations(self):
        """Test embedded admin client database management: create, get, list, delete"""
        # Check if seekdb package is available and properly configured
        try:
            import sys
            project_root_str = str(project_root)
            if project_root_str in sys.path:
                sys.path.remove(project_root_str)
            import pylibseekdb as seekdb
            if not hasattr(seekdb, 'open') and not hasattr(seekdb, '_initialize_module'):
                pytest.fail(
                    "❌ seekdb embedded package is not properly installed!\n"
                    "The 'seekdb' module exists but lacks required methods.\n"
                    "Required: 'open' method or '_initialize_module' method\n\n"
                    "Solution: Please install the seekdb embedded package from correct source:\n"
                    "  pip install seekdb\n"
                    "Or contact the seekdb package maintainer for installation guide."
                )
        except ImportError:
            pytest.fail(
                "❌ seekdb embedded package is not installed!\n"
                "The 'seekdb' module cannot be imported.\n\n"
                "Solution: Please install the seekdb embedded package:\n"
                "  pip install seekdb\n"
                "Or contact the seekdb package maintainer for installation guide."
            )
        finally:
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
        
        # Create admin client (returns _AdminClientProxy)
        admin = pyseekdb.AdminClient(
            path=SEEKDB_PATH
        )
        
        # Verify admin client type
        assert admin is not None
        assert hasattr(admin, '_server')
        assert isinstance(admin._server, pyseekdb.SeekdbEmbeddedClient)
        
        # Test database operations
        test_db_name = "test_embedded_db"
        try:
            print(f"\n✅ Embedded admin client created successfully: path={SEEKDB_PATH}")
            
            # Step 1: List all databases before test
            print(f"\n📋 Step 1: List all databases")
            databases_before = admin.list_databases()
            assert databases_before is not None
            assert isinstance(databases_before, (list, tuple))
            print(f"   Found {len(databases_before)} databases before test")
            for db in databases_before[:3]:
                print(f"   - {db.name} (tenant={db.tenant})")
            
            # Step 2: Create new database
            print(f"\n📝 Step 2: Create database '{test_db_name}'")
            admin.create_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' created")
            
            # Step 3: Get the created database and verify
            print(f"\n🔍 Step 3: Get database '{test_db_name}' to verify creation")
            db = admin.get_database(test_db_name)
            assert db is not None
            assert db.name == test_db_name
            assert db.tenant is None  # Embedded mode has no tenant
            print(f"   ✅ Database retrieved: {db.name}")
            print(f"      - Name: {db.name}")
            print(f"      - Tenant: {db.tenant}")
            print(f"      - Charset: {db.charset}")
            print(f"      - Collation: {db.collation}")
            
            # Step 4: Delete the database
            print(f"\n🗑️  Step 4: Delete database '{test_db_name}'")
            admin.delete_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' deleted")
            
            # Step 5: List databases again to verify deletion
            print(f"\n📋 Step 5: List all databases to verify deletion")
            databases_after = admin.list_databases()
            assert databases_after is not None
            print(f"   Found {len(databases_after)} databases after deletion")
            # Verify the test database is not in the list
            db_names = [db.name for db in databases_after]
            assert test_db_name not in db_names, f"Database '{test_db_name}' should be deleted"
            print(f"   ✅ Verified: '{test_db_name}' is not in the database list")
            
            print(f"\n🎉 All database management operations completed successfully!")
            
        except Exception as e:
            # Cleanup: try to delete test database if it exists
            try:
                admin.delete_database(test_db_name)
            except:
                pass
            pytest.fail(f"Embedded admin client test failed: {e}\n"
                       f"Hint: Please ensure seekdb embedded package is properly installed")
    
    def test_server_admin_database_operations(self):
        """Test server admin client database management: create, get, list, delete"""
        # Create admin client (returns _AdminClientProxy)
        admin = pyseekdb.AdminClient(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",  # Default tenant for seekdb Server
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        # Verify admin client type
        assert admin is not None
        assert hasattr(admin, '_server')
        assert isinstance(admin._server, pyseekdb.RemoteServerClient)
        
        # Test database operations
        test_db_name = "test_server_db"
        try:
            print(f"\n✅ Server admin client created successfully: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}")
            
            # Step 1: List all databases before test
            print(f"\n📋 Step 1: List all databases")
            databases_before = admin.list_databases()
            assert databases_before is not None
            assert isinstance(databases_before, (list, tuple))
            print(f"   Found {len(databases_before)} databases before test")
            for db in databases_before[:3]:
                print(f"   - {db.name} (tenant={db.tenant})")
            
            # Step 2: Create new database
            print(f"\n📝 Step 2: Create database '{test_db_name}'")
            admin.create_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' created")
            
            # Step 3: Get the created database and verify
            print(f"\n🔍 Step 3: Get database '{test_db_name}' to verify creation")
            db = admin.get_database(test_db_name)
            assert db is not None
            assert db.name == test_db_name
            assert db.tenant == "sys"  # Server mode has tenant (default "sys")
            print(f"   ✅ Database retrieved: {db.name}")
            print(f"      - Name: {db.name}")
            print(f"      - Tenant: {db.tenant}")
            print(f"      - Charset: {db.charset}")
            print(f"      - Collation: {db.collation}")
            
            # Step 4: Delete the database
            print(f"\n🗑️  Step 4: Delete database '{test_db_name}'")
            admin.delete_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' deleted")
            
            # Step 5: List databases again to verify deletion
            print(f"\n📋 Step 5: List all databases to verify deletion")
            databases_after = admin.list_databases()
            assert databases_after is not None
            print(f"   Found {len(databases_after)} databases after deletion")
            # Verify the test database is not in the list
            db_names = [db.name for db in databases_after]
            assert test_db_name not in db_names, f"Database '{test_db_name}' should be deleted"
            print(f"   ✅ Verified: '{test_db_name}' is not in the database list")
            
            print(f"\n🎉 All database management operations completed successfully!")
            
        except Exception as e:
            # Cleanup: try to delete test database if it exists
            try:
                admin.delete_database(test_db_name)
            except:
                pass
            pytest.fail(f"Server admin client test failed ({SERVER_HOST}:{SERVER_PORT}): {e}\n"
                       f"Hint: Please ensure seekdb Server is running on port {SERVER_PORT}")
    
    def test_oceanbase_admin_database_operations(self):
        """Test OceanBase admin client database management: create, get, list, delete"""
        # Create admin client (returns _AdminClientProxy)
        admin = pyseekdb.AdminClient(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        # Verify admin client type
        assert admin is not None
        assert hasattr(admin, '_server')
        assert isinstance(admin._server, pyseekdb.RemoteServerClient)
        assert admin._server.tenant == OB_TENANT
        
        # Test database operations
        test_db_name = "test_oceanbase_db"
        try:
            print(f"\n✅ OceanBase admin client created successfully: {OB_USER}@{OB_TENANT}@{OB_HOST}:{OB_PORT}")
            
            # Step 1: List all databases before test
            print(f"\n📋 Step 1: List all databases in tenant '{OB_TENANT}'")
            databases_before = admin.list_databases()
            assert databases_before is not None
            assert isinstance(databases_before, (list, tuple))
            # Verify tenant is set for OceanBase
            for db in databases_before:
                assert db.tenant == OB_TENANT, f"Database {db.name} should have tenant {OB_TENANT}"
            print(f"   Found {len(databases_before)} databases before test")
            for db in databases_before[:3]:
                print(f"   - {db.name} (tenant={db.tenant})")
            
            # Step 2: Create new database
            print(f"\n📝 Step 2: Create database '{test_db_name}'")
            admin.create_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' created in tenant '{OB_TENANT}'")
            
            # Step 3: Get the created database and verify
            print(f"\n🔍 Step 3: Get database '{test_db_name}' to verify creation")
            db = admin.get_database(test_db_name)
            assert db is not None
            assert db.name == test_db_name
            assert db.tenant == OB_TENANT  # OceanBase mode has tenant
            print(f"   ✅ Database retrieved: {db.name}")
            print(f"      - Name: {db.name}")
            print(f"      - Tenant: {db.tenant}")
            print(f"      - Charset: {db.charset}")
            print(f"      - Collation: {db.collation}")
            
            # Step 4: Delete the database
            print(f"\n🗑️  Step 4: Delete database '{test_db_name}'")
            admin.delete_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' deleted from tenant '{OB_TENANT}'")
            
            # Step 5: List databases again to verify deletion
            print(f"\n📋 Step 5: List all databases to verify deletion")
            databases_after = admin.list_databases()
            assert databases_after is not None
            print(f"   Found {len(databases_after)} databases after deletion")
            # Verify the test database is not in the list
            db_names = [db.name for db in databases_after]
            assert test_db_name not in db_names, f"Database '{test_db_name}' should be deleted"
            print(f"   ✅ Verified: '{test_db_name}' is not in the database list")
            
            print(f"\n🎉 All database management operations completed successfully!")
            
        except Exception as e:
            # Cleanup: try to delete test database if it exists
            try:
                admin.delete_database(test_db_name)
            except:
                pass
            pytest.fail(f"OceanBase admin client test failed ({OB_HOST}:{OB_PORT}): {e}\n"
                       f"Hint: Please ensure OceanBase is running and tenant '{OB_TENANT}' is created")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("pyseekdb - AdminClient Database Management Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

