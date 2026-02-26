"""
AdminClient database management tests using admin_client fixture
Testing all database CRUD operations for all three modes
"""

import contextlib

import pytest


class TestAdminDatabaseManagement:
    """Test AdminClient database management operations using parameterized admin_client fixture"""

    def test_admin_database_operations(self, admin_client):
        """
        Test AdminClient database management: create, get, list, delete.

        Tests include:
        - list_databases: List all databases before and after operations
        - create_database: Create a new test database
        - get_database: Retrieve and verify the created database
        - delete_database: Delete the test database
        - Verification: Ensure database is deleted

        Automatically runs for: embedded, server, oceanbase
        """
        # Verify admin client is properly initialized
        assert admin_client is not None
        assert hasattr(admin_client, "_server")

        # Determine expected tenant based on client type
        server_mode = getattr(admin_client._server, "mode", "")
        server_class_name = admin_client._server.__class__.__name__
        if server_mode == "SeekdbEmbeddedClient" or server_class_name == "SeekdbEmbeddedClient":
            expected_tenant = None
            test_db_name = "test_embedded_db"
        elif admin_client._server.tenant == "sys":
            expected_tenant = "sys"
            test_db_name = "test_server_db"
        else:
            expected_tenant = admin_client._server.tenant
            test_db_name = "test_oceanbase_db"

        try:
            # Step 1: List all databases before test
            print("\n📋 Step 1: List all databases")
            databases_before = admin_client.list_databases()
            assert databases_before is not None
            assert isinstance(databases_before, (list, tuple))
            print(f"   Found {len(databases_before)} databases before test")
            for db in databases_before[:3]:
                print(f"   - {db.name} (tenant={db.tenant})")

            # Step 2: Create new database
            print(f"\n📝 Step 2: Create database '{test_db_name}'")
            admin_client.create_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' created")

            # Step 3: Get the created database and verify
            print(f"\n🔍 Step 3: Get database '{test_db_name}' to verify creation")
            db = admin_client.get_database(test_db_name)
            assert db is not None
            assert db.name == test_db_name
            assert db.tenant == expected_tenant, f"Expected tenant {expected_tenant}, got {db.tenant}"
            print(f"   ✅ Database retrieved: {db.name}")
            print(f"      - Name: {db.name}")
            print(f"      - Tenant: {db.tenant}")
            print(f"      - Charset: {db.charset}")
            print(f"      - Collation: {db.collation}")

            # Step 4: Delete the database
            print(f"\n🗑️  Step 4: Delete database '{test_db_name}'")
            admin_client.delete_database(test_db_name)
            print(f"   ✅ Database '{test_db_name}' deleted")

            # Step 5: List databases again to verify deletion
            print("\n📋 Step 5: List all databases to verify deletion")
            databases_after = admin_client.list_databases()
            assert databases_after is not None
            print(f"   Found {len(databases_after)} databases after deletion")
            # Verify the test database is not in the list
            db_names = [db.name for db in databases_after]
            assert test_db_name not in db_names, f"Database '{test_db_name}' should be deleted"
            print(f"   ✅ Verified: '{test_db_name}' is not in the database list")

            print("\n🎉 All database management operations completed successfully!")

        except Exception as e:
            # Cleanup: try to delete test database if it exists
            with contextlib.suppress(Exception):
                admin_client.delete_database(test_db_name)
            pytest.fail(f"Admin client test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
