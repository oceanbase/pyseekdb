# 2. AdminClient Connection and Database Management

The `AdminClient` class provides database management operations. It uses the same connection modes as `Client` but only exposes database management methods.

## 2.1 Embedded/Server AdminClient

```python
import pyseekdb

# Embedded mode - Database management
admin = pyseekdb.AdminClient(path="./seekdb")

# Remote server mode - Database management (seekdb Server)
admin = pyseekdb.AdminClient(
    host="127.0.0.1",
    port=2881,
    user="root",
    password=""  # Can be retrieved from SEEKDB_PASSWORD environment variable
)

# Remote server mode - Database management (OceanBase Server)
admin = pyseekdb.AdminClient(
    host="127.0.0.1",
    port=2881,
    tenant="sys",  # Default tenant for OceanBase
    user="root",
    password=""  # Can be retrieved from SEEKDB_PASSWORD environment variable
)
```


## 2.2 AdminClient Methods

| Method                    | Description                                        |
|---------------------------|----------------------------------------------------|
| `create_database(name, tenant=DEFAULT_TENANT)` | Create a new database (uses client's tenant for remote oceanbase server mode) |
| `get_database(name, tenant=DEFAULT_TENANT)`    | Get database object with metadata (uses client's tenant for remote oceanbase server mode) |
| `delete_database(name, tenant=DEFAULT_TENANT)`  | Delete a database (uses client's tenant for remote oceanbase server mode) |
| `list_databases(limit=None, offset=None, tenant=DEFAULT_TENANT)` | List all databases with optional pagination (uses client's tenant for remote oceanbase server mode) |

**Parameters:**
- `name` (str): Database name
- `tenant` (str, optional): Tenant name (uses client's tenant if different, ignored for seekdb)
- `limit` (int, optional): Maximum number of results to return
- `offset` (int, optional): Number of results to skip for pagination

## 2.4 Database Object

The `get_database()` and `list_databases()` methods return `Database` objects with the following properties:

- `name` (str): Database name
- `tenant` (str, optional): Tenant name (None for embedded/server mode)
- `charset` (str, optional): Character set
- `collation` (str, optional): Collation
- `metadata` (dict): Additional metadata
