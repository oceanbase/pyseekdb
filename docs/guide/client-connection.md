# Client Connection

## 1. Client Connection

The `Client` class provides a unified interface for connecting to seekdb in different modes. It automatically selects the appropriate connection mode based on the parameters provided.

### 1.1 Embedded seekdb Client

Connect to a local embedded seekdb instance:

```python
import pyseekdb

# Create embedded client with explicit path
client = pyseekdb.Client(
    path="./seekdb",      # Path to seekdb data directory
    database="demo"        # Database name
)

# Create embedded client with default path (current working directory)
# If path is not provided, uses seekdb.db in the current process working directory
client = pyseekdb.Client(
    database="demo"        # Database name (path defaults to current working directory/seekdb.db)
)
```

### 1.2 Remote Server Client

Connect to a remote server (supports both seekdb Server and OceanBase Server):

```python
import pyseekdb

# Create remote server client (seekdb Server)
client = pyseekdb.Client(
    host="127.0.0.1",      # Server host
    port=2881,              # Server port (default: 2881)
    database="demo",        # Database name
    user="root",            # Username (default: "root")
    password=""             # Password (can be retrieved from SEEKDB_PASSWORD environment variable)
)

# Create remote server client (OceanBase Server)
client = pyseekdb.Client(
    host="127.0.0.1",      # Server host
    port=2881,              # Server port (default: 2881)
    tenant="sys",          # Tenant name (default: sys)
    database="demo",       # Database name
    user="root",           # Username (default: "root")
    password=""             # Password (can be retrieved from SEEKDB_PASSWORD environment variable)
)
```

**Note:** If the `password` parameter is not provided (empty string), the client will automatically retrieve it from the `SEEKDB_PASSWORD` environment variable. This is useful for keeping passwords out of your code:

```bash
export SEEKDB_PASSWORD="your_password"
```

```python
# Password will be automatically retrieved from SEEKDB_PASSWORD environment variable
client = pyseekdb.Client(
    host="127.0.0.1",
    port=2881,
    database="demo",
    user="root"
    # password parameter omitted - will use SEEKDB_PASSWORD from environment
)
```

### 1.3 Client Methods and Properties

| Method / Property     | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `create_collection()`  | Create a new collection (see Collection Management)            |
| `get_collection()`    | Get an existing collection object                              |
| `delete_collection()` | Delete a collection                                            |
| `list_collections()`  | List all collections in the current database                   |
| `has_collection()`    | Check if a collection exists                                   |
| `get_or_create_collection()` | Get an existing collection or create it if it doesn't exist |
| `count_collection()`  | Count the number of collections in the current database         |

**Note:** The `Client` factory function returns a proxy that only exposes collection operations. For database management operations, use `AdminClient` (see section 2).
