# SeekDBClient

SeekDBClient is a unified Python client that wraps three database connection modes—embedded SeekDB, remote SeekDB servers, and OceanBase—behind a single, concise API.

## Table of Contents

1. [Installation](#installation)
2. [Client Connection](#1-client-connection)
3. [AdminClient Connection and Database Management](#2-adminclient-connection-and-database-management)
4. [Collection (Table) Management](#3-collection-table-management)
5. [DML Operations](#4-dml-operations)
6. [DQL Operations](#5-dql-operations)
7. [Testing](#testing)

## Installation

```bash
poetry install
```

## 1. Client Connection

The `Client` class provides a unified interface for connecting to SeekDB in different modes. It automatically selects the appropriate connection mode based on the parameters provided.

### 1.1 Embedded SeekDB Client

Connect to a local embedded SeekDB instance:

```python
import seekdbclient

# Create embedded client
client = seekdbclient.Client(
    path="./seekdb",      # Path to SeekDB data directory
    database="demo"        # Database name
)

# Execute SQL queries
rows = client.execute("SELECT 1")
print(rows)
```

### 1.2 Remote SeekDB Server Client

Connect to a remote SeekDB server:

```python
import seekdbclient

# Create server client
client = seekdbclient.Client(
    host="127.0.0.1",      # Server host
    port=2882,              # Server port (default: 2882)
    database="demo",        # Database name
    user="root",            # Username (default: "root")
    password=""             # Password
)
```

### 1.3 OceanBase Client

Connect to OceanBase database:

```python
import seekdbclient

# Create OceanBase client
client = seekdbclient.OBClient(
    host="127.0.0.1",       # Server host
    port=11402,             # OceanBase port
    tenant="mysql",         # Tenant name
    database="test",        # Database name
    user="root",            # Username
    password=""             # Password
)
```

### 1.4 Client Methods and Properties

| Method / Property     | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `execute(sql)`        | Execute SQL statement and return cursor results (commits automatically when needed) |
| `is_connected()`      | Check whether an underlying connection is active               |
| `get_raw_connection()`| Access the underlying seekdb / pymysql connection             |
| `mode`                | Returns the concrete client class name (`SeekdbEmbeddedClient`, `SeekdbServerClient`, or `OceanBaseServerClient`) |
| `create_collection()`  | Create a new collection (see Collection Management)            |
| `get_collection()`    | Get an existing collection object                              |
| `delete_collection()` | Delete a collection                                            |
| `list_collections()`  | List all collections in the current database                   |
| `has_collection()`    | Check if a collection exists                                   |
| `get_or_create_collection()` | Get an existing collection or create it if it doesn't exist |

**Note:** The `Client` factory function returns a proxy that only exposes collection operations. For database management operations, use `AdminClient` (see section 2).

## 2. AdminClient Connection and Database Management

The `AdminClient` class provides database management operations. It uses the same connection modes as `Client` but only exposes database management methods.

### 2.1 Embedded/Server AdminClient

```python
import seekdbclient

# Embedded mode - Database management
admin = seekdbclient.AdminClient(path="./seekdb")

# Server mode - Database management
admin = seekdbclient.AdminClient(
    host="127.0.0.1",
    port=2881,
    user="root",
    password=""
)

# Use context manager
with seekdbclient.AdminClient(host="127.0.0.1", port=2881, user="root") as admin:
    # Create database
    admin.create_database("my_database")
    
    # List all databases
    databases = admin.list_databases()
    for db in databases:
        print(f"Database: {db.name}")
    
    # Get database information
    db = admin.get_database("my_database")
    print(f"Database: {db.name}, Charset: {db.charset}")
    
    # Delete database
    admin.delete_database("my_database")
```

### 2.2 OceanBase AdminClient

```python
import seekdbclient

# OceanBase mode - Database management (multi-tenant)
admin = seekdbclient.OBAdminClient(
    host="127.0.0.1",
    port=11402,
    tenant="mysql",        # Tenant name
    user="root",
    password=""
)

# Use context manager
with seekdbclient.OBAdminClient(
    host="127.0.0.1",
    port=11402,
    tenant="mysql",
    user="root"
) as admin:
    # Create database in tenant
    admin.create_database("analytics", tenant="mysql")
    
    # List databases in tenant
    databases = admin.list_databases(tenant="mysql")
    for db in databases:
        print(f"Database: {db.name}, Tenant: {db.tenant}")
    
    # Get database
    db = admin.get_database("analytics", tenant="mysql")
    
    # Delete database
    admin.delete_database("analytics", tenant="mysql")
```

### 2.3 AdminClient Methods

| Method                    | Description                                        |
|---------------------------|----------------------------------------------------|
| `create_database(name, tenant=DEFAULT_TENANT)` | Create a new database (tenant ignored for embedded/server mode) |
| `get_database(name, tenant=DEFAULT_TENANT)`    | Get database object with metadata (tenant ignored for embedded/server mode) |
| `delete_database(name, tenant=DEFAULT_TENANT)`  | Delete a database (tenant ignored for embedded/server mode) |
| `list_databases(limit=None, offset=None, tenant=DEFAULT_TENANT)` | List all databases with optional pagination (tenant ignored for embedded/server mode) |

**Parameters:**
- `name` (str): Database name
- `tenant` (str, optional): Tenant name (required for OceanBase, ignored for embedded/server mode)
- `limit` (int, optional): Maximum number of results to return
- `offset` (int, optional): Number of results to skip for pagination

**Note:** 
- Embedded/Server mode: No tenant concept (tenant=None in Database objects)
- OceanBase mode: Multi-tenant architecture (tenant is set in Database objects)

### 2.4 Database Object

The `get_database()` and `list_databases()` methods return `Database` objects with the following properties:

- `name` (str): Database name
- `tenant` (str, optional): Tenant name (None for embedded/server mode)
- `charset` (str, optional): Character set
- `collation` (str, optional): Collation
- `metadata` (dict): Additional metadata

## 3. Collection (Table) Management

Collections are the primary data structures in SeekDBClient, similar to tables in traditional databases. Each collection stores documents with vector embeddings, metadata, and full-text search capabilities.

### 3.1 Creating a Collection

```python
import seekdbclient

# Create a client
client = seekdbclient.Client(host="127.0.0.1", port=2881, database="test")

# Create a collection with vector dimension
collection = client.create_collection(
    name="my_collection",
    dimension=128  # Vector dimension (required)
)

# Get or create collection (creates if doesn't exist)
collection = client.get_or_create_collection(
    name="my_collection",
    dimension=128
)
```

### 3.2 Getting a Collection

```python
# Get an existing collection
collection = client.get_collection("my_collection")

# Check if collection exists
if client.has_collection("my_collection"):
    collection = client.get_collection("my_collection")
```

### 3.3 Listing Collections

```python
# List all collections
collections = client.list_collections()
for coll in collections:
    print(f"Collection: {coll.name}, Dimension: {coll.dimension}")
```

### 3.4 Deleting a Collection

```python
# Delete a collection
client.delete_collection("my_collection")
```

### 3.5 Collection Properties

Each `Collection` object has the following properties:

- `name` (str): Collection name
- `id` (str, optional): Collection unique identifier
- `dimension` (int, optional): Vector dimension
- `metadata` (dict): Collection metadata
- `client`: Reference to the client that created it

## 4. DML Operations

DML (Data Manipulation Language) operations allow you to insert, update, and delete data in collections.

### 4.1 Add Data

The `add()` method inserts new records into a collection. If a record with the same ID already exists, an error will be raised.

```python
# Add single item
collection.add(
    ids="item1",
    vectors=[0.1, 0.2, 0.3],
    documents="This is a document",
    metadatas={"category": "AI", "score": 95}
)

# Add multiple items
collection.add(
    ids=["item1", "item2", "item3"],
    vectors=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ],
    documents=[
        "Document 1",
        "Document 2",
        "Document 3"
    ],
    metadatas=[
        {"category": "AI", "score": 95},
        {"category": "ML", "score": 88},
        {"category": "DL", "score": 92}
    ]
)

# Add with only vectors
collection.add(
    ids=["vec1", "vec2"],
    vectors=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
)

# Add with only documents (no vectors)
collection.add(
    ids="doc1",
    documents="Text document without vector"
)
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs (required)
- `vectors` (List[float] or List[List[float]], optional): Single vector or list of vectors
- `documents` (str or List[str], optional): Single document or list of documents
- `metadatas` (dict or List[dict], optional): Single metadata dict or list of metadata dicts

### 4.2 Update Data

The `update()` method updates existing records in a collection. Records must exist, otherwise an error will be raised.

```python
# Update single item
collection.update(
    ids="item1",
    metadatas={"category": "AI", "score": 98}  # Update metadata only
)

# Update multiple items
collection.update(
    ids=["item1", "item2"],
    vectors=[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],  # Update vectors
    documents=["Updated document 1", "Updated document 2"]  # Update documents
)

# Update specific fields
collection.update(
    ids="item1",
    documents="New document text"  # Only update document
)
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs to update (required)
- `vectors` (List[float] or List[List[float]], optional): New vectors
- `documents` (str or List[str], optional): New documents
- `metadatas` (dict or List[dict], optional): New metadata

### 4.3 Upsert Data

The `upsert()` method inserts new records or updates existing ones. If a record with the given ID exists, it will be updated; otherwise, a new record will be inserted.

```python
# Upsert single item (insert or update)
collection.upsert(
    ids="item1",
    vectors=[0.1, 0.2, 0.3],
    documents="Document text",
    metadatas={"category": "AI", "score": 95}
)

# Upsert multiple items
collection.upsert(
    ids=["item1", "item2", "item3"],
    vectors=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ],
    documents=["Doc 1", "Doc 2", "Doc 3"],
    metadatas=[
        {"category": "AI"},
        {"category": "ML"},
        {"category": "DL"}
    ]
)
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs (required)
- `vectors` (List[float] or List[List[float]], optional): Vectors
- `documents` (str or List[str], optional): Documents
- `metadatas` (dict or List[dict], optional): Metadata

### 4.4 Delete Data

The `delete()` method removes records from a collection. You can delete by IDs, metadata filters, or document filters.

```python
# Delete by IDs
collection.delete(ids=["item1", "item2", "item3"])

# Delete by single ID
collection.delete(ids="item1")

# Delete by metadata filter
collection.delete(where={"category": {"$eq": "AI"}})

# Delete by comparison operator
collection.delete(where={"score": {"$lt": 50}})

# Delete by document filter
collection.delete(where_document={"$contains": "obsolete"})

# Delete with combined filters
collection.delete(
    where={"category": {"$eq": "AI"}},
    where_document={"$contains": "deprecated"}
)
```

**Parameters:**
- `ids` (str or List[str], optional): Single ID or list of IDs to delete
- `where` (dict, optional): Metadata filter conditions (see Filter Operators section)
- `where_document` (dict, optional): Document filter conditions

**Note:** At least one of `ids`, `where`, or `where_document` must be provided.

## 5. DQL Operations

DQL (Data Query Language) operations allow you to retrieve data from collections using various query methods.

### 5.1 Query (Vector Similarity Search)

The `query()` method performs vector similarity search to find the most similar documents to the query vector(s).

```python
# Basic vector similarity query
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    n_results=3
)

# Iterate over results
for item in results:
    print(f"ID: {item._id}, Distance: {item.distance}")
    print(f"Document: {item.document}")
    print(f"Metadata: {item.metadata}")

# Query with metadata filter
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"category": {"$eq": "AI"}},
    n_results=5
)

# Query with comparison operator
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"score": {"$gte": 90}},
    n_results=5
)

# Query with document filter
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where_document={"$contains": "machine learning"},
    n_results=5
)

# Query with combined filters
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
    where_document={"$contains": "machine"},
    n_results=5
)

# Query with $in operator
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"tag": {"$in": ["ml", "python"]}},
    n_results=5
)

# Query with logical operators ($or)
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={
        "$or": [
            {"category": {"$eq": "AI"}},
            {"tag": {"$eq": "python"}}
        ]
    },
    n_results=5
)

# Query with multiple vectors (batch query)
results = collection.query(
    query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    n_results=2
)
# Returns List[QueryResult], one for each query vector
for i, result in enumerate(results):
    print(f"Query {i}: {len(result)} results")

# Query with specific fields
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    include=["documents", "metadatas", "embeddings"],
    n_results=3
)

# Query by texts (will be embedded automatically if embedding function is available)
# Note: Currently requires query_embeddings to be provided directly
results = collection.query(
    query_texts=["my query text"],
    n_results=10
)
```

**Parameters:**
- `query_embeddings` (List[float] or List[List[float]], optional): Single vector or list of vectors for batch queries
- `query_texts` (str or List[str], optional): Query text(s) to be embedded (requires embedding function)
- `n_results` (int, required): Number of similar results to return (default: 10)
- `where` (dict, optional): Metadata filter conditions (see Filter Operators section)
- `where_document` (dict, optional): Document content filter
- `include` (List[str], optional): List of fields to include: `["documents", "metadatas", "embeddings"]`

**Returns:**
- If single vector/text provided: `QueryResult` object containing query results
- If multiple vectors/texts provided: `List[QueryResult]` objects, one for each query vector

**QueryResult Object:**
- Iterable: `for item in results: ...`
- Indexable: `results[0]` to get first item
- Each item has:
  - `_id`: Record ID (always included)
  - `document`: Document text (if included)
  - `embedding`: Vector embedding (if included)
  - `metadata`: Metadata dictionary (if included)
  - `distance`: Similarity distance (always included for query)

### 5.2 Get (Retrieve by IDs or Filters)

The `get()` method retrieves documents from a collection without vector similarity search. It supports filtering by IDs, metadata, and document content.

```python
# Get by single ID
results = collection.get(ids="123")

# Get by multiple IDs
results = collection.get(ids=["1", "2", "3"])

# Get by metadata filter
results = collection.get(
    where={"category": {"$eq": "AI"}},
    limit=10
)

# Get by comparison operator
results = collection.get(
    where={"score": {"$gte": 90}},
    limit=10
)

# Get by $in operator
results = collection.get(
    where={"tag": {"$in": ["ml", "python"]}},
    limit=10
)

# Get by logical operators ($or)
results = collection.get(
    where={
        "$or": [
            {"category": {"$eq": "AI"}},
            {"tag": {"$eq": "python"}}
        ]
    },
    limit=10
)

# Get by document content filter
results = collection.get(
    where_document={"$contains": "machine learning"},
    limit=10
)

# Get with combined filters
results = collection.get(
    where={"category": {"$eq": "AI"}},
    where_document={"$contains": "machine"},
    limit=10
)

# Get with pagination
results = collection.get(limit=2, offset=1)

# Get with specific fields
results = collection.get(
    ids=["1", "2"],
    include=["documents", "metadatas", "embeddings"]
)

# Get all data (up to limit)
results = collection.get(limit=100)
```

**Parameters:**
- `ids` (str or List[str], optional): Single ID or list of IDs to retrieve
- `where` (dict, optional): Metadata filter conditions (see Filter Operators section)
- `where_document` (dict, optional): Document content filter using `$contains` for full-text search
- `limit` (int, optional): Maximum number of results to return
- `offset` (int, optional): Number of results to skip for pagination
- `include` (List[str], optional): List of fields to include: `["documents", "metadatas", "embeddings"]`

**Returns:**
- If single ID provided: `QueryResult` object containing get results for that ID
- If multiple IDs provided: `List[QueryResult]` objects, one for each ID
- If filters provided (no IDs): `QueryResult` object containing all matching results

**Note:** If no parameters provided, returns all data (up to limit).

### 5.3 Hybrid Search

The `hybrid_search()` method combines full-text search and vector similarity search with ranking.

```python
# Hybrid search with both full-text and vector search
results = collection.hybrid_search(
    query={
        "where_document": {"$contains": "machine learning"},
        "where": {"category": {"$eq": "science"}},
        "n_results": 10
    },
    knn={
        "query_texts": ["AI research"],
        "where": {"year": {"$gte": 2020}},
        "n_results": 10
    },
    rank={"rrf": {}},  # Reciprocal Rank Fusion
    n_results=5,
    include=["documents", "metadatas", "embeddings"]
)
```

**Parameters:**
- `query` (dict, optional): Full-text search configuration with:
  - `where_document`: Document filter conditions
  - `where`: Metadata filter conditions
  - `n_results`: Number of results for full-text search
- `knn` (dict, optional): Vector search configuration with:
  - `query_texts` or `query_embeddings`: Query vectors/texts
  - `where`: Metadata filter conditions
  - `n_results`: Number of results for vector search
- `rank` (dict, optional): Ranking configuration (e.g., `{"rrf": {"rank_window_size": 60, "rank_constant": 60}}`)
- `n_results` (int): Final number of results to return after ranking (default: 10)
- `include` (List[str], optional): Fields to include in results

**Returns:**
Search results dictionary containing ids, distances, metadatas, documents, embeddings, etc.

### 5.4 Filter Operators

#### Metadata Filters (`where` parameter)

- `$eq`: Equal to
  ```python
  where={"category": {"$eq": "AI"}}
  ```

- `$ne`: Not equal to
  ```python
  where={"status": {"$ne": "deleted"}}
  ```

- `$gt`: Greater than
  ```python
  where={"score": {"$gt": 90}}
  ```

- `$gte`: Greater than or equal to
  ```python
  where={"score": {"$gte": 90}}
  ```

- `$lt`: Less than
  ```python
  where={"score": {"$lt": 50}}
  ```

- `$lte`: Less than or equal to
  ```python
  where={"score": {"$lte": 50}}
  ```

- `$in`: Value in array
  ```python
  where={"tag": {"$in": ["ml", "python", "ai"]}}
  ```

- `$nin`: Value not in array
  ```python
  where={"tag": {"$nin": ["deprecated", "old"]}}
  ```

- `$or`: Logical OR
  ```python
  where={
      "$or": [
          {"category": {"$eq": "AI"}},
          {"tag": {"$eq": "python"}}
      ]
  }
  ```

- `$and`: Logical AND
  ```python
  where={
      "$and": [
          {"category": {"$eq": "AI"}},
          {"score": {"$gte": 90}}
      ]
  }
  ```

#### Document Filters (`where_document` parameter)

- `$contains`: Full-text search (contains substring)
  ```python
  where_document={"$contains": "machine learning"}
  ```

- `$regex`: Regular expression matching (if supported)
  ```python
  where_document={"$regex": "pattern.*"}
  ```

- `$or`: Logical OR for document filters
  ```python
  where_document={
      "$or": [
          {"$contains": "machine learning"},
          {"$contains": "artificial intelligence"}
      ]
  }
  ```

- `$and`: Logical AND for document filters
  ```python
  where_document={
      "$and": [
          {"$contains": "machine"},
          {"$contains": "learning"}
      ]
  }
  ```

### 5.5 Collection Information Methods

```python
# Get item count
count = collection.count()
print(f"Collection has {count} items")

# Get detailed collection information
info = collection.describe()
print(f"Name: {info['name']}, Dimension: {info['dimension']}")
```

## Testing

```bash
# Run all tests
python3 -m pytest seekdbclient/tests/ -v

# Run tests with log output
python3 -m pytest seekdbclient/tests/ -v -s

# Run specific test
python3 -m pytest seekdbclient/tests/test_client_creation.py::TestClientCreation::test_create_server_client -v

# Run specific test file
python3 -m pytest seekdbclient/tests/test_client_creation.py -v
```

### Environment Variables (Optional)

`test_client_creation.py` honors the following overrides:

```bash
export SEEKDB_PATH=/data/seekdb
export SEEKDB_DATABASE=demo
export SERVER_HOST=127.0.0.1
export SERVER_PORT=2881           # SeekDB Server port
export SERVER_USER=root
export SERVER_PASSWORD=secret
export OB_HOST=127.0.0.1
export OB_PORT=11402               # OceanBase port
export OB_TENANT=mysql             # OceanBase tenant
export OB_USER=root
export OB_PASSWORD=
```

## Architecture

- **ClientAPI**: Collection operations interface
- **AdminAPI**: Database operations interface
- **ServerAPI (BaseClient)**: Implements both interfaces
- **_ClientProxy**: Exposes only collection operations
- **_AdminClientProxy**: Exposes only database operations

```
Client() → _ClientProxy → BaseClient (ServerAPI)
AdminClient() → _AdminClientProxy → BaseClient (ServerAPI)
```

## License

MIT License
