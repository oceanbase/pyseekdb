# SeekDBClient

SeekDBClient is a unified Python client that wraps three database connection modes—embedded SeekDB, remote SeekDB servers, and OceanBase—behind a single, concise API.

## Table of Contents

1. [Installation](#installation)
2. [Client Connection](#1-client-connection)
3. [AdminClient Connection and Database Management](#2-adminclient-connection-and-database-management)
4. [Collection (Table) Management](#3-collection-table-management)
5. [DML Operations](#4-dml-operations)
6. [DQL Operations](#5-dql-operations)
7. [Embedding Functions](#6-embedding-functions)
8. [Testing](#testing)

## Installation

```bash
# install pyseekdb and all dependencies
pip install -U pyseekdb[all]
# install pyseekdb (no default embedding support)
pip install -U pyseekdb
```

## 1. Client Connection

The `Client` class provides a unified interface for connecting to SeekDB in different modes. It automatically selects the appropriate connection mode based on the parameters provided.

### 1.1 Embedded SeekDB Client

Connect to a local embedded SeekDB instance:

```python
import pyseekdb

# Create embedded client
client = pyseekdb.Client(
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
import pyseekdb

# Create server client
client = pyseekdb.Client(
    host="127.0.0.1",      # Server host
    port=2881,              # Server port (default: 2881)
    database="demo",        # Database name
    user="root",            # Username (default: "root")
    password=""             # Password
)
```

### 1.3 OceanBase Client

Connect to OceanBase database:

```python
import pyseekdb

# Create OceanBase client
client = pyseekdb.OBClient(
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
| `count_collection()`  | Count the number of collections in the current database         |

**Note:** The `Client` factory function returns a proxy that only exposes collection operations. For database management operations, use `AdminClient` (see section 2).

## 2. AdminClient Connection and Database Management

The `AdminClient` class provides database management operations. It uses the same connection modes as `Client` but only exposes database management methods.

### 2.1 Embedded/Server AdminClient

```python
import pyseekdb

# Embedded mode - Database management
admin = pyseekdb.AdminClient(path="./seekdb")

# Server mode - Database management
admin = pyseekdb.AdminClient(
    host="127.0.0.1",
    port=2881,
    user="root",
    password=""
)

# Use context manager
with pyseekdb.AdminClient(host="127.0.0.1", port=2881, user="root") as admin:
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
import pyseekdb

# OceanBase mode - Database management (multi-tenant)
admin = pyseekdb.OBAdminClient(
    host="127.0.0.1",
    port=11402,
    tenant="mysql",        # Tenant name
    user="root",
    password=""
)

# Use context manager
with pyseekdb.OBAdminClient(
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
import pyseekdb
from pyseekdb import DefaultEmbeddingFunction, HNSWConfiguration

# Create a client
client = pyseekdb.Client(host="127.0.0.1", port=2881, database="test")

# Create a collection with vector dimension (traditional way)
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=128, distance='cosine')
)

# Create a collection with default embedding function (auto-calculates dimension)
collection = client.create_collection(
    name="my_collection",
    embedding_function=DefaultEmbeddingFunction()  # Uses default model (384 dimensions)
)

# Create a collection with custom embedding function
ef = DefaultEmbeddingFunction(model_name='all-MiniLM-L6-v2')
config = HNSWConfiguration(dimension=384, distance='cosine')  # Must match EF dimension
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=ef
)

# Create a collection without embedding function (vectors must be provided manually)
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=128, distance='cosine'),
    embedding_function=None  # Explicitly disable embedding function
)

# Get or create collection (creates if doesn't exist)
collection = client.get_or_create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=128, distance='cosine'),
    embedding_function=DefaultEmbeddingFunction()
)
```

**Parameters:**
- `name` (str): Collection name (required)
- `configuration` (HNSWConfiguration, optional): Index configuration with dimension and distance metric
  - If not provided, uses default (dimension=384, distance='cosine')
  - If set to `None`, dimension will be calculated from `embedding_function`
- `embedding_function` (EmbeddingFunction, optional): Function to convert documents to vectors
  - If not provided, uses `DefaultEmbeddingFunction()` (384 dimensions)
  - If set to `None`, collection will not have an embedding function
  - If provided, the dimension will be automatically calculated and validated against `configuration.dimension`

**Note:** When `embedding_function` is provided, the system will automatically calculate the vector dimension by calling the function. If `configuration.dimension` is also provided, it must match the embedding function's dimension, otherwise a `ValueError` will be raised.

### 3.2 Getting a Collection

```python
# Get an existing collection (uses default embedding function if collection doesn't have one)
collection = client.get_collection("my_collection")

# Get collection with specific embedding function
ef = DefaultEmbeddingFunction(model_name='all-MiniLM-L6-v2')
collection = client.get_collection("my_collection", embedding_function=ef)

# Get collection without embedding function
collection = client.get_collection("my_collection", embedding_function=None)

# Check if collection exists
if client.has_collection("my_collection"):
    collection = client.get_collection("my_collection")
```

**Parameters:**
- `name` (str): Collection name (required)
- `embedding_function` (EmbeddingFunction, optional): Embedding function to use for this collection
  - If not provided, uses `DefaultEmbeddingFunction()` by default
  - If set to `None`, collection will not have an embedding function
  - **Important:** The embedding function set here will be used for all operations on this collection (add, upsert, update, query, hybrid_search) when documents/texts are provided without vectors

### 3.3 Listing Collections

```python
# List all collections
collections = client.list_collections()
for coll in collections:
    print(f"Collection: {coll.name}, Dimension: {coll.dimension}")

# Count collections in database
collection_count = client.count_collection()
print(f"Database has {collection_count} collections")
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
- `embedding_function` (EmbeddingFunction, optional): Embedding function associated with this collection
- `metadata` (dict): Collection metadata
- `client`: Reference to the client that created it

**Accessing Embedding Function:**
```python
collection = client.get_collection("my_collection")
if collection.embedding_function is not None:
    print(f"Collection uses embedding function: {collection.embedding_function}")
    print(f"Embedding dimension: {collection.embedding_function.dimension}")
```

## 4. DML Operations

DML (Data Manipulation Language) operations allow you to insert, update, and delete data in collections.

### 4.1 Add Data

The `add()` method inserts new records into a collection. If a record with the same ID already exists, an error will be raised.

**Behavior with Embedding Function:**

1. **If `vectors` are provided:** Vectors are used directly, `embedding_function` is NOT called (even if provided)
2. **If `vectors` are NOT provided but `documents` are provided:**
   - If collection has an `embedding_function` (set during creation or retrieval), it will automatically generate vectors from documents
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `vectors` nor `documents` are provided:** A `ValueError` will be raised

```python
# Add single item with vectors (embedding_function not used)
collection.add(
    ids="item1",
    vectors=[0.1, 0.2, 0.3],
    documents="This is a document",
    metadatas={"category": "AI", "score": 95}
)

# Add multiple items with vectors (embedding_function not used)
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

# Add with only vectors (no documents)
collection.add(
    ids=["vec1", "vec2"],
    vectors=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
)

# Add with only documents - vectors auto-generated by embedding_function
# Requires: collection must have embedding_function set
collection.add(
    ids=["doc1", "doc2"],
    documents=["Text document 1", "Text document 2"],
    metadatas=[{"tag": "A"}, {"tag": "B"}]
)
# The collection's embedding_function will automatically convert documents to vectors
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs (required)
- `vectors` (List[float] or List[List[float]], optional): Single vector or list of vectors
  - If provided, used directly (embedding_function is ignored)
  - If not provided, must provide `documents` and collection must have `embedding_function`
- `documents` (str or List[str], optional): Single document or list of documents
  - If `vectors` not provided, `documents` will be converted to vectors using collection's `embedding_function`
- `metadatas` (dict or List[dict], optional): Single metadata dict or list of metadata dicts

**Note:** The `embedding_function` used is the one associated with the collection (set during `create_collection()` or `get_collection()`). You cannot override it per-operation.

### 4.2 Update Data

The `update()` method updates existing records in a collection. Records must exist, otherwise an error will be raised.

**Behavior with Embedding Function:**

1. **If `vectors` are provided:** Vectors are used directly, `embedding_function` is NOT called
2. **If `vectors` are NOT provided but `documents` are provided:**
   - If collection has an `embedding_function`, it will automatically generate vectors from documents
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `vectors` nor `documents` are provided:** Only metadata will be updated (metadata-only update is allowed)

```python
# Update single item - metadata only (embedding_function not used)
collection.update(
    ids="item1",
    metadatas={"category": "AI", "score": 98}  # Update metadata only
)

# Update multiple items with vectors (embedding_function not used)
collection.update(
    ids=["item1", "item2"],
    vectors=[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],  # Update vectors
    documents=["Updated document 1", "Updated document 2"]  # Update documents
)

# Update with documents only - vectors auto-generated by embedding_function
# Requires: collection must have embedding_function set
collection.update(
    ids="item1",
    documents="New document text",  # Vectors will be auto-generated
    metadatas={"category": "AI"}
)

# Update specific fields - only document (vectors auto-generated)
collection.update(
    ids="item1",
    documents="New document text"  # Only update document, vectors auto-generated
)
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs to update (required)
- `vectors` (List[float] or List[List[float]], optional): New vectors
  - If provided, used directly (embedding_function is ignored)
  - If not provided, can provide `documents` to auto-generate vectors
- `documents` (str or List[str], optional): New documents
  - If `vectors` not provided, `documents` will be converted to vectors using collection's `embedding_function`
- `metadatas` (dict or List[dict], optional): New metadata

**Note:** Metadata-only updates (no vectors, no documents) are allowed. The `embedding_function` used is the one associated with the collection.

### 4.3 Upsert Data

The `upsert()` method inserts new records or updates existing ones. If a record with the given ID exists, it will be updated; otherwise, a new record will be inserted.

**Behavior with Embedding Function:**

1. **If `vectors` are provided:** Vectors are used directly, `embedding_function` is NOT called
2. **If `vectors` are NOT provided but `documents` are provided:**
   - If collection has an `embedding_function`, it will automatically generate vectors from documents
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `vectors` nor `documents` are provided:** Only metadata will be upserted (metadata-only upsert is allowed)

```python
# Upsert single item with vectors (embedding_function not used)
collection.upsert(
    ids="item1",
    vectors=[0.1, 0.2, 0.3],
    documents="Document text",
    metadatas={"category": "AI", "score": 95}
)

# Upsert multiple items with vectors (embedding_function not used)
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

# Upsert with documents only - vectors auto-generated by embedding_function
# Requires: collection must have embedding_function set
collection.upsert(
    ids=["item1", "item2"],
    documents=["Document 1", "Document 2"],
    metadatas=[{"category": "AI"}, {"category": "ML"}]
)
# The collection's embedding_function will automatically convert documents to vectors
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs (required)
- `vectors` (List[float] or List[List[float]], optional): Vectors
  - If provided, used directly (embedding_function is ignored)
  - If not provided, can provide `documents` to auto-generate vectors
- `documents` (str or List[str], optional): Documents
  - If `vectors` not provided, `documents` will be converted to vectors using collection's `embedding_function`
- `metadatas` (dict or List[dict], optional): Metadata

**Note:** Metadata-only upserts (no vectors, no documents) are allowed. The `embedding_function` used is the one associated with the collection.

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

**Behavior with Embedding Function:**

1. **If `query_embeddings` are provided:** Vectors are used directly, `embedding_function` is NOT called
2. **If `query_embeddings` are NOT provided but `query_texts` are provided:**
   - If collection has an `embedding_function`, it will automatically generate query vectors from texts
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `query_embeddings` nor `query_texts` are provided:** A `ValueError` will be raised

```python
# Basic vector similarity query (embedding_function not used)
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    n_results=3
)

# Iterate over results
for item in results:
    print(f"ID: {item._id}, Distance: {item.distance}")
    print(f"Document: {item.document}")
    print(f"Metadata: {item.metadata}")

# Query by texts - vectors auto-generated by embedding_function
# Requires: collection must have embedding_function set
results = collection.query(
    query_texts=["my query text"],
    n_results=10
)
# The collection's embedding_function will automatically convert query_texts to query_embeddings

# Query by multiple texts (batch query)
results = collection.query(
    query_texts=["query text 1", "query text 2"],
    n_results=5
)
# Returns List[QueryResult], one for each query text
for i, result in enumerate(results):
    print(f"Query {i}: {len(result)} results")

# Query with metadata filter (using query_texts)
results = collection.query(
    query_texts=["AI research"],
    where={"category": {"$eq": "AI"}},
    n_results=5
)

# Query with comparison operator (using query_texts)
results = collection.query(
    query_texts=["machine learning"],
    where={"score": {"$gte": 90}},
    n_results=5
)

# Query with document filter (using query_texts)
results = collection.query(
    query_texts=["neural networks"],
    where_document={"$contains": "machine learning"},
    n_results=5
)

# Query with combined filters (using query_texts)
results = collection.query(
    query_texts=["AI research"],
    where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
    where_document={"$contains": "machine"},
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
```

**Parameters:**
- `query_embeddings` (List[float] or List[List[float]], optional): Single vector or list of vectors for batch queries
  - If provided, used directly (embedding_function is ignored)
  - If not provided, must provide `query_texts` and collection must have `embedding_function`
- `query_texts` (str or List[str], optional): Query text(s) to be embedded
  - If `query_embeddings` not provided, `query_texts` will be converted to vectors using collection's `embedding_function`
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

**Note:** The `embedding_function` used is the one associated with the collection. You cannot override it per-query.

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

**Behavior with Embedding Function:**

In the `knn` parameter:
1. **If `query_embeddings` are provided:** Vectors are used directly, `embedding_function` is NOT called
2. **If `query_embeddings` are NOT provided but `query_texts` are provided:**
   - If collection has an `embedding_function`, it will automatically generate query vectors from texts
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `query_embeddings` nor `query_texts` are provided in `knn`:** Only full-text search will be performed (if `query` is provided)

```python
# Hybrid search with both full-text and vector search (using query_texts)
results = collection.hybrid_search(
    query={
        "where_document": {"$contains": "machine learning"},
        "where": {"category": {"$eq": "science"}},
        "n_results": 10
    },
    knn={
        "query_texts": ["AI research"],  # Will be embedded automatically
        "where": {"year": {"$gte": 2020}},
        "n_results": 10
    },
    rank={"rrf": {}},  # Reciprocal Rank Fusion
    n_results=5,
    include=["documents", "metadatas", "embeddings"]
)

# Hybrid search with query_embeddings (embedding_function not used)
results = collection.hybrid_search(
    query={
        "where_document": {"$contains": "machine learning"},
        "n_results": 10
    },
    knn={
        "query_embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Used directly
        "n_results": 10
    },
    rank={"rrf": {}},
    n_results=5
)

# Hybrid search with multiple query texts (batch)
results = collection.hybrid_search(
    query={
        "where_document": {"$contains": "AI"},
        "n_results": 10
    },
    knn={
        "query_texts": ["machine learning", "neural networks"],  # Multiple queries
        "n_results": 10
    },
    rank={"rrf": {}},
    n_results=5
)
```

**Parameters:**
- `query` (dict, optional): Full-text search configuration with:
  - `where_document`: Document filter conditions
  - `where`: Metadata filter conditions
  - `n_results`: Number of results for full-text search
- `knn` (dict, optional): Vector search configuration with:
  - `query_texts` (str or List[str], optional): Query text(s) to be embedded
    - If `query_embeddings` not provided, `query_texts` will be converted to vectors using collection's `embedding_function`
  - `query_embeddings` (List[float] or List[List[float]], optional): Query vector(s)
    - If provided, used directly (embedding_function is ignored)
  - `where`: Metadata filter conditions (optional)
  - `n_results`: Number of results for vector search (optional)
- `rank` (dict, optional): Ranking configuration (e.g., `{"rrf": {"rank_window_size": 60, "rank_constant": 60}}`)
- `n_results` (int): Final number of results to return after ranking (default: 10)
- `include` (List[str], optional): Fields to include in results

**Returns:**
Search results dictionary containing ids, distances, metadatas, documents, embeddings, etc.

**Note:** The `embedding_function` used is the one associated with the collection. You cannot override it per-search.

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

# Preview first few items in collection
preview = collection.peek(limit=5)
for item in preview:
    print(f"ID: {item._id}, Document: {item.document}")

# Count collections in database
collection_count = client.count_collection()
print(f"Database has {collection_count} collections")
```

**Methods:**
- `collection.count()` - Get the number of items in the collection
- `collection.describe()` - Get detailed collection information
- `collection.peek(limit=10)` - Quickly preview the first few items in the collection
- `client.count_collection()` - Count the number of collections in the current database

## 6. Embedding Functions

Embedding functions convert text documents into vector embeddings for similarity search. SeekDBClient supports both built-in and custom embedding functions.

### 6.1 Default Embedding Function

The `DefaultEmbeddingFunction` uses sentence-transformers and is the default embedding function if none is specified.

```python
from pyseekdb import DefaultEmbeddingFunction

# Use default model (all-MiniLM-L6-v2, 384 dimensions)
ef = DefaultEmbeddingFunction()

# Use custom model
ef = DefaultEmbeddingFunction(model_name='all-MiniLM-L6-v2')

# Get embedding dimension
print(f"Dimension: {ef.dimension}")  # 384

# Generate embeddings
embeddings = ef(["Hello world", "How are you?"])
print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
```

**Note:** The `DefaultEmbeddingFunction` requires the `sentence-transformers` package. Install it with:
```bash
pip install sentence-transformers
```

### 6.2 Creating Custom Embedding Functions

You can create custom embedding functions by implementing the `EmbeddingFunction` protocol. The function must:

1. Implement `__call__` method that accepts `Documents` (str or List[str]) and returns `Embeddings` (List[List[float]])
2. Optionally implement a `dimension` property to return the vector dimension

#### Example 1: Simple Hash-Based Embedding Function

```python
from typing import List, Union
import hashlib
from pyseekdb import EmbeddingFunction, Documents, Embeddings

class SimpleHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A simple custom embedding function that uses hash-based vectorization.
    
    This creates fixed-dimensional vectors by hashing the text.
    """
    
    def __init__(self, dimension: int = 128):
        """
        Initialize the hash-based embedding function.
        
        Args:
            dimension: The dimension of the embedding vectors (default: 128)
        """
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents using hash-based vectorization.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors (List[List[float]])
        """
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        embeddings = []
        for doc in input:
            # Create hash-based embedding
            hash_obj = hashlib.md5(doc.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()
            
            # Convert hash to vector
            vector = []
            for i in range(0, min(len(hash_hex), self._dimension * 2), 2):
                # Convert hex pair to float in range [0, 1]
                hex_pair = hash_hex[i:i+2]
                value = int(hex_pair, 16) / 255.0
                vector.append(value)
            
            # Pad or truncate to exact dimension
            while len(vector) < self._dimension:
                vector.append(0.0)
            vector = vector[:self._dimension]
            
            embeddings.append(vector)
        
        return embeddings

# Use the custom embedding function
ef = SimpleHashEmbeddingFunction(dimension=128)
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=128, distance='cosine'),
    embedding_function=ef
)
```

#### Example 2: Sentence-Transformer Custom Embedding Function

```python
from typing import List, Union
from pyseekdb import EmbeddingFunction, Documents, Embeddings

class SentenceTransformerCustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using sentence-transformers with a specific model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the sentence-transformer embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None
    
    def _ensure_model_loaded(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Get dimension from model
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                self._dimension = len(test_embedding[0])
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it with: pip install sentence-transformers"
                )
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        self._ensure_model_loaded()
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors
        """
        self._ensure_model_loaded()
        
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Generate embeddings
        embeddings = self._model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]

# Use the custom embedding function
ef = SentenceTransformerCustomEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device='cpu'
)
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=384, distance='cosine'),
    embedding_function=ef
)
```

#### Example 3: OpenAI Embedding Function

```python
from typing import List, Union
import os
import openai
from pyseekdb import EmbeddingFunction, Documents, Embeddings

class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using OpenAI's embedding API.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        """
        Initialize the OpenAI embedding function.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Dimension for text-embedding-ada-002 is 1536
        self._dimension = 1536 if "ada-002" in model_name else None
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        if self._dimension is None:
            # Call API to get dimension (or use known values)
            raise ValueError("Dimension not set for this model")
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors
        """
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Call OpenAI API
        response = openai.Embedding.create(
            model=self.model_name,
            input=input,
            api_key=self.api_key
        )
        
        # Extract embeddings
        embeddings = [item['embedding'] for item in response['data']]
        return embeddings

# Use the custom embedding function
ef = OpenAIEmbeddingFunction(
    model_name='text-embedding-ada-002',
    api_key='your-api-key'
)
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=1536, distance='cosine'),
    embedding_function=ef
)
```

### 6.3 Embedding Function Requirements

When creating a custom embedding function, ensure:

1. **Implement `__call__` method:**
   - Accepts: `str` or `List[str]` (single document or list of documents)
   - Returns: `List[List[float]]` (list of embedding vectors)
   - Each vector must have the same dimension

2. **Implement `dimension` property (recommended):**
   - Returns: `int` (the dimension of vectors produced by this function)
   - This helps validate dimension consistency when creating collections

3. **Handle edge cases:**
   - Single string input should be converted to list
   - Empty input should return empty list
   - All vectors in the output must have the same dimension

### 6.4 Using Custom Embedding Functions

Once you've created a custom embedding function, use it when creating or getting collections:

```python
# Create collection with custom embedding function
ef = MyCustomEmbeddingFunction()
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=ef.dimension, distance='cosine'),
    embedding_function=ef
)

# Get collection with custom embedding function
collection = client.get_collection("my_collection", embedding_function=ef)

# Use the collection - documents will be automatically embedded
collection.add(
    ids=["doc1", "doc2"],
    documents=["Document 1", "Document 2"],  # Vectors auto-generated
    metadatas=[{"tag": "A"}, {"tag": "B"}]
)

# Query with texts - query vectors auto-generated
results = collection.query(
    query_texts=["my query"],
    n_results=10
)
```

## Testing

```bash
# Run all tests
python3 -m pytest pyseekdb/tests/ -v

# Run tests with log output
python3 -m pytest pyseekdb/tests/ -v -s

# Run specific test
python3 -m pytest pyseekdb/tests/test_client_creation.py::TestClientCreation::test_create_server_client -v

# Run specific test file
python3 -m pytest pyseekdb/tests/test_client_creation.py -v
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

This package is licensed under Apache 2.0.
