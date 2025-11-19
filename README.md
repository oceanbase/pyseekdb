# pyseekdb

pyseekdb aims to provide developers with simple and easy-to-use APIs, reducing the learning curve and entry barriers. Compared to relational databases, database systems like MongoDB, ES, and Milvus simplify their main operations into KV operations in their Client APIs, making them more beginner-friendly. This SDK provides efficient and easy-to-use APIs for applications to access seekdb and OceanBase's AI-related features. Advanced users can still use MySQL-compatible drivers to directly manipulate database objects in seekdb and OceanBase through SQL statements.

To achieve the above design goals, this SDK follows the following design principles:
1. Fixed data model with schema-free interfaces. For beginners and application prototype development, users do not need to explicitly define relational table structures
2. The main data managed is text (or text fragments) and their attributes
3. All data operations are single Collection operations, cross-collection operations are not supported
4. Each record stores and manages **one** text

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
pip install -U pyseekdb
```

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

# Execute SQL queries
rows = client.execute("SELECT 1")
print(rows)
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

## 2. AdminClient Connection and Database Management

The `AdminClient` class provides database management operations. It uses the same connection modes as `Client` but only exposes database management methods.

### 2.1 Embedded/Server AdminClient

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


### 2.2 AdminClient Methods

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

### 2.4 Database Object

The `get_database()` and `list_databases()` methods return `Database` objects with the following properties:

- `name` (str): Database name
- `tenant` (str, optional): Tenant name (None for embedded/server mode)
- `charset` (str, optional): Character set
- `collation` (str, optional): Collation
- `metadata` (dict): Additional metadata

## 3. Collection (Table) Management

Collections are the primary data structures in pyseekdb, similar to tables in traditional databases. Each collection stores documents with vector embeddings, metadata, and full-text search capabilities.

### 3.1 Creating a Collection

```python
import pyseekdb
from pyseekdb import DefaultEmbeddingFunction, HNSWConfiguration

# Create a client
client = pyseekdb.Client(host="127.0.0.1", port=2881, database="test")

# Create a collection
collection = client.create_collection(
    name="my_collection",
    # embedding_function=DefaultEmbeddingFunction()  # Uses default model (384 dimensions)
)

# Create a collection with custom embedding function
ef = UserDefinedEmbeddingFunction(model_name='all-MiniLM-L6-v2') #UserDefinedEmbeddingFunction should be defined before
config = HNSWConfiguration(dimension=384, distance='cosine')  # Must match EF dimension
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=ef
)

# Create a collection without embedding function (embeddings must be provided manually)
collection = client.create_collection(
    name="my_collection",
    configuration=HNSWConfiguration(dimension=128, distance='cosine'),
    embedding_function=None  # Explicitly disable embedding function
)

# Get or create collection (creates if doesn't exist)
collection = client.get_or_create_collection(
    name="my_collection",
)
```

**Parameters:**
- `name` (str): Collection name (required)
- `configuration` (HNSWConfiguration, optional): Index configuration with dimension and distance metric
  - If not provided, uses default (dimension=384, distance='cosine')
  - If set to `None`, dimension will be calculated from `embedding_function`
- `embedding_function` (EmbeddingFunction, optional): Function to convert documents to embeddings
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
  - **Important:** The embedding function set here will be used for all operations on this collection (add, upsert, update, query, hybrid_search) when documents/texts are provided without embeddings

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
- `distance` (str): Distance metric used by the index (e.g., 'l2', 'cosine', 'inner_product')
- `metadata` (dict): Collection metadata

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

1. **If `embeddings` are provided:** Embeddings are used directly, `embedding_function` is NOT called (even if provided)
2. **If `embeddings` are NOT provided but `documents` are provided:**
   - If collection has an `embedding_function` (set during creation or retrieval), it will automatically generate embeddings from documents
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `embeddings` nor `documents` are provided:** A `ValueError` will be raised

```python
# Add single item with embeddings (embedding_function not used)
collection.add(
    ids="item1",
    embeddings=[0.1, 0.2, 0.3],
    documents="This is a document",
    metadatas={"category": "AI", "score": 95}
)

# Add multiple items with embeddings (embedding_function not used)
collection.add(
    ids=["item1", "item2", "item3"],
    embeddings=[
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

# Add with only embeddings (no documents)
collection.add(
    ids=["vec1", "vec2"],
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
)

# Add with only documents - embeddings auto-generated by embedding_function
# Requires: collection must have embedding_function set
collection.add(
    ids=["doc1", "doc2"],
    documents=["Text document 1", "Text document 2"],
    metadatas=[{"tag": "A"}, {"tag": "B"}]
)
# The collection's embedding_function will automatically convert documents to embeddings
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs (required)
- `embeddings` (List[float] or List[List[float]], optional): Single embedding or list of embeddings
  - If provided, used directly (embedding_function is ignored)
  - If not provided, must provide `documents` and collection must have `embedding_function`
- `documents` (str or List[str], optional): Single document or list of documents
  - If `embeddings` not provided, `documents` will be converted to embeddings using collection's `embedding_function`
- `metadatas` (dict or List[dict], optional): Single metadata dict or list of metadata dicts

**Note:** The `embedding_function` used is the one associated with the collection (set during `create_collection()` or `get_collection()`). You cannot override it per-operation.

### 4.2 Update Data

The `update()` method updates existing records in a collection. Records must exist, otherwise an error will be raised.

**Behavior with Embedding Function:**

1. **If `embeddings` are provided:** Embeddings are used directly, `embedding_function` is NOT called
2. **If `embeddings` are NOT provided but `documents` are provided:**
   - If collection has an `embedding_function`, it will automatically generate embeddings from documents
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `embeddings` nor `documents` are provided:** Only metadata will be updated (metadata-only update is allowed)

```python
# Update single item - metadata only (embedding_function not used)
collection.update(
    ids="item1",
    metadatas={"category": "AI", "score": 98}  # Update metadata only
)

# Update multiple items with embeddings (embedding_function not used)
collection.update(
    ids=["item1", "item2"],
    embeddings=[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],  # Update embeddings
    documents=["Updated document 1", "Updated document 2"]  # Update documents
)

# Update with documents only - embeddings auto-generated by embedding_function
# Requires: collection must have embedding_function set
collection.update(
    ids="item1",
    documents="New document text",  # Embeddings will be auto-generated
    metadatas={"category": "AI"}
)

# Update specific fields - only document (embeddings auto-generated)
collection.update(
    ids="item1",
    documents="New document text"  # Only update document, embeddings auto-generated
)
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs to update (required)
- `embeddings` (List[float] or List[List[float]], optional): New embeddings
  - If provided, used directly (embedding_function is ignored)
  - If not provided, can provide `documents` to auto-generate embeddings
- `documents` (str or List[str], optional): New documents
  - If `embeddings` not provided, `documents` will be converted to embeddings using collection's `embedding_function`
- `metadatas` (dict or List[dict], optional): New metadata

**Note:** Metadata-only updates (no embeddings, no documents) are allowed. The `embedding_function` used is the one associated with the collection.

### 4.3 Upsert Data

The `upsert()` method inserts new records or updates existing ones. If a record with the given ID exists, it will be updated; otherwise, a new record will be inserted.

**Behavior with Embedding Function:**

1. **If `embeddings` are provided:** Embeddings are used directly, `embedding_function` is NOT called
2. **If `embeddings` are NOT provided but `documents` are provided:**
   - If collection has an `embedding_function`, it will automatically generate embeddings from documents
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `embeddings` nor `documents` are provided:** Only metadata will be upserted (metadata-only upsert is allowed)

```python
# Upsert single item with embeddings (embedding_function not used)
collection.upsert(
    ids="item1",
    embeddings=[0.1, 0.2, 0.3],
    documents="Document text",
    metadatas={"category": "AI", "score": 95}
)

# Upsert multiple items with embeddings (embedding_function not used)
collection.upsert(
    ids=["item1", "item2", "item3"],
    embeddings=[
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

# Upsert with documents only - embeddings auto-generated by embedding_function
# Requires: collection must have embedding_function set
collection.upsert(
    ids=["item1", "item2"],
    documents=["Document 1", "Document 2"],
    metadatas=[{"category": "AI"}, {"category": "ML"}]
)
# The collection's embedding_function will automatically convert documents to embeddings
```

**Parameters:**
- `ids` (str or List[str]): Single ID or list of IDs (required)
- `embeddings` (List[float] or List[List[float]], optional): Embeddings
  - If provided, used directly (embedding_function is ignored)
  - If not provided, can provide `documents` to auto-generate embeddings
- `documents` (str or List[str], optional): Documents
  - If `embeddings` not provided, `documents` will be converted to embeddings using collection's `embedding_function`
- `metadatas` (dict or List[dict], optional): Metadata

**Note:** Metadata-only upserts (no embeddings, no documents) are allowed. The `embedding_function` used is the one associated with the collection.

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

1. **If `query_embeddings` are provided:** embeddings are used directly, `embedding_function` is NOT called
2. **If `query_embeddings` are NOT provided but `query_texts` are provided:**
   - If collection has an `embedding_function`, it will automatically generate query embeddings from texts
   - If collection does NOT have an `embedding_function`, a `ValueError` will be raised
3. **If neither `query_embeddings` nor `query_texts` are provided:** A `ValueError` will be raised

```python
# Basic vector similarity query (embedding_function not used)
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    n_results=3
)

# Iterate over results
for i in range(len(results["ids"][0])):
    print(f"ID: {results['ids'][0][i]}, Distance: {results['distances'][0][i]}")
    if results.get("documents"):
        print(f"Document: {results['documents'][0][i]}")
    if results.get("metadatas"):
        print(f"Metadata: {results['metadatas'][0][i]}")

# Query by texts - embeddings auto-generated by embedding_function
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
# Returns dict with lists of lists, one list per query text
for i in range(len(results["ids"])):
    print(f"Query {i}: {len(results['ids'][i])} results")

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

# Query with multiple embeddings (batch query)
results = collection.query(
    query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    n_results=2
)
# Returns dict with lists of lists, one list per query embedding
for i in range(len(results["ids"])):
    print(f"Query {i}: {len(results['ids'][i])} results")

# Query with specific fields
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    include=["documents", "metadatas", "embeddings"],
    n_results=3
)
```

**Parameters:**
- `query_embeddings` (List[float] or List[List[float]], optional): Single embedding or list of embeddings for batch queries
  - If provided, used directly (embedding_function is ignored)
  - If not provided, must provide `query_texts` and collection must have `embedding_function`
- `query_texts` (str or List[str], optional): Query text(s) to be embedded
  - If `query_embeddings` not provided, `query_texts` will be converted to embeddings using collection's `embedding_function`
- `n_results` (int, required): Number of similar results to return (default: 10)
- `where` (dict, optional): Metadata filter conditions (see Filter Operators section)
- `where_document` (dict, optional): Document content filter
- `include` (List[str], optional): List of fields to include: `["documents", "metadatas", "embeddings"]`

**Returns:**
Dict with keys (chromadb-compatible format):
- `ids`: `List[List[str]]` - List of ID lists, one list per query
- `documents`: `Optional[List[List[str]]]` - List of document lists, one list per query (if included)
- `metadatas`: `Optional[List[List[Dict]]]` - List of metadata lists, one list per query (if included)
- `embeddings`: `Optional[List[List[List[float]]]]` - List of embedding lists, one list per query (if included)
- `distances`: `Optional[List[List[float]]]` - List of distance lists, one list per query

**Usage:**
```python
# Single query
results = collection.query(query_embeddings=[0.1, 0.2, 0.3], n_results=5)
# results["ids"][0] contains IDs for the query
# results["documents"][0] contains documents for the query
# results["distances"][0] contains distances for the query

# Multiple queries
results = collection.query(query_embeddings=[[0.1, 0.2], [0.3, 0.4]], n_results=5)
# results["ids"][0] contains IDs for first query
# results["ids"][1] contains IDs for second query
```

**Note:** The `embedding_function` used is the one associated with the collection. You cannot override it per-query.

### 5.2 Get (Retrieve by IDs or Filters)

The `get()` method retrieves documents from a collection without vector similarity search. It supports filtering by IDs, metadata, and document content.

```python
# Get by single ID
results = collection.get(ids="123")

# Get by multiple IDs
results = collection.get(ids=["1", "2", "3"])

# Get by metadata filter (simplified equality - both forms are supported)
results = collection.get(
    where={"category": "AI"},
    limit=10
)
# Or use explicit $eq operator:
# where={"category": {"$eq": "AI"}}

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

# Get by logical operators ($or) - simplified equality
results = collection.get(
    where={
        "$or": [
            {"category": "AI"},
            {"tag": "python"}
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
Dict with keys (chromadb-compatible format):
- `ids`: `List[str]` - List of IDs
- `documents`: `Optional[List[str]]` - List of documents (if included)
- `metadatas`: `Optional[List[Dict]]` - List of metadata dictionaries (if included)
- `embeddings`: `Optional[List[List[float]]]` - List of embeddings (if included)

**Usage:**
```python
# Get by single ID
results = collection.get(ids="123")
# results["ids"] contains ["123"]
# results["documents"] contains document for ID "123"

# Get by multiple IDs
results = collection.get(ids=["1", "2", "3"])
# results["ids"] contains ["1", "2", "3"]
# results["documents"] contains documents for all IDs

# Get by filter
results = collection.get(where={"category": {"$eq": "AI"}}, limit=10)
# results["ids"] contains all matching IDs
# results["documents"] contains all matching documents
```

**Note:** If no parameters provided, returns all data (up to limit).

### 5.3 Hybrid Search

The `hybrid_search()` method combines full-text search and vector similarity search with ranking.

**Behavior with Embedding Function:**

In the `knn` parameter:
1. **If `query_embeddings` are provided:** embeddings are used directly, `embedding_function` is NOT called
2. **If `query_embeddings` are NOT provided but `query_texts` are provided:**
   - If collection has an `embedding_function`, it will automatically generate query embeddings from texts
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
    - If `query_embeddings` not provided, `query_texts` will be converted to embeddings using collection's `embedding_function`
  - `query_embeddings` (List[float] or List[List[float]], optional): Query vector(s)
    - If provided, used directly (embedding_function is ignored)
  - `where`: Metadata filter conditions (optional)
  - `n_results`: Number of results for vector search (optional)
- `rank` (dict, optional): Ranking configuration (e.g., `{"rrf": {"rank_window_size": 60, "rank_constant": 60}}`)
- `n_results` (int): Final number of results to return after ranking (default: 10)
- `include` (List[str], optional): Fields to include in results

**Returns:**
Dict with keys (query-compatible format):
- `ids`: `List[List[str]]` - List of ID lists (one list for hybrid search result)
- `documents`: `Optional[List[List[str]]]` - List of document lists (if included)
- `metadatas`: `Optional[List[List[Dict]]]` - List of metadata lists (if included)
- `embeddings`: `Optional[List[List[List[float]]]]` - List of embedding lists (if included)
- `distances`: `Optional[List[List[float]]]` - List of distance lists

**Usage:**
```python
# Hybrid search returns results in query-compatible format
results = collection.hybrid_search(
    query={"where_document": {"$contains": "machine learning"}},
    knn={"query_texts": ["AI research"]},
    rank={"rrf": {}},
    n_results=5
)
# results["ids"][0] contains IDs for the hybrid search
# results["documents"][0] contains documents for the hybrid search
# results["distances"][0] contains distances for the hybrid search
```

**Note:** The `embedding_function` used is the one associated with the collection. You cannot override it per-search.

### 5.4 Filter Operators

#### Metadata Filters (`where` parameter)

- `$eq`: Equal to (simplified form is also supported)
  ```python
  # Simplified form (recommended for equality)
  where={"category": "AI"}
  
  # Explicit $eq operator (also supported)
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
  # Simplified equality form
  where={
      "$or": [
          {"category": "AI"},
          {"tag": "python"}
      ]
  }
  # Or with explicit $eq:
  # where={"$or": [{"category": {"$eq": "AI"}}, {"tag": {"$eq": "python"}}]}
  ```

- `$and`: Logical AND
  ```python
  # Simplified equality form
  where={
      "$and": [
          {"category": "AI"},
          {"score": {"$gte": 90}}
      ]
  }
  # Or with explicit $eq:
  # where={"$and": [{"category": {"$eq": "AI"}}, {"score": {"$gte": 90}}]}
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

# Preview first few items in collection (returns all columns by default)
preview = collection.peek(limit=5)
for i in range(len(preview["ids"])):
    print(f"ID: {preview['ids'][i]}, Document: {preview['documents'][i]}")
    print(f"Metadata: {preview['metadatas'][i]}, Embedding: {preview['embeddings'][i]}")

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

Embedding functions convert text documents into vector embeddings for similarity search. pyseekdb supports both built-in and custom embedding functions.

### 6.1 Default Embedding Function

The `DefaultEmbeddingFunction` uses all-MiniLM-L6-v2' and is the default embedding function if none is specified.

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

### 6.2 Creating Custom Embedding Functions

You can create custom embedding functions by implementing the `EmbeddingFunction` protocol. The function must:

1. Implement `__call__` method that accepts `Documents` (str or List[str]) and returns `Embeddings` (List[List[float]])
2. Optionally implement a `dimension` property to return the vector dimension

#### Example: Sentence-Transformer Custom Embedding Function

```python
from typing import List, Union
from pyseekdb import EmbeddingFunction

Documents = Union[str, List[str]]
Embeddings = List[List[float]]
Embedding = List[float]

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
            List of embedding embeddings
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

#### Example: OpenAI Embedding Function

```python
from typing import List, Union
import os
import openai
from pyseekdb import EmbeddingFunction

Documents = Union[str, List[str]]
Embeddings = List[List[float]]
Embedding = List[float]

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
            List of embedding embeddings
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
   - Returns: `List[List[float]]` (list of embeddings)
   - Each vector must have the same dimension

2. **Implement `dimension` property (recommended):**
   - Returns: `int` (the dimension of embeddings produced by this function)
   - This helps validate dimension consistency when creating collections

3. **Handle edge cases:**
   - Single string input should be converted to list
   - Empty input should return empty list
   - All embeddings in the output must have the same dimension

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
    documents=["Document 1", "Document 2"],  # Embeddings auto-generated
    metadatas=[{"tag": "A"}, {"tag": "B"}]
)

# Query with texts - query embeddings auto-generated
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

# Run all seekdb-embeded, seekdb-server, oceanbase cases
python3 -m pytest pyseekdb/tests/ -k test_{embedded, server, oceanbase}

# Run specific test
python3 -m pytest pyseekdb/tests/test_client_creation.py::TestClientCreation::test_create_server_client -v

# Run specific test file
python3 -m pytest pyseekdb/tests/test_client_creation.py -v
```

## License

This package is licensed under Apache 2.0.
