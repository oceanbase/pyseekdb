# Collection Management

## 3. Collection (Table) Management

Collections are the primary data structures in pyseekdb, similar to tables in traditional databases. Each collection stores documents with vector embeddings, metadata, and full-text search capabilities.

### 3.1 Creating a Collection

```python
import pyseekdb
from pyseekdb import (
    DefaultEmbeddingFunction,
    HNSWConfiguration,
    Configuration,
    FulltextAnalyzerConfig
)

# Create a client
client = pyseekdb.Client(host="127.0.0.1", port=2881, database="test")

# Create a collection with default configuration
collection = client.create_collection(
    name="my_collection"
    # embedding_function defaults to DefaultEmbeddingFunction() (384 dimensions)
)

# Create a collection with custom embedding function
# Dimension will be automatically calculated from embedding function
ef = UserDefinedEmbeddingFunction(model_name='all-MiniLM-L6-v2')
collection = client.create_collection(
    name="my_collection",
    embedding_function=ef
)

# Recommended: Create a collection with Configuration wrapper
# Using IK parser (default for Chinese text)
config = Configuration(
    hnsw=HNSWConfiguration(dimension=384, distance='cosine'),
    fulltext_config=FulltextAnalyzerConfig(analyzer='ik')
)
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=ef
)

# Recommended: Create a collection with Configuration (only HNSW config, uses default parser)
config = Configuration(
    hnsw=HNSWConfiguration(dimension=384, distance='cosine')
)
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=ef
)

# Create a collection with Space parser (for space-separated languages)
config = Configuration(
    hnsw=HNSWConfiguration(dimension=384, distance='cosine'),
    fulltext_config=FulltextAnalyzerConfig(analyzer='space')
)
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=ef
)

# Create a collection with Ngram parser and custom parameters
config = Configuration(
    hnsw=HNSWConfiguration(dimension=384, distance='cosine'),
    fulltext_config=FulltextAnalyzerConfig(analyzer='ngram', properties={'ngram_token_size': 3})
)
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=ef
)

# Create a collection without embedding function (embeddings must be provided manually)
# Recommended: Use Configuration wrapper
config = Configuration(
    hnsw=HNSWConfiguration(dimension=128, distance='cosine')
)
collection = client.create_collection(
    name="my_collection",
    configuration=config,
    embedding_function=None  # Explicitly disable embedding function
)

# Get or create collection (creates if doesn't exist)
collection = client.get_or_create_collection(
    name="my_collection",
)
```

**Parameters:**
- `name` (str): Collection name (required). Must be non-empty, use only letters/digits/underscore (`[a-zA-Z0-9_]`), and be at most 512 characters.
- `configuration` (Configuration, HNSWConfiguration, or None, optional): Index configuration
  - **Recommended:** `Configuration` - Wrapper class that can include both `HNSWConfiguration` and `FulltextAnalyzerConfig`
    - Use `Configuration(hnsw=HNSWConfiguration(...))` even when only vector index config is needed
    - Allows easy addition of fulltext parser config later
  - `HNSWConfiguration`: Vector index configuration with `dimension` and `distance` metric (backward compatibility)
  - If not provided, uses default (dimension=384, distance='cosine', analyzer='ik')
  - If set to `None`, dimension will be calculated from `embedding_function`
- `embedding_function` (EmbeddingFunction, optional): Function to convert documents to embeddings
  - If not provided, uses `DefaultEmbeddingFunction()` (384 dimensions)
  - If set to `None`, collection will not have an embedding function
  - If provided, the dimension will be automatically calculated and validated against `configuration.dimension`

**Fulltext Parser Options:**
- `'ik'` (default): IK parser for Chinese text segmentation
- `'space'`: Space-separated tokenizer for languages like English
- `'ngram'`: N-gram tokenizer
- `'ngram2'`: 2-gram tokenizer
- `'beng'`: Bengali text parser

For more information about parser, please refer to [create_index section tokenizer_option](https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004479548#tokenizer_option).

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
