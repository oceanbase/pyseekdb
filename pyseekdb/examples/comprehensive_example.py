"""
Comprehensive Example: Complete guide to all SeekDBClient features

This example demonstrates all available operations:
1. Client connection (all modes)
2. Collection management
3. DML operations (add, update, upsert, delete)
4. DQL operations (query, get, hybrid_search)
5. Filter operators
6. Collection information methods

This is a complete reference for all client capabilities.
"""
import uuid
import random
import pyseekdb

# ============================================================================
# PART 1: CLIENT CONNECTION
# ============================================================================

# Option 1: Embedded mode (local SeekDB)
# client = pyseekdb.Client(
#     path="./seekdb",
#     database="test"
# )

# Option 2: Server mode (remote SeekDB server)
client = pyseekdb.Client(
    host="127.0.0.1",
    port=2881,
    database="test",
    user="root",
    password=""
)

# Option 3: OceanBase mode
# ob_client = pyseekdb.OBClient(
#     host="127.0.0.1",
#     port=11402,
#     tenant="mysql",
#     database="test",
#     user="root",
#     password=""
# )

# ============================================================================
# PART 2: COLLECTION MANAGEMENT
# ============================================================================

collection_name = "comprehensive_example"
dimension = 128

# 2.1 Create a collection
from pyseekdb import HNSWConfiguration
config = HNSWConfiguration(dimension=dimension, distance='cosine')
collection = client.get_or_create_collection(
    name=collection_name,
    configuration=config,
    embedding_function=None  # Explicitly set to None since we're using custom 128-dim vectors
)

# 2.2 Check if collection exists
exists = client.has_collection(collection_name)

# 2.3 Get collection object
retrieved_collection = client.get_collection(collection_name, embedding_function=None)

# 2.4 List all collections
all_collections = client.list_collections()

# 2.5 Get or create collection (creates if doesn't exist)
config2 = HNSWConfiguration(dimension=64, distance='cosine')
collection2 = client.get_or_create_collection(
    name="another_collection",
    configuration=config2,
    embedding_function=None  # Explicitly set to None since we're using custom 64-dim vectors
)

# ============================================================================
# PART 3: DML OPERATIONS - ADD DATA
# ============================================================================

# Generate sample data
random.seed(42)
documents = [
    "Machine learning is transforming the way we solve problems",
    "Python programming language is widely used in data science",
    "Vector databases enable efficient similarity search",
    "Neural networks mimic the structure of the human brain",
    "Natural language processing helps computers understand human language",
    "Deep learning requires large amounts of training data",
    "Reinforcement learning agents learn through trial and error",
    "Computer vision enables machines to interpret visual information"
]

# Generate vectors (in real usage, use an embedding model)
vectors = []
for i in range(len(documents)):
    vector = [random.random() for _ in range(dimension)]
    vectors.append(vector)

ids = [str(uuid.uuid4()) for _ in documents]

# 3.1 Add single item
single_id = str(uuid.uuid4())
collection.add(
    ids=single_id,
    documents="This is a single document",
    vectors=[random.random() for _ in range(dimension)],
    metadatas={"type": "single", "category": "test"}
)

# 3.2 Add multiple items
collection.add(
    ids=ids,
    documents=documents,
    vectors=vectors,
    metadatas=[
        {"category": "AI", "score": 95, "tag": "ml", "year": 2023},
        {"category": "Programming", "score": 88, "tag": "python", "year": 2022},
        {"category": "Database", "score": 92, "tag": "vector", "year": 2023},
        {"category": "AI", "score": 90, "tag": "neural", "year": 2022},
        {"category": "NLP", "score": 87, "tag": "language", "year": 2023},
        {"category": "AI", "score": 93, "tag": "deep", "year": 2023},
        {"category": "AI", "score": 85, "tag": "reinforcement", "year": 2022},
        {"category": "CV", "score": 91, "tag": "vision", "year": 2023}
    ]
)

# 3.3 Add with only vectors (no documents)
vector_only_ids = [str(uuid.uuid4()) for _ in range(2)]
collection.add(
    ids=vector_only_ids,
    vectors=[[random.random() for _ in range(dimension)] for _ in range(2)],
    metadatas=[{"type": "vector_only"}, {"type": "vector_only"}]
)

# ============================================================================
# PART 4: DML OPERATIONS - UPDATE DATA
# ============================================================================

# 4.1 Update single item
collection.update(
    ids=ids[0],
    metadatas={"category": "AI", "score": 98, "tag": "ml", "year": 2024, "updated": True}
)

# 4.2 Update multiple items
collection.update(
    ids=ids[1:3],
    documents=["Updated document 1", "Updated document 2"],
    vectors=[[random.random() for _ in range(dimension)] for _ in range(2)],
    metadatas=[
        {"category": "Programming", "score": 95, "updated": True},
        {"category": "Database", "score": 97, "updated": True}
    ]
)

# 4.3 Update vectors
new_vectors = [[random.random() for _ in range(dimension)] for _ in range(2)]
collection.update(
    ids=ids[2:4],
    vectors=new_vectors
)

# ============================================================================
# PART 5: DML OPERATIONS - UPSERT DATA
# ============================================================================

# 5.1 Upsert existing item (will update)
collection.upsert(
    ids=ids[0],
    documents="Upserted document (was updated)",
    vectors=[random.random() for _ in range(dimension)],
    metadatas={"category": "AI", "upserted": True}
)

# 5.2 Upsert new item (will insert)
new_id = str(uuid.uuid4())
collection.upsert(
    ids=new_id,
    documents="This is a new document from upsert",
    vectors=[random.random() for _ in range(dimension)],
    metadatas={"category": "New", "upserted": True}
)

# 5.3 Upsert multiple items
upsert_ids = [ids[4], str(uuid.uuid4())]  # One existing, one new
collection.upsert(
    ids=upsert_ids,
    documents=["Upserted doc 1", "Upserted doc 2"],
    vectors=[[random.random() for _ in range(dimension)] for _ in range(2)],
    metadatas=[{"upserted": True}, {"upserted": True}]
)

# ============================================================================
# PART 6: DQL OPERATIONS - QUERY (VECTOR SIMILARITY SEARCH)
# ============================================================================

# 6.1 Basic vector similarity query
query_vector = vectors[0]  # Query with first document's vector
results = collection.query(
    query_embeddings=query_vector,
    n_results=3
)
print(f"Query results: {len(results)} items")

# 6.2 Query with metadata filter
results = collection.query(
    query_embeddings=query_vector,
    where={"category": {"$eq": "AI"}},
    n_results=5
)

# 6.3 Query with comparison operators
results = collection.query(
    query_embeddings=query_vector,
    where={"score": {"$gte": 90}},
    n_results=5
)

# 6.4 Query with $in operator
results = collection.query(
    query_embeddings=query_vector,
    where={"tag": {"$in": ["ml", "python", "neural"]}},
    n_results=5
)

# 6.5 Query with logical operators ($or)
results = collection.query(
    query_embeddings=query_vector,
    where={
        "$or": [
            {"category": {"$eq": "AI"}},
            {"tag": {"$eq": "python"}}
        ]
    },
    n_results=5
)

# 6.6 Query with logical operators ($and)
results = collection.query(
    query_embeddings=query_vector,
    where={
        "$and": [
            {"category": {"$eq": "AI"}},
            {"score": {"$gte": 90}}
        ]
    },
    n_results=5
)

# 6.7 Query with document filter
results = collection.query(
    query_embeddings=query_vector,
    where_document={"$contains": "machine learning"},
    n_results=5
)

# 6.8 Query with combined filters
results = collection.query(
    query_embeddings=query_vector,
    where={"category": {"$eq": "AI"}, "year": {"$gte": 2023}},
    where_document={"$contains": "learning"},
    n_results=5
)

# 6.9 Query with multiple vectors (batch query)
batch_vectors = [vectors[0], vectors[1]]
batch_results = collection.query(
    query_embeddings=batch_vectors,
    n_results=2
)

# 6.10 Query with specific fields
results = collection.query(
    query_embeddings=query_vector,
    include=["documents", "metadatas", "embeddings"],
    n_results=2
)

# ============================================================================
# PART 7: DQL OPERATIONS - GET (RETRIEVE BY IDS OR FILTERS)
# ============================================================================

# 7.1 Get by single ID
result = collection.get(ids=ids[0])

# 7.2 Get by multiple IDs
results = collection.get(ids=ids[:3])

# 7.3 Get by metadata filter
results = collection.get(
    where={"category": {"$eq": "AI"}},
    limit=5
)

# 7.4 Get with comparison operators
results = collection.get(
    where={"score": {"$gte": 90}},
    limit=5
)

# 7.5 Get with $in operator
results = collection.get(
    where={"tag": {"$in": ["ml", "python"]}},
    limit=5
)

# 7.6 Get with logical operators
results = collection.get(
    where={
        "$or": [
            {"category": {"$eq": "AI"}},
            {"category": {"$eq": "Programming"}}
        ]
    },
    limit=5
)

# 7.7 Get by document filter
results = collection.get(
    where_document={"$contains": "Python"},
    limit=5
)

# 7.8 Get with pagination
results_page1 = collection.get(limit=2, offset=0)
results_page2 = collection.get(limit=2, offset=2)

# 7.9 Get with specific fields
results = collection.get(
    ids=ids[:2],
    include=["documents", "metadatas", "embeddings"]
)

# 7.10 Get all data
all_results = collection.get(limit=100)

# ============================================================================
# PART 8: DQL OPERATIONS - HYBRID SEARCH
# ============================================================================

# 8.1 Hybrid search with full-text and vector search
# Note: This requires query_embeddings to be provided directly
# In real usage, you might have an embedding function
hybrid_results = collection.hybrid_search(
    query={
        "where_document": {"$contains": "machine learning"},
        "where": {"category": {"$eq": "AI"}},
        "n_results": 10
    },
    knn={
        "query_embeddings": [vectors[0]],
        "where": {"year": {"$gte": 2022}},
        "n_results": 10
    },
    rank={"rrf": {}},  # Reciprocal Rank Fusion
    n_results=5,
    include=["documents", "metadatas"]
)
print(f"Hybrid search: {len(hybrid_results.get('ids', []))} results")

# ============================================================================
# PART 9: DML OPERATIONS - DELETE DATA
# ============================================================================

# 9.1 Delete by IDs
delete_ids = [vector_only_ids[0], new_id]
collection.delete(ids=delete_ids)

# 9.2 Delete by metadata filter
collection.delete(where={"type": {"$eq": "vector_only"}})

# 9.3 Delete by document filter
collection.delete(where_document={"$contains": "Updated document"})

# 9.4 Delete with combined filters
collection.delete(
    where={"category": {"$eq": "CV"}},
    where_document={"$contains": "vision"}
)

# ============================================================================
# PART 10: COLLECTION INFORMATION
# ============================================================================

# 10.1 Get collection count
count = collection.count()
print(f"Collection count: {count} items")


# 10.3 Preview first few items in collection
preview = collection.peek(limit=5)
print(f"Preview: {len(preview)} items")
for item in preview:
    print(f"  ID: {item._id}, Document: {item.document}")

# 10.4 Count collections in database
collection_count = client.count_collection()
print(f"Database has {collection_count} collections")

# ============================================================================
# PART 11: CLEANUP
# ============================================================================

# Delete test collections
try:
    client.delete_collection("another_collection")
except Exception as e:
    print(f"Could not delete 'another_collection': {e}")

# Uncomment to delete main collection
client.delete_collection(collection_name)
