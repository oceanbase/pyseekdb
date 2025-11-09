"""
Simple Example: Basic usage of SeekDBClient

This example demonstrates the most common operations:
1. Create a client connection
2. Create a collection
3. Add data to the collection
4. Query the collection
5. Print query results

This is a minimal example to get you started quickly.
"""
import uuid
import seekdbclient

# ==================== Step 1: Create Client Connection ====================
# You can use embedded mode, server mode, or OceanBase mode
# For this example, we'll use server mode (you can change to embedded or OceanBase)

# Server mode (connecting to remote SeekDB server)
client = seekdbclient.Client(
    host="127.0.0.1",
    port=2881,
    database="test",
    user="root",
    password=""
)

# Alternative: Embedded mode (local SeekDB)
# client = seekdbclient.Client(
#     path="./seekdb",
#     database="test"
# )

# Alternative: OceanBase mode
# client = seekdbclient.OBClient(
#     host="127.0.0.1",
#     port=11402,
#     tenant="mysql",
#     database="test",
#     user="root",
#     password=""
# )

# ==================== Step 2: Create a Collection ====================
# A collection is like a table that stores documents with vector embeddings
collection_name = "my_simple_collection"
dimension = 128  # Vector dimension (must match your embedding model)

# Create collection
collection = client.create_collection(
    name=collection_name,
    dimension=dimension
)

# ==================== Step 3: Add Data to Collection ====================
# Generate some sample data
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Vector databases enable semantic search",
    "Neural networks are inspired by the human brain",
    "Natural language processing helps computers understand text"
]

# Generate simple vectors (in real usage, you would use an embedding model)
# For demonstration, we'll create random vectors
import random
random.seed(42)  # For reproducibility

vectors = []
for i in range(len(documents)):
    # Generate a random vector of dimension 128
    vector = [random.random() for _ in range(dimension)]
    vectors.append(vector)

# Generate unique IDs for each document
ids = [str(uuid.uuid4()) for _ in documents]

# Add data to collection
collection.add(
    ids=ids,
    documents=documents,
    vectors=vectors,
    metadatas=[
        {"category": "AI", "index": 0},
        {"category": "Programming", "index": 1},
        {"category": "Database", "index": 2},
        {"category": "AI", "index": 3},
        {"category": "NLP", "index": 4}
    ]
)

# ==================== Step 4: Query the Collection ====================
# Create a query vector (in real usage, you would embed your query text)
# For demonstration, we'll use a vector similar to the first document
query_vector = vectors[0]  # Query with vector similar to first document

# Perform vector similarity search
results = collection.query(
    query_embeddings=query_vector,
    n_results=3  # Return top 3 most similar documents
)


# ==================== Step 5: Print Query Results ====================
print(f"Query results: {len(results)} items found")
for i, item in enumerate(results, 1):
    print(f"Result {i}: ID={item._id}, Distance={item.distance:.4f}, Document={item.document[:50]}...")

# ==================== Step 6: Delete the Collection ====================
client.delete_collection(collection_name)
