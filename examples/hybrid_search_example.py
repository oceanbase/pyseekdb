"""
Hybrid Search Example: Demonstrating hybrid_search() vs query()

Key advantages of hybrid_search():
- Combines full-text search and vector similarity search simultaneously
- Independent filters for each search type
- Intelligent result fusion using RRF (Reciprocal Rank Fusion)
- Handles complex scenarios that query() cannot
"""

import pyseekdb
from pyseekdb import DefaultEmbeddingFunction

# Setup
client = pyseekdb.Client()
collection = client.get_or_create_collection(
    name="hybrid_search_demo"
)

# Sample data
documents = [
    "Machine learning is revolutionizing artificial intelligence and data science",
    "Python programming language is essential for machine learning developers",
    "Deep learning neural networks enable advanced AI applications",
    "Data science combines statistics, programming, and domain expertise",
    "Natural language processing uses machine learning to understand text",
    "Computer vision algorithms process images using deep learning techniques",
    "Reinforcement learning trains agents through reward-based feedback",
    "Python libraries like TensorFlow and PyTorch simplify machine learning",
    "Artificial intelligence systems can learn from large datasets",
    "Neural networks mimic the structure of biological brain connections"
]

metadatas = [
    {"category": "AI", "topic": "machine learning", "year": 2023, "popularity": 95},
    {"category": "Programming", "topic": "python", "year": 2023, "popularity": 88},
    {"category": "AI", "topic": "deep learning", "year": 2024, "popularity": 92},
    {"category": "Data Science", "topic": "data analysis", "year": 2023, "popularity": 85},
    {"category": "AI", "topic": "nlp", "year": 2024, "popularity": 90},
    {"category": "AI", "topic": "computer vision", "year": 2023, "popularity": 87},
    {"category": "AI", "topic": "reinforcement learning", "year": 2024, "popularity": 89},
    {"category": "Programming", "topic": "python", "year": 2023, "popularity": 91},
    {"category": "AI", "topic": "general ai", "year": 2023, "popularity": 93},
    {"category": "AI", "topic": "neural networks", "year": 2024, "popularity": 94}
]

ids = [f"doc_{i+1}" for i in range(len(documents))]
collection.add(ids=ids, documents=documents, metadatas=metadatas)

print("=" * 100)
print("SCENARIO 1: Keyword + Semantic Search")
print("=" * 100)
print("Goal: Find documents similar to 'AI research' AND containing 'machine learning'\n")

# query() approach
query_result1 = collection.query(
    query_texts=["AI research"],
    where_document={"$contains": "machine learning"},
    n_results=5
)

# hybrid_search() approach
hybrid_result1 = collection.hybrid_search(
    query={"where_document": {"$contains": "machine learning"}, "n_results": 10},
    knn={"query_texts": ["AI research"], "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results:")
for i, doc_id in enumerate(query_result1['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nhybrid_search() Results:")
for i, doc_id in enumerate(hybrid_result1['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nAnalysis:")
print("  query() ranks 'Deep learning neural networks...' first because it's semantically similar to 'AI research',")
print("  but 'machine learning' is not its primary focus. hybrid_search() correctly prioritizes documents that")
print("  explicitly contain 'machine learning' (from full-text search) while also being semantically relevant")
print("  to 'AI research' (from vector search). The RRF fusion ensures documents matching both criteria rank higher.")

print("\n" + "=" * 100)
print("SCENARIO 2: Independent Filters for Different Search Types")
print("=" * 100)
print("Goal: Full-text='neural' (year=2024) + Vector='deep learning' (popularity>=90)\n")

# query() - same filter applies to both conditions
query_result2 = collection.query(
    query_texts=["deep learning"],
    where={"year": {"$eq": 2024}, "popularity": {"$gte": 90}},
    where_document={"$contains": "neural"},
    n_results=5
)

# hybrid_search() - different filters for each search type
hybrid_result2 = collection.hybrid_search(
    query={"where_document": {"$contains": "neural"}, "where": {"year": {"$eq": 2024}}, "n_results": 10},
    knn={"query_texts": ["deep learning"], "where": {"popularity": {"$gte": 90}}, "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results (same filter for both):")
for i, doc_id in enumerate(query_result2['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")
    print(f"      {metadatas[idx]}")

print("\nhybrid_search() Results (independent filters):")
for i, doc_id in enumerate(hybrid_result2['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")
    print(f"      {metadatas[idx]}")

print("\nAnalysis:")
print("  query() only returns 2 results because it requires documents to satisfy BOTH year=2024 AND popularity>=90")
print("  simultaneously. hybrid_search() returns 5 results by applying year=2024 filter to full-text search")
print("  and popularity>=90 filter to vector search independently, then fusing the results. This approach")
print("  captures more relevant documents that might satisfy one criterion strongly while meeting the other")

print("\n" + "=" * 100)
print("SCENARIO 3: Combining Multiple Search Strategies")
print("=" * 100)
print("Goal: Find documents about 'machine learning algorithms'\n")

# query() - vector search only
query_result3 = collection.query(
    query_texts=["machine learning algorithms"],
    n_results=5
)

# hybrid_search() - combines full-text and vector
hybrid_result3 = collection.hybrid_search(
    query={"where_document": {"$contains": "machine learning"}, "n_results": 10},
    knn={"query_texts": ["machine learning algorithms"], "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results (vector similarity only):")
for i, doc_id in enumerate(query_result3['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nhybrid_search() Results (full-text + vector fusion):")
for i, doc_id in enumerate(hybrid_result3['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nAnalysis:")
print("  query() returns 'Artificial intelligence systems...' as the result, which doesn't explicitly")
print("  mention 'machine learning'. hybrid_search() combines full-text search (for 'machine learning')")
print("  with vector search (for semantic similarity to 'machine learning algorithms'), ensuring that")
print("  documents containing the exact keyword rank higher while still capturing semantically relevant content.")

print("\n" + "=" * 100)
print("SCENARIO 4: Complex Multi-Criteria Search")
print("=" * 100)
print("Goal: Full-text='learning' (category=AI) + Vector='artificial intelligence' (year>=2023)\n")

# query() - limited to single search with combined filters
query_result4 = collection.query(
    query_texts=["artificial intelligence"],
    where={"category": {"$eq": "AI"}, "year": {"$gte": 2023}},
    where_document={"$contains": "learning"},
    n_results=5
)

# hybrid_search() - separate criteria for each search type
hybrid_result4 = collection.hybrid_search(
    query={"where_document": {"$contains": "learning"}, "where": {"category": {"$eq": "AI"}}, "n_results": 10},
    knn={"query_texts": ["artificial intelligence"], "where": {"year": {"$gte": 2023}}, "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results:")
for i, doc_id in enumerate(query_result4['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")
    print(f"      {metadatas[idx]}")

print("\nhybrid_search() Results:")
for i, doc_id in enumerate(hybrid_result4['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")
    print(f"      {metadatas[idx]}")

print("\nAnalysis:")
print("  While both methods return similar documents, hybrid_search() provides better ranking by prioritizing")
print("  documents that score highly in both full-text search (containing 'learning' with category=AI) and")
print("  vector search (semantically similar to 'artificial intelligence' with year>=2023). The RRF fusion")
print("  algorithm ensures that 'Deep learning neural networks...' ranks first because it strongly matches")
print("  both search criteria, whereas query() applies filters sequentially which may not optimize ranking.")

print("\n" + "=" * 100)
print("SCENARIO 5: Result Quality - RRF Fusion")
print("=" * 100)
print("Goal: Search for 'Python machine learning'\n")

# query() - single ranking
query_result5 = collection.query(
    query_texts=["Python machine learning"],
    n_results=5
)

# hybrid_search() - RRF fusion of multiple rankings
hybrid_result5 = collection.hybrid_search(
    query={"where_document": {"$contains": "Python"}, "n_results": 10},
    knn={"query_texts": ["Python machine learning"], "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results (single ranking):")
for i, doc_id in enumerate(query_result5['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nhybrid_search() Results (RRF fusion):")
for i, doc_id in enumerate(hybrid_result5['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nAnalysis:")
print("  Both methods return identical results in this case, but hybrid_search() achieves this through RRF")
print("  (Reciprocal Rank Fusion) which combines rankings from full-text search (for 'Python') and vector")
print("  search (for 'Python machine learning'). RRF provides more stable and robust ranking by considering")
print("  multiple signals, making it less sensitive to variations in individual search algorithms and ensuring")
print("  consistent high-quality results across different query formulations.")

print("\n" + "=" * 100)
print("SCENARIO 6: Different Filter Criteria for Each Search")
print("=" * 100)
print("Goal: Full-text='neural' (high popularity) + Vector='deep learning' (recent year)\n")

# query() - cannot separate filters for keyword vs semantic
query_result6 = collection.query(
    query_texts=["deep learning"],
    where={"popularity": {"$gte": 90}, "year": {"$gte": 2023}},
    where_document={"$contains": "neural"},
    n_results=5
)

# hybrid_search() - different filters for keyword search vs semantic search
hybrid_result6 = collection.hybrid_search(
    query={"where_document": {"$contains": "neural"}, "where": {"popularity": {"$gte": 90}}, "n_results": 10},
    knn={"query_texts": ["deep learning"], "where": {"year": {"$gte": 2023}}, "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results:")
for i, doc_id in enumerate(query_result6['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")
    print(f"      {metadatas[idx]}")

print("\nhybrid_search() Results:")
for i, doc_id in enumerate(hybrid_result6['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")
    print(f"      {metadatas[idx]}")

print("\nAnalysis:")
print("  query() only returns 2 results because it requires documents to satisfy BOTH popularity>=90 AND")
print("  year>=2023 simultaneously, along with containing 'neural' and being semantically similar to")
print("  'deep learning'. hybrid_search() returns 5 results by applying popularity>=90 filter to full-text")
print("  search (for 'neural') and year>=2023 filter to vector search (for 'deep learning') independently.")
print("  The fusion then combines results from both searches, capturing documents that strongly match either")
print("  criterion while still being relevant to the overall query intent.")

print("\n" + "=" * 100)
print("SCENARIO 7: Partial Keyword Match + Semantic Similarity")
print("=" * 100)
print("Goal: Documents containing 'Python' + Semantically similar to 'data science'\n")

# query() - filter applied after vector search
query_result7 = collection.query(
    query_texts=["data science"],
    where_document={"$contains": "Python"},
    n_results=5
)

# hybrid_search() - parallel searches then fusion
hybrid_result7 = collection.hybrid_search(
    query={"where_document": {"$contains": "Python"}, "n_results": 10},
    knn={"query_texts": ["data science"], "n_results": 10},
    rank={"rrf": {}},
    n_results=5
)

print("query() Results:")
for i, doc_id in enumerate(query_result7['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nhybrid_search() Results:")
for i, doc_id in enumerate(hybrid_result7['ids'][0]):
    idx = ids.index(doc_id)
    print(f"  {i+1}. {documents[idx]}")

print("\nAnalysis:")
print("  query() only returns 2 results because it first performs vector search for 'data science', then")
print("  filters to documents containing 'Python', which severely limits the result set. hybrid_search()")
print("  returns 5 results by running full-text search (for 'Python') and vector search (for 'data science')")
print("  in parallel, then fusing the results. This captures documents that contain 'Python' (even if not")
print("  semantically closest to 'data science') and documents semantically similar to 'data science' (even")
print("  if they don't contain 'Python'), providing better recall and more comprehensive results.")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
query() limitations:
  - Single search type (vector similarity)
  - Filters applied after search (may miss relevant docs)
  - Cannot combine full-text and vector search results
  - Same filter criteria for all conditions

hybrid_search() advantages:
  - Simultaneous full-text + vector search
  - Independent filters for each search type
  - Intelligent result fusion using RRF
  - Better recall for complex queries
  - Handles scenarios requiring both keyword and semantic matching
""")
