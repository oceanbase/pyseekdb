"""
SQLAlchemy Hybrid Search Example: full-text + vector search with plain SQL

pyseekdb's Collection API is the easiest way to run hybrid search. If your
application already uses SQLAlchemy, you can run the same hybrid search on your
own tables:

1. Create an engine (seekdb server / OceanBase via PyMySQL, or embedded seekdb)
2. Create a table with a FULLTEXT index and a VECTOR index
3. Insert rows; embeddings come from pyseekdb's DefaultEmbeddingFunction
4. Build a DBMS_HYBRID_SEARCH parameter (full-text query + knn + RRF rank)
5. Ask DBMS_HYBRID_SEARCH.GET_SQL for the SQL and run it with SQLAlchemy

Requires SQLAlchemy 2.x: pip install sqlalchemy
Embedded mode requires a binding that exposes SeekdbInstance.connection_options()
(seekdb, or pylibseekdb >= 1.4).
"""

import json
import os
import sys
import tempfile

import pymysql
import sqlalchemy as sa

import pyseekdb

TABLE_NAME = "sqlalchemy_hybrid_search_demo"

# ==================== Step 1: Create an SQLAlchemy engine ====================
mode = os.getenv("MODE", "embedded")
database = os.getenv("DATABASE", "test")
embedded_instance = None
embedded_tmpdir = None

if mode == "embedded":
    try:
        import seekdb as embedded_binding
    except ModuleNotFoundError:
        import pylibseekdb as embedded_binding

    if not callable(getattr(getattr(embedded_binding, "SeekdbInstance", None), "connection_options", None)):
        print("Skipping: the installed embedded binding has no SeekdbInstance.connection_options().")
        print("Upgrade to seekdb (or pylibseekdb >= 1.4), or run with MODE=server / MODE=oceanbase.")
        sys.exit(0)

    embedded_tmpdir = tempfile.TemporaryDirectory()
    embedded_instance = embedded_binding.open(db_dir=os.path.join(embedded_tmpdir.name, "seekdb.db"))
    # connection_options() returns PyMySQL arguments (unix socket or host/port, user)
    # for the MySQL endpoint of this embedded instance.
    options = embedded_instance.connection_options()
    engine = sa.create_engine(
        "mysql+pymysql://",
        creator=lambda: pymysql.connect(database=database, charset="utf8mb4", **options),
        poolclass=sa.pool.NullPool,
    )
elif mode in {"server", "oceanbase"}:
    tenant = os.getenv("TENANT", "sys" if mode == "server" else "test")
    url = sa.URL.create(
        "mysql+pymysql",
        username=f"{os.getenv('SEEKDB_USER', 'root')}@{tenant}",  # user@tenant
        password=os.getenv("SEEKDB_PASSWORD", ""),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "2881")),
        database=database,
        query={"charset": "utf8mb4"},
    )
    engine = sa.create_engine(url)
else:
    raise ValueError(f"Unsupported MODE: {mode}")

# ==================== Step 2: Create a table with full-text and vector indexes ====================
# DBMS_HYBRID_SEARCH needs a heap-organized table when the table has a user-defined
# primary key, so keep ORGANIZATION = HEAP (pyseekdb collections use the same layout).
embedding_function = pyseekdb.DefaultEmbeddingFunction()
dimension = embedding_function.dimension  # 384 for the default model

create_table_sql = f"""
CREATE TABLE {TABLE_NAME} (
    id INT PRIMARY KEY,
    category VARCHAR(64),
    content VARCHAR(1024),
    embedding VECTOR({dimension}),
    FULLTEXT INDEX idx_content (content),
    VECTOR INDEX idx_embedding (embedding) WITH (distance=cosine, type=hnsw, lib=vsag)
) ORGANIZATION = HEAP
"""

documents = [
    ("AI", "Machine learning is revolutionizing artificial intelligence and data science"),
    ("Programming", "Python programming language is essential for machine learning developers"),
    ("AI", "Deep learning neural networks enable advanced AI applications"),
    ("Data Science", "Data science combines statistics, programming, and domain expertise"),
    ("AI", "Natural language processing uses machine learning to understand text"),
    ("AI", "Computer vision algorithms process images using deep learning techniques"),
    ("Programming", "Python libraries like TensorFlow and PyTorch simplify machine learning"),
    ("AI", "Neural networks mimic the structure of biological brain connections"),
]

try:
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
        conn.execute(sa.text(create_table_sql))

        # ==================== Step 3: Insert rows with embeddings ====================
        embeddings = embedding_function([content for _, content in documents])
        # A lightweight SQLAlchemy Core table is enough for INSERT/SELECT statements.
        demo_table = sa.table(
            TABLE_NAME, sa.column("id"), sa.column("category"), sa.column("content"), sa.column("embedding")
        )
        conn.execute(
            sa.insert(demo_table),
            [
                {
                    "id": i + 1,
                    "category": category,
                    "content": content,
                    # A vector literal is a JSON-style array string, e.g. '[0.1,0.2,...]'
                    "embedding": json.dumps([float(x) for x in embedding]),
                }
                for i, ((category, content), embedding) in enumerate(zip(documents, embeddings, strict=True))
            ],
        )

    # ==================== Step 4: Build the hybrid search parameter ====================
    query_text = "AI research"
    query_vector = [float(x) for x in embedding_function([query_text])[0]]
    search_parm = {
        # Full-text branch: documents that contain "machine learning", only in category "AI"
        "query": {
            "bool": {
                "must": [{"match": {"content": "machine learning"}}],
                "filter": [{"term": {"category": "AI"}}],
            }
        },
        # Vector branch: nearest neighbours of the query embedding (no category filter here,
        # each branch has its own filters)
        "knn": {"field": "embedding", "k": 5, "query_vector": query_vector},
        # Fuse both branches with Reciprocal Rank Fusion
        "rank": {"rrf": {}},
        "size": 5,
        "_source": ["id", "category", "content"],
    }

    # ==================== Step 5: Run the hybrid search with SQLAlchemy ====================
    # @search_parm is a session variable, so keep both statements on one connection.
    with engine.connect() as conn:
        conn.execute(sa.text("SET @search_parm = :parm"), {"parm": json.dumps(search_parm)})
        query_sql = conn.execute(
            sa.text("SELECT DBMS_HYBRID_SEARCH.GET_SQL(:table_name, @search_parm) AS query_sql"),
            {"table_name": TABLE_NAME},
        ).scalar_one()
        # The SQL embeds the query vector, so only print its beginning.
        print("SQL generated by DBMS_HYBRID_SEARCH.GET_SQL (truncated):")
        print(f"  {query_sql[:200]}...\n")

        # The generated SQL is a normal SELECT, so SQLAlchemy can run it and map the rows.
        rows = conn.execute(sa.text(query_sql)).mappings().all()

    print(f"Hybrid search results for full-text 'machine learning' (category=AI) + vector '{query_text}':")
    for i, row in enumerate(rows, start=1):
        print(f"  {i}. [{row['category']}] {row['content']}  (score={row.get('_score')})")

    if not rows:
        raise RuntimeError("hybrid search returned no rows")
finally:
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
    engine.dispose()
    if embedded_instance is not None:
        embedded_instance.close()
    if embedded_tmpdir is not None:
        embedded_tmpdir.cleanup()
