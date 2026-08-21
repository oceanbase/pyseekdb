"""
Namespace collection example (LakeBase 4.6.1.0+).

Demonstrates:
1. Creating a namespace-enabled collection with an explicit IVF schema
2. Creating a namespace and adding documents with record ids
3. Vector query inside the namespace
"""

import os

import pyseekdb
from pyseekdb import FulltextIndexConfig, IVFConfiguration, Schema, VectorIndexConfig

# Connect to LakeBase / OceanBase (adjust host/port/credentials)
mode = os.getenv("MODE", "oceanbase")
if mode == "embedded":
    client = pyseekdb.Client()
elif mode in {"server", "oceanbase"}:
    client = pyseekdb.Client(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "2881")),
        tenant=os.getenv("TENANT", "sys" if mode == "server" else "test"),
        database=os.getenv("DATABASE", "test"),
        user=os.getenv("SEEKDB_USER", "root"),
        password=os.getenv("SEEKDB_PASSWORD", ""),
    )
else:
    raise ValueError(f"Unsupported MODE: {mode}")

schema = Schema(
    vector_index=VectorIndexConfig(
        ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
        embedding_function=None,
    ),
    fulltext_index=FulltextIndexConfig(analyzer="ik"),
)

collection_name = "demo_namespace_collection"
collection = client.create_collection(
    name=collection_name,
    schema=schema,
    use_namespace=True,
    partition_count=4,
)

ns = collection.get_or_create_namespace("demo")

# Record ids are per-document keys inside this namespace (see docs/guide/namespace.md).
ns.add(
    ids=["item_alpha", "item_beta"],
    embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    documents=["alpha product", "beta product"],
    metadatas=[{"tier": "a"}, {"tier": "b"}],
)

results = ns.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["documents", "metadatas"])
print("Top match:", results["ids"][0][0], results["documents"][0][0])

collection.delete_namespace("demo")
client.delete_collection(collection_name)
