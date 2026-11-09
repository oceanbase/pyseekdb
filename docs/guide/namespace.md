# Namespace Collections

Namespace collections partition data inside a single physical collection. Each **namespace** is a logical slice (similar to a tenant or dataset) that shares catalog tables and IVF-backed vector indexes, while keeping documents isolated by `namespace_id`.

**Requirements**

- LakeBase **4.6.1.0** or newer (`OceanBase Database AI` in `SELECT version()`)
- `use_namespace=True` when creating the collection
- An explicit **IVF** `Schema` (the default non-namespace path builds **HNSW**, which namespace collections do not support)

## 1. Create a namespace-enabled collection

```python
import pyseekdb
from pyseekdb import IVFConfiguration, FulltextIndexConfig, VectorIndexConfig, Schema

client = pyseekdb.Client(host="127.0.0.1", port=2881, database="test")

schema = Schema(
    vector_index=VectorIndexConfig(
        ivf=IVFConfiguration(dimension=384, distance="cosine", centroids_fresh_mode="spfresh"),
        embedding_function=None,  # or pass your embedding function
    ),
    fulltext_index=FulltextIndexConfig(analyzer="ik"),
)

collection = client.create_collection(
    name="products",
    schema=schema,
    use_namespace=True,
    partition_count=4,  # optional; controls table partitioning
)
```

If you omit `schema` with `use_namespace=True`, pyseekdb raises a clear error instead of silently building the default HNSW schema.

**Not supported yet**

- Default / HNSW schemas (`configuration=HNSWConfiguration(...)` without an IVF `Schema`)
- `SparseVectorIndexConfig` on namespace collections

## 2. Manage namespaces

```python
ns = collection.create_namespace("electronics")
# or idempotently:
ns = collection.get_or_create_namespace("electronics")

collection.list_namespaces()   # ["electronics", ...]
collection.delete_namespace("electronics")
```

## 3. Add and query data inside a namespace

Namespace DML uses **record ids** — per-document business keys you pass as `ids` in `namespace.add()`, `get()`, `update()`, and `delete()`. They are stored in `data_content.id` (JSON), not as the table primary key.

```python
ns.add(
    ids=["doc_1", "doc_2"],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    documents=["phone", "laptop"],
    metadatas=[{"brand": "A"}, {"brand": "B"}],
)

rows = ns.get(ids=["doc_1"], include=["documents", "metadatas"])
hits = ns.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=5)
```

**Record id rules (namespace path)**

- Must be non-empty strings
- Currently limited to `[A-Za-z0-9_]` (letters, digits, underscore)
- Maximum length 512 characters

Standard (non-namespace) collections accept broader `_id` values (for example UUIDs with hyphens). If you need UUID-style ids on the namespace path, normalize them (for example `doc_a1b2c3d4`) or wait for a future relaxation.

## 4. `get_or_create_collection` compatibility

`get_or_create_collection(..., use_namespace=...)` must match how the collection was originally created:

```python
# OK — reuses the existing namespace collection
client.get_or_create_collection("products", schema=schema, use_namespace=True)

# Raises ValueError — name already used by a standard HNSW collection
client.get_or_create_collection("legacy_coll", use_namespace=True)
```

Delete the old collection or pick a new name when switching between standard and namespace modes.

## 5. Collection-level hybrid search

Namespace collections also expose `collection.hybrid_search(...)` across namespaces. See integration examples under `tests/integration_tests/test_namespace_*.py` for FTS + vector patterns.

## See also

- [Collection management](collection-management.md) — standard HNSW collections
- [DML operations](dml.md) — `add` / `update` / `delete` on standard collections
- `examples/namespace_example.py` — minimal runnable sample
