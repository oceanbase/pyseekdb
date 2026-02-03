# pyseekdb

Welcome to pyseekdb's documentation.

pyseekdb is a Python SDK for seekdb and OceanBase AI search. It supports embedded and server deployments with vector, full-text, and hybrid retrieval, and exposes a collection-first API for application workflows. For advanced database operations, you can use MySQL-compatible drivers to run SQL against seekdb and OceanBase.

## Features

- **Unified API**: Single interface for embedded and remote server modes.
- **Vector Operations**: Efficient vector similarity search.
- **Hybrid Search**: Combine vector and full-text search.
- **Embedding Functions**: Built-in support for various embedding models.
- **Collection Management**: Easy collection (table) creation and management.
- **Database Management**: Admin operations for database management.

## Installation

```bash
pip install -U pyseekdb
# or with uv
uv add pyseekdb
```

## Quick Start

Embedded Mode

```python
import pyseekdb

client = pyseekdb.Client(path="./seekdb.db", database="test")
collection = client.get_or_create_collection("my_collection")
```

Remote Server Mode

```python
import pyseekdb

client = pyseekdb.Client(
    host="localhost",
    port=2881,
    tenant="sys",
    database="test",
    user="root",
    password="pass"
)
collection = client.get_or_create_collection("my_collection")
```

Admin Client

```python
import pyseekdb

admin = pyseekdb.AdminClient(path="./seekdb.db")
admin.create_database("new_db")
databases = admin.list_databases()
```

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

guide/index
api/index
```
