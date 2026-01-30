"""
Integration tests for hybrid_search `_source` inference from include against a real database.

These tests require a running OceanBase/MySQL-compatible endpoint that supports
`DBMS_HYBRID_SEARCH.GET_SQL`.
"""

import contextlib
import json
import time
import uuid

from pymysql.converters import escape_string

from pyseekdb import HNSWConfiguration
from pyseekdb.client.meta_info import CollectionNames


class TestCollectionHybridSearchSourceInferenceRealDB:
    _QUERY_TIMEOUT_SECONDS = 10.0
    _QUERY_RETRY_INTERVAL_SECONDS = 0.2

    def _unique_collection_name(self, prefix: str) -> str:
        # Keep names short to avoid MySQL/OceanBase identifier length limits after
        # internal table-name prefixing (e.g. "c$v1$...").
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _create_test_collection(self, client, collection_name: str, dimension: int = 3):
        config = HNSWConfiguration(dimension=dimension, distance="l2")
        collection = client.create_collection(name=collection_name, configuration=config, embedding_function=None)
        return collection, collection.dimension

    def _generate_query_vector(self, dimension: int) -> list[float]:
        base = [1.0, 2.0, 3.0]
        if dimension <= len(base):
            return base[:dimension]
        extended = base * ((dimension // len(base)) + 1)
        return extended[:dimension]

    def _insert_test_data(self, collection, dimension: int):
        test_data = [
            (
                "Machine learning is a subset of artificial intelligence",
                [1.0, 2.0, 3.0],
                {"category": "AI", "tag": "ml"},
            ),
            (
                "Python programming language is widely used in data science",
                [2.0, 3.0, 4.0],
                {"category": "Programming", "tag": "python"},
            ),
        ]

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        for document, base_vec, metadata in test_data:
            record_id = str(uuid.uuid4())
            ids.append(record_id)
            documents.append(document)
            metadatas.append(metadata)

            if dimension <= len(base_vec):
                embedding = base_vec[:dimension]
            else:
                embedding = (base_vec * ((dimension // len(base_vec)) + 1))[:dimension]

            embeddings.append(embedding)

        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return ids

    def _get_sql_query(self, client, table_name: str, search_parm: dict) -> str:
        search_parm_json = json.dumps(search_parm, ensure_ascii=False)
        client._server._execute(f"SET @search_parm = '{escape_string(search_parm_json)}'")
        get_sql_query = f"SELECT DBMS_HYBRID_SEARCH.GET_SQL('{table_name}', @search_parm) as query_sql FROM dual"  # noqa: S608
        rows = client._server._execute(get_sql_query)
        assert rows and rows[0].get("query_sql")
        query_sql = rows[0]["query_sql"]
        if isinstance(query_sql, str):
            return query_sql.strip().strip("'\"")
        return str(query_sql)

    def test_include_infers_source_result_shape_matrix(self, server_client):
        """
        Verify `_source` inference end-to-end:
        1) GET_SQL result columns match requested include (avoid returning large unused columns like embedding)
        2) SDK return shape matches include
        """
        collection_name = self._unique_collection_name("hs_include_matrix")
        collection = None
        try:
            collection, dimension = self._create_test_collection(server_client, collection_name)
            self._insert_test_data(collection, dimension=dimension)

            query_vector = self._generate_query_vector(dimension)
            knn = {"query_embeddings": query_vector, "n_results": 2}
            table_name = (
                CollectionNames.table_name_v2(collection.id)
                if getattr(collection, "id", None)
                else CollectionNames.table_name(collection.name)
            )

            def execute_get_sql(include: list[str] | None) -> tuple[dict, list[dict]]:
                search_parm = server_client._server._build_search_parm(
                    query=None,
                    knn=knn,
                    rank=None,
                    n_results=2,
                    include=include,
                    dimension=dimension,
                )
                query_sql = self._get_sql_query(server_client, table_name, search_parm)
                deadline = time.time() + self._QUERY_TIMEOUT_SECONDS
                last_exc: Exception | None = None
                while time.time() < deadline:
                    try:
                        rows = server_client._server._execute(query_sql)
                        if rows:
                            return search_parm, rows
                    except Exception as exc:
                        last_exc = exc
                    time.sleep(self._QUERY_RETRY_INTERVAL_SECONDS)
                if last_exc is not None:
                    raise AssertionError("Timed out waiting for GET_SQL query to return rows") from last_exc
                raise AssertionError("Timed out waiting for GET_SQL query to return rows")

            def assert_columns(rows: list[dict], *, present: set[str], absent: set[str]) -> None:
                assert rows
                keys = {str(k).lower() for k in rows[0]}
                for col in present:
                    assert col in keys
                for col in absent:
                    assert col not in keys

            # 1) include=None: default returns documents+metadatas; should not return embedding column
            _, rows = execute_get_sql(include=None)
            assert_columns(rows, present={"document", "metadata"}, absent={"embedding"})

            default_include = collection.hybrid_search(knn=knn, n_results=2)
            assert set(default_include.keys()) == {"ids", "distances", "documents", "metadatas"}
            assert all(isinstance(d, str) for d in default_include["documents"][0])
            assert all(isinstance(m, dict) and m for m in default_include["metadatas"][0])

            # 2) include=[]: ids/distances only; should not return document/metadata/embedding columns
            _, rows = execute_get_sql(include=[])
            assert_columns(rows, present=set(), absent={"document", "metadata", "embedding"})

            ids_only = collection.hybrid_search(knn=knn, n_results=2, include=[])
            assert set(ids_only.keys()) == {"ids", "distances"}

            # 3) include=["documents"]: only document column
            _, rows = execute_get_sql(include=["documents"])
            assert_columns(rows, present={"document"}, absent={"metadata", "embedding"})

            docs_only = collection.hybrid_search(knn=knn, n_results=2, include=["documents"])
            assert set(docs_only.keys()) == {"ids", "distances", "documents"}
            assert all(isinstance(d, str) for d in docs_only["documents"][0])

            # 4) include=["metadatas"]: only metadata column
            _, rows = execute_get_sql(include=["metadatas"])
            assert_columns(rows, present={"metadata"}, absent={"document", "embedding"})

            metadatas_only = collection.hybrid_search(knn=knn, n_results=2, include=["metadatas"])
            assert set(metadatas_only.keys()) == {"ids", "distances", "metadatas"}
            assert all(isinstance(m, dict) and m for m in metadatas_only["metadatas"][0])

            # 5) include=["embeddings"]: only embedding column
            _, rows = execute_get_sql(include=["embeddings"])
            assert_columns(rows, present={"embedding"}, absent={"document", "metadata"})

            embeddings_only = collection.hybrid_search(knn=knn, n_results=2, include=["embeddings"])
            assert set(embeddings_only.keys()) == {"ids", "distances", "embeddings"}
            first_embedding = embeddings_only["embeddings"][0][0]
            assert isinstance(first_embedding, list)
            assert len(first_embedding) == dimension

            # 6) include=["documents","embeddings"]: document+embedding columns
            _, rows = execute_get_sql(include=["documents", "embeddings"])
            assert_columns(rows, present={"document", "embedding"}, absent={"metadata"})

            docs_and_embeddings = collection.hybrid_search(knn=knn, n_results=2, include=["documents", "embeddings"])
            assert set(docs_and_embeddings.keys()) == {"ids", "distances", "documents", "embeddings"}
            assert all(isinstance(d, str) for d in docs_and_embeddings["documents"][0])
            assert all(isinstance(e, list) and len(e) == dimension for e in docs_and_embeddings["embeddings"][0])
        finally:
            with contextlib.suppress(Exception):
                server_client.delete_collection(name=collection_name)

    # NOTE: `HybridSearch` fluent builder was removed on `develop` (rollback enhanced hybrid search).
    # Keep this file focused on verifying OceanBase GET_SQL `_source` inference and result shapes.
