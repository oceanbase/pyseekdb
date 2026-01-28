"""
Integration tests for hybrid_search `_source` inference from include against a real database.

These tests require a running OceanBase/MySQL-compatible endpoint that supports
`DBMS_HYBRID_SEARCH.GET_SQL`.
"""

import json
import time
import uuid

from pymysql.converters import escape_string

from pyseekdb import HNSWConfiguration, HybridSearch
from pyseekdb.client.meta_info import CollectionNames


class TestCollectionHybridSearchSourceInferenceRealDB:
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

    def _insert_test_data(self, client, collection, dimension: int):
        if getattr(collection, "id", None):
            table_name = CollectionNames.table_name_v2(collection.id)
        else:
            table_name = CollectionNames.table_name(collection.name)

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

        inserted_ids = []
        for document, base_vec, metadata in test_data:
            record_id = str(uuid.uuid4())
            inserted_ids.append(record_id)
            record_id_escaped = record_id.replace("'", "''")

            if dimension <= len(base_vec):
                embedding = base_vec[:dimension]
            else:
                embedding = (base_vec * ((dimension // len(base_vec)) + 1))[:dimension]

            vector_str = "[" + ",".join(map(str, embedding)) + "]"
            metadata_str = json.dumps(metadata, ensure_ascii=False).replace("'", "\\'")
            document_str = document.replace("'", "\\'")

            sql = (
                f"INSERT INTO `{table_name}` (_id, document, embedding, metadata) "  # noqa: S608
                f"VALUES (CAST('{record_id_escaped}' AS BINARY), "
                f"'{document_str}', '{vector_str}', '{metadata_str}')"
            )
            client._server._execute(sql)

        return inserted_ids

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

    def test_get_sql_infers_source_excludes_embedding_columns(self, server_client):
        """
        When include does not request embeddings, GET_SQL should generate a query that does not
        return the embedding column.
        """
        collection_name = self._unique_collection_name("hs_get_sql_no_vec")
        collection, dimension = self._create_test_collection(server_client, collection_name)
        self._insert_test_data(server_client, collection, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        knn = {"query_embeddings": query_vector, "n_results": 2}

        search_parm = server_client._server._build_search_parm(
            query=None,
            knn=knn,
            rank=None,
            n_results=2,
            include=None,
            dimension=dimension,
        )
        table_name = (
            CollectionNames.table_name_v2(collection.id)
            if getattr(collection, "id", None)
            else CollectionNames.table_name(collection.name)
        )
        query_sql = self._get_sql_query(server_client, table_name, search_parm)
        rows = server_client._server._execute(query_sql)
        assert rows
        keys = {str(k).lower() for k in rows[0]}
        assert "embedding" not in keys

    def test_get_sql_infers_source_includes_embedding_columns(self, server_client):
        """
        When include requests embeddings, inferred `_source` should allow embedding to be returned
        by the generated SQL.
        """
        collection_name = self._unique_collection_name("hs_get_sql_vec")
        collection, dimension = self._create_test_collection(server_client, collection_name)
        self._insert_test_data(server_client, collection, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        knn = {"query_embeddings": query_vector, "n_results": 2}

        search_parm = server_client._server._build_search_parm(
            query=None,
            knn=knn,
            rank=None,
            n_results=2,
            include=["embeddings"],
            dimension=dimension,
        )
        table_name = (
            CollectionNames.table_name_v2(collection.id)
            if getattr(collection, "id", None)
            else CollectionNames.table_name(collection.name)
        )
        query_sql = self._get_sql_query(server_client, table_name, search_parm)
        rows = server_client._server._execute(query_sql)
        assert rows
        keys = {str(k).lower() for k in rows[0]}
        assert "embedding" in keys

    def test_include_infers_source_result_shape_matrix(self, server_client):
        """
        Result shape correctness across common include patterns.
        """
        collection_name = self._unique_collection_name("hs_include_matrix")
        collection, dimension = self._create_test_collection(server_client, collection_name)
        self._insert_test_data(server_client, collection, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        knn = {"query_embeddings": query_vector, "n_results": 2}

        default_include = collection.hybrid_search(knn=knn, n_results=2)
        assert default_include is not None
        assert set(default_include.keys()) == {"ids", "distances", "documents", "metadatas"}
        assert all(isinstance(d, str) for d in default_include["documents"][0])
        assert all(isinstance(m, dict) and m for m in default_include["metadatas"][0])
        assert "embeddings" not in default_include

        docs_only = collection.hybrid_search(knn=knn, n_results=2, include=["documents"])
        assert docs_only is not None
        assert set(docs_only.keys()) == {"ids", "distances", "documents"}
        assert all(isinstance(d, str) for d in docs_only["documents"][0])

        metadatas_only = collection.hybrid_search(knn=knn, n_results=2, include=["metadatas"])
        assert metadatas_only is not None
        assert set(metadatas_only.keys()) == {"ids", "distances", "metadatas"}
        assert all(isinstance(m, dict) and m for m in metadatas_only["metadatas"][0])

        embeddings_only = collection.hybrid_search(knn=knn, n_results=2, include=["embeddings"])
        assert embeddings_only is not None
        assert set(embeddings_only.keys()) == {"ids", "distances", "embeddings"}
        first_embedding = embeddings_only["embeddings"][0][0]
        assert isinstance(first_embedding, list)
        assert len(first_embedding) == dimension

        docs_and_embeddings = collection.hybrid_search(knn=knn, n_results=2, include=["documents", "embeddings"])
        assert docs_and_embeddings is not None
        assert set(docs_and_embeddings.keys()) == {"ids", "distances", "documents", "embeddings"}
        assert all(isinstance(d, str) for d in docs_and_embeddings["documents"][0])
        assert all(isinstance(e, list) and len(e) == dimension for e in docs_and_embeddings["embeddings"][0])

    def test_search_builder_overrides_include_and_n_results(self, server_client):
        """
        When passing builder via search=, builder include and n_results override the call-site.
        """
        collection_name = self._unique_collection_name("hs_search_precedence")
        collection, dimension = self._create_test_collection(server_client, collection_name)
        self._insert_test_data(server_client, collection, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        hs = HybridSearch().knn(query_embeddings=query_vector, n_results=2).limit(1).select("embeddings")

        results = collection.hybrid_search(search=hs, n_results=2, include=["documents", "metadatas"])
        assert results is not None
        assert set(results.keys()) == {"ids", "distances", "embeddings"}
        assert len(results["ids"][0]) == 1
        first = results["embeddings"][0][0]
        assert isinstance(first, list)
        assert len(first) == dimension
