"""
Integration tests for return_fields with Collection.hybrid_search against a real database.

These tests require a running OceanBase/MySQL-compatible endpoint that supports
`DBMS_HYBRID_SEARCH.GET_SQL`.

"""

import json
import time
import uuid
from typing import List

import pytest

from pyseekdb import HNSWConfiguration, HybridSearch
from pyseekdb.client.meta_info import CollectionNames


class TestCollectionHybridSearchSourceRealDB:
    def _unique_collection_name(self, prefix: str) -> str:
        # Keep names short to avoid MySQL/OceanBase identifier length limits after
        # internal table-name prefixing (e.g. "c$v1$...").
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _create_test_collection(self, client, collection_name: str, dimension: int = 3):
        config = HNSWConfiguration(dimension=dimension, distance="l2")
        collection = client.create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )
        return collection, collection.dimension

    def _generate_query_vector(self, dimension: int) -> List[float]:
        base = [1.0, 2.0, 3.0]
        if dimension <= len(base):
            return base[:dimension]
        extended = base * ((dimension // len(base)) + 1)
        return extended[:dimension]

    def _insert_test_data(self, client, collection_name: str, dimension: int):
        table_name = CollectionNames.table_name(collection_name)
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
                f"INSERT INTO `{table_name}` (_id, document, embedding, metadata) "
                f"VALUES (CAST('{record_id_escaped}' AS BINARY), "
                f"'{document_str}', '{vector_str}', '{metadata_str}')"
            )
            client._server._execute(sql)

        return inserted_ids

    def test_source_excludes_embedding_knn_only(self, server_client):
        """
        When return_fields omits `embedding`, returned embeddings should be None
        even if include asks for it.
        """
        collection_name = self._unique_collection_name("hs_src_knn")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)

        results = collection.hybrid_search(
            knn={"query_embeddings": query_vector, "n_results": 2},
            n_results=2,
            include=["documents", "embeddings"],
            return_fields=["_id", "document"],
        )

        assert results is not None
        assert results.get("ids") is not None
        assert results.get("distances") is not None
        assert results.get("embeddings") is not None
        assert results.get("documents") is not None
        assert len(results["embeddings"]) == 1
        assert all(e is None for e in results["embeddings"][0])
        assert all(d is not None for d in results["documents"][0])
        assert all(isinstance(d, str) for d in results["documents"][0])

    def test_source_excludes_document_query_only(self, server_client):
        """
        When return_fields omits `document`, returned documents should be None
        even if include asks for it.
        """
        collection_name = self._unique_collection_name("hs_src_query")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        results = collection.hybrid_search(
            query={"where_document": {"$contains": "machine learning"}, "n_results": 2},
            n_results=2,
            include=["documents", "metadatas"],
            return_fields=["_id", "metadata"],
        )

        assert results is not None
        assert results.get("documents") is not None
        assert results.get("metadatas") is not None
        assert len(results["documents"]) == 1
        assert all(d is None for d in results["documents"][0])
        assert all(isinstance(m, dict) and m for m in results["metadatas"][0])

    def test_return_fields_empty_include_all(self, server_client):
        """
        When return_fields is an empty list, no stored columns should be returned
        even if include asks for them.
        """
        collection_name = self._unique_collection_name("hs_rf_empty")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)

        results = collection.hybrid_search(
            knn={"query_embeddings": query_vector, "n_results": 2},
            n_results=2,
            include=["documents", "metadatas", "embeddings"],
            return_fields=[],
        )

        assert results is not None
        assert results.get("documents") is not None
        assert results.get("metadatas") is not None
        assert results.get("embeddings") is not None
        assert len(results["documents"]) == 1
        assert len(results["metadatas"]) == 1
        assert len(results["embeddings"]) == 1
        assert all(d is None for d in results["documents"][0])
        assert all(m == {} for m in results["metadatas"][0])
        assert all(e is None for e in results["embeddings"][0])

    def test_return_fields_and_include_matrix_knn(self, server_client):
        """
        Matrix test covering:
        - return_fields None vs [] (normalized)
        - include default/empty/list behavior
        - include can request fields that are filtered out by return_fields
        """
        collection_name = self._unique_collection_name("hs_rf_matrix")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        knn = {"query_embeddings": query_vector, "n_results": 2}

        default_baseline = collection.hybrid_search(knn=knn, n_results=2)
        assert default_baseline is not None
        assert default_baseline.get("documents") is not None
        assert default_baseline.get("metadatas") is not None
        assert "embeddings" not in default_baseline
        assert all(isinstance(d, str) for d in default_baseline["documents"][0])
        assert all(
            isinstance(m, dict) and m for m in default_baseline["metadatas"][0]
        )

        baseline = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["documents", "metadatas", "embeddings"],
        )
        assert baseline is not None
        assert baseline.get("documents") is not None
        assert baseline.get("metadatas") is not None
        assert baseline.get("embeddings") is not None
        assert all(isinstance(d, str) for d in baseline["documents"][0])
        assert all(isinstance(m, dict) and m for m in baseline["metadatas"][0])
        assert all(isinstance(e, list) and len(e) == dimension for e in baseline["embeddings"][0])

        empty_list = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["documents", "metadatas", "embeddings"],
            return_fields=[],
        )
        assert empty_list is not None
        assert empty_list.get("documents") is not None
        assert empty_list.get("metadatas") is not None
        assert empty_list.get("embeddings") is not None
        assert all(d is None for d in empty_list["documents"][0])
        assert all(m == {} for m in empty_list["metadatas"][0])
        assert all(e is None for e in empty_list["embeddings"][0])

        default_empty_list = collection.hybrid_search(
            knn=knn,
            n_results=2,
            return_fields=[],
        )
        assert default_empty_list is not None
        assert default_empty_list.get("documents") is not None
        assert default_empty_list.get("metadatas") is not None
        assert all(d is None for d in default_empty_list["documents"][0])
        assert all(m == {} for m in default_empty_list["metadatas"][0])
        assert "embeddings" not in default_empty_list

        default_include = collection.hybrid_search(
            knn=knn,
            n_results=2,
            return_fields=["_id", "embedding"],
        )
        assert default_include is not None
        assert default_include.get("documents") is not None
        assert default_include.get("metadatas") is not None
        assert all(d is None for d in default_include["documents"][0])
        assert all(m == {} for m in default_include["metadatas"][0])
        assert "embeddings" not in default_include

        ids_only = collection.hybrid_search(knn=knn, n_results=2, include=[])
        assert ids_only is not None
        assert set(ids_only.keys()) == {"ids", "distances"}
        assert len(ids_only["ids"][0]) == 2
        assert len(ids_only["distances"][0]) == 2

        embeddings_denied = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["embeddings"],
            return_fields=["_id", "document"],
        )
        assert embeddings_denied is not None
        assert set(embeddings_denied.keys()) == {"ids", "distances", "embeddings"}
        assert embeddings_denied.get("embeddings") is not None
        assert all(e is None for e in embeddings_denied["embeddings"][0])

        docs_only = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["documents"],
            return_fields=["_id", "document", "embedding"],
        )
        assert docs_only is not None
        assert set(docs_only.keys()) == {"ids", "distances", "documents"}
        assert docs_only.get("documents") is not None
        assert all(isinstance(d, str) for d in docs_only["documents"][0])
        assert "metadatas" not in docs_only
        assert "embeddings" not in docs_only

        metadatas_only = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["metadatas"],
            return_fields=["_id", "metadata"],
        )
        assert metadatas_only is not None
        assert set(metadatas_only.keys()) == {"ids", "distances", "metadatas"}
        assert all(isinstance(m, dict) and m for m in metadatas_only["metadatas"][0])

        singular_document = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["document"],
            return_fields=["_id", "document"],
        )
        assert singular_document is not None
        assert set(singular_document.keys()) == {"ids", "distances", "documents"}
        assert all(isinstance(d, str) for d in singular_document["documents"][0])

        singular_metadata = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["metadata"],
            return_fields=["_id", "metadata"],
        )
        assert singular_metadata is not None
        assert set(singular_metadata.keys()) == {"ids", "distances", "metadatas"}
        assert all(isinstance(m, dict) and m for m in singular_metadata["metadatas"][0])

        singular_embedding = collection.hybrid_search(
            knn=knn,
            n_results=2,
            include=["embedding"],
            return_fields=["_id", "embedding"],
        )
        assert singular_embedding is not None
        assert set(singular_embedding.keys()) == {"ids", "distances", "embeddings"}
        first_embedding = singular_embedding["embeddings"][0][0]
        assert isinstance(first_embedding, list)
        assert len(first_embedding) == dimension

    def test_return_fields_without_id_still_returns_ids(self, server_client):
        """
        return_fields does not need to include `_id` for the SDK to return `ids`.
        """
        collection_name = self._unique_collection_name("hs_no_id")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        results = collection.hybrid_search(
            knn={"query_embeddings": query_vector, "n_results": 2},
            n_results=2,
            include=["documents"],
            return_fields=["document"],
        )

        assert results is not None
        assert set(results.keys()) == {"ids", "distances", "documents"}
        assert results.get("ids") is not None
        assert len(results["ids"][0]) == 2
        assert all(isinstance(d, str) for d in results["documents"][0])

    def test_search_builder_overrides_include_and_n_results(self, server_client):
        """
        When passing builder via search=, builder include and n_results override the
        call-site include/n_results.
        """
        collection_name = self._unique_collection_name("hs_search_precedence")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        hs = (
            HybridSearch()
            .knn(query_embeddings=query_vector, n_results=2)
            .limit(1)
            .select("embeddings")
        )

        results = collection.hybrid_search(
            search=hs,
            n_results=2,
            include=["documents", "metadatas"],
            return_fields=["_id", "document", "metadata", "embedding"],
        )
        assert results is not None
        assert set(results.keys()) == {"ids", "distances", "embeddings"}
        assert len(results["ids"][0]) == 1
        assert results.get("embeddings") is not None
        first = results["embeddings"][0][0]
        assert isinstance(first, list)
        assert len(first) == dimension

    def test_query_knn_rank_rrf_with_return_fields_filtering(self, server_client):
        """
        Query + KNN + RRF rank path works, and return_fields can still filter stored columns.
        """
        collection_name = self._unique_collection_name("hs_qk_rrf")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        results = collection.hybrid_search(
            query={"where_document": {"$contains": "machine learning"}, "boost": 0.5},
            knn={"query_embeddings": query_vector, "boost": 0.5, "n_results": 2},
            rank={"rrf": {"rank_window_size": 10, "rank_constant": 60}},
            n_results=2,
            include=["documents", "metadatas", "embeddings"],
            return_fields=["_id", "document"],
        )

        assert results is not None
        assert results.get("documents") is not None
        assert results.get("metadatas") is not None
        assert results.get("embeddings") is not None
        assert all(isinstance(d, str) for d in results["documents"][0])
        assert all(m == {} for m in results["metadatas"][0])
        assert all(e is None for e in results["embeddings"][0])

    def test_source_includes_embedding_knn_only(self, server_client):
        """
        When return_fields includes `embedding`, returned embeddings should be present.
        """
        collection_name = self._unique_collection_name("hs_src_knn_inc")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)

        results = collection.hybrid_search(
            knn={"query_embeddings": query_vector, "n_results": 2},
            n_results=2,
            include=["embeddings"],
            return_fields=["_id", "embedding"],
        )

        assert results is not None
        assert results.get("embeddings") is not None
        assert len(results["embeddings"]) == 1
        assert any(e is not None for e in results["embeddings"][0])

        first = results["embeddings"][0][0]
        assert isinstance(first, list)
        assert len(first) == dimension

    def test_builder_source_round_trip(self, server_client):
        """
        Same as above, but return_fields is provided via HybridSearch builder.
        """
        collection_name = self._unique_collection_name("hs_builder")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        hs = (
            HybridSearch()
            .knn(query_embeddings=query_vector, n_results=2)
            .limit(2)
            .return_fields(["_id", "embedding"])
            .select("embeddings")
        )

        results = collection.hybrid_search(hs)
        assert results is not None
        assert results.get("embeddings") is not None
        assert any(e is not None for e in results["embeddings"][0])

    def test_builder_return_fields_empty_round_trip(self, server_client):
        """
        When return_fields is an empty list (builder), it is normalized and no
        stored columns are returned.
        """
        collection_name = self._unique_collection_name("hs_builder_empty")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )
        self._insert_test_data(server_client, collection_name, dimension=dimension)
        time.sleep(1)

        query_vector = self._generate_query_vector(dimension)
        hs = (
            HybridSearch()
            .knn(query_embeddings=query_vector, n_results=2)
            .limit(2)
            .return_fields([])
            .select("documents", "metadatas", "embeddings")
        )

        results = collection.hybrid_search(hs)
        assert results is not None
        assert results.get("documents") is not None
        assert results.get("metadatas") is not None
        assert results.get("embeddings") is not None
        assert all(d is None for d in results["documents"][0])
        assert all(m == {} for m in results["metadatas"][0])
        assert all(e is None for e in results["embeddings"][0])

    def test_builder_and_callsite_return_fields_conflict_raises(self, server_client):
        """
        Guardrail: do not allow both the builder and the call-site to set return_fields.
        """
        collection_name = self._unique_collection_name("hs_rf_conflict")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )

        query_vector = self._generate_query_vector(dimension)
        hs = (
            HybridSearch()
            .knn(query_embeddings=query_vector, n_results=2)
            .limit(2)
            .return_fields(["_id"])
        )

        with pytest.raises(
            ValueError, match="Do not mix HybridSearch\\.return_fields\\(\\) with return_fields="
        ):
            collection.hybrid_search(hs, return_fields=["_id", "document"])

    def test_source_kwarg_is_rejected(self, server_client):
        """
        User-facing API does not accept `_source`; it must be `return_fields`.
        """
        collection_name = self._unique_collection_name("hs_reject_source")
        collection, dimension = self._create_test_collection(
            server_client, collection_name
        )

        query_vector = self._generate_query_vector(dimension)
        with pytest.raises(TypeError, match="Use return_fields= instead of _source="):
            collection.hybrid_search(
                knn={"query_embeddings": query_vector, "n_results": 2},
                n_results=2,
                include=["documents"],
                _source=["_id"],
            )
