"""
Collection hybrid search tests using HybridSearch builder - mirrors test_collection_hybrid_search.py
but passes HybridSearch instances to collection.hybrid_search().
"""
import pytest
import sys
import os
import time
import json
import uuid
from pathlib import Path
from typing import List

# Add project src path (prefer local source over installed package)
project_root = Path(__file__).parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

import pyseekdb
from pyseekdb import (
    HybridSearch,
    DOCUMENT,
    TEXT,
    EMBEDDINGS,
    K,
    IDS,
    DOCUMENTS,
    METADATAS,
    EMBEDDINGS_FIELD,
    SCORES,
)


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb.db"))
SEEKDB_DATABASE = os.environ.get('SEEKDB_DATABASE', 'test')

# Server mode
SERVER_HOST = os.environ.get('SERVER_HOST', '127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))
SERVER_DATABASE = os.environ.get('SERVER_DATABASE', 'test')
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SERVER_PASSWORD', '')

# OceanBase mode
OB_HOST = os.environ.get('OB_HOST', 'localhost')
OB_PORT = int(os.environ.get('OB_PORT', '11202'))
OB_TENANT = os.environ.get('OB_TENANT', 'mysql')
OB_DATABASE = os.environ.get('OB_DATABASE', 'test')
OB_USER = os.environ.get('OB_USER', 'root')
OB_PASSWORD = os.environ.get('OB_PASSWORD', '')


class TestCollectionHybridSearchWithBuilder:
    """Test collection.hybrid_search() using HybridSearch builder for all three modes"""

    def _hs(self, collection, build):
        hs = HybridSearch()
        hs = build(hs)
        return collection.hybrid_search(hs)

    def _create_test_collection(self, client, collection_name: str, dimension: int = None):
        """Helper method to create a test collection"""
        from pyseekdb import HNSWConfiguration
        if dimension is not None:
            config = HNSWConfiguration(dimension=dimension, distance='l2')
            collection = client.create_collection(
                name=collection_name,
                configuration=config,
                embedding_function=None
            )
        else:
            collection = client.create_collection(name=collection_name)
        return collection, collection.dimension

    def _generate_query_vector(self, dimension: int, base_vector: List[float] = [1.0, 2.0, 3.0]) -> List[float]:
        if dimension <= len(base_vector):
            return base_vector[:dimension]
        extended = base_vector * ((dimension // len(base_vector)) + 1)
        return extended[:dimension]

    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        table_name = f"c$v1${collection_name}"
        base_vectors = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1],
            [2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [2.2, 3.2, 4.2],
            [1.4, 2.4, 3.4]
        ]

        test_data = [
            {
                "document": "Machine learning is a subset of artificial intelligence",
                "base_vector": base_vectors[0],
                "metadata": {"category": "AI", "page": 1, "score": 95, "tag": "ml"}
            },
            {
                "document": "Python programming language is widely used in data science",
                "base_vector": base_vectors[1],
                "metadata": {"category": "Programming", "page": 2, "score": 88, "tag": "python"}
            },
            {
                "document": "Deep learning algorithms for neural networks",
                "base_vector": base_vectors[2],
                "metadata": {"category": "AI", "page": 3, "score": 92, "tag": "ml"}
            },
            {
                "document": "Data science with Python and machine learning",
                "base_vector": base_vectors[3],
                "metadata": {"category": "Data Science", "page": 4, "score": 90, "tag": "python"}
            },
            {
                "document": "Introduction to artificial intelligence and neural networks",
                "base_vector": base_vectors[4],
                "metadata": {"category": "AI", "page": 5, "score": 85, "tag": "neural"}
            },
            {
                "document": "Advanced machine learning techniques and algorithms",
                "base_vector": base_vectors[5],
                "metadata": {"category": "AI", "page": 6, "score": 93, "tag": "ml"}
            },
            {
                "document": "Python tutorial for beginners in programming",
                "base_vector": base_vectors[6],
                "metadata": {"category": "Programming", "page": 7, "score": 87, "tag": "python"}
            },
            {
                "document": "Natural language processing with machine learning",
                "base_vector": base_vectors[7],
                "metadata": {"category": "AI", "page": 8, "score": 91, "tag": "nlp"}
            }
        ]

        inserted_ids = []
        for data in test_data:
            id_str = str(uuid.uuid4())
            inserted_ids.append(id_str)
            id_str_escaped = id_str.replace("'", "''")

            base_vec = data["base_vector"]
            if dimension <= len(base_vec):
                embedding = base_vec[:dimension]
            else:
                embedding = base_vec * ((dimension // len(base_vec)) + 1)
                embedding = embedding[:dimension]

            vector_str = "[" + ",".join(map(str, embedding)) + "]"
            metadata_str = json.dumps(data["metadata"], ensure_ascii=False).replace("'", "\\'")
            document_str = data["document"].replace("'", "\\'")

            sql = f"""INSERT INTO `{table_name}` (_id, document, embedding, metadata) 
                     VALUES (CAST('{id_str_escaped}' AS BINARY), '{document_str}', '{vector_str}', '{metadata_str}')"""
            client._server._execute(sql)

        print(f"   Inserted {len(test_data)} test records (dimension={dimension})")
        return inserted_ids

    def _cleanup_collection(self, client, collection_name: str):
        table_name = f"c$v1${collection_name}"
        try:
            client._server._execute(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"   Cleaned up test table: {table_name}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup test table: {cleanup_error}")

    # -------------------- OceanBase --------------------
    def test_oceanbase_hybrid_search_full_text_only(self):
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with full-text search only")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.contains("machine learning"))
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")

            forbidden_phrase = "machine learning"
            print(f"   Testing hybrid_search with $not_contains filter")
            results_not = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.not_contains(forbidden_phrase))
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results_not is not None
            assert "documents" in results_not
            docs_not = results_not["documents"][0] if results_not.get("documents") else []
            for doc in docs_not:
                if doc:
                    assert forbidden_phrase not in doc.lower()
        finally:
            self._cleanup_collection(client, collection_name)

    def test_oceanbase_hybrid_search_vector_only(self):
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with vector search only")
            results = self._hs(
                collection,
                lambda hs: hs
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=5)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            )

            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results")

            distances = results["distances"][0]
            assert len(distances) > 0
            for dist in distances:
                assert dist >= 0
            assert min(distances) < 10.0
        finally:
            self._cleanup_collection(client, collection_name)

    def test_oceanbase_hybrid_search_combined(self):
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with both full-text and vector search")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.contains("machine learning"), n_results=10, boost=0.4)
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=10, boost=1.6)
                .rank("rrf", rank_window_size=60, rank_constant=60)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            )

            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results after RRF ranking")
        finally:
            self._cleanup_collection(client, collection_name)

    def test_oceanbase_hybrid_search_with_metadata_filter(self):
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with metadata filter")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(
                    DOCUMENT.contains("machine"),
                    K("category") == "AI",
                    K("page") >= 1,
                    K("page") <= 5,
                    n_results=10
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("category") == "AI",
                    K("score") >= 90,
                    n_results=10
                )
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with metadata filters")
            for metadata in results["metadatas"][0]:
                if metadata:
                    assert metadata.get("category") == "AI"
        finally:
            self._cleanup_collection(client, collection_name)

    def test_oceanbase_hybrid_search_with_logical_operators(self):
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with logical operators")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(
                    DOCUMENT.contains(["machine", "learning"]),
                    (K("tag") == "ml") | (K("tag") == "python"),
                    n_results=10
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("tag").is_in(["ml", "python"]),
                    n_results=10
                )
                .rank()
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with logical operators")
            for metadata in results["metadatas"][0]:
                if metadata and "tag" in metadata:
                    assert metadata["tag"] in ["ml", "python"]
        finally:
            self._cleanup_collection(client, collection_name)

    def test_oceanbase_hybrid_search_scalar_in_nin_and_id(self):
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            inserted_ids = self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            results_in = self._hs(
                collection,
                lambda hs: hs
                .query(K("tag").is_in(["ml", "python"]), n_results=10)
                .limit(5)
                .select(METADATAS)
            )
            assert results_in and results_in.get("metadatas")
            for metadata in results_in["metadatas"][0]:
                if metadata:
                    assert metadata.get("tag") in ["ml", "python"]

            results_nin = self._hs(
                collection,
                lambda hs: hs
                .query(K("tag").not_in(["ml", "python"]), n_results=10)
                .limit(5)
                .select(METADATAS)
            )
            assert results_nin and results_nin.get("metadatas")
            for metadata in results_nin["metadatas"][0]:
                if metadata:
                    assert metadata.get("tag") not in ["ml", "python"]

            target_id = inserted_ids[0]
            results_id = self._hs(
                collection,
                lambda hs: hs
                .query(K("#id").is_in([target_id]), n_results=5)
                .limit(5)
                .select(METADATAS)
            )
            assert results_id and results_id.get("ids") and len(results_id["ids"][0]) > 0
            assert target_id in results_id["ids"][0]
        finally:
            self._cleanup_collection(client, collection_name)

    # -------------------- Seekdb Server --------------------
    def test_server_hybrid_search_full_text_only(self):
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with full-text search only (SeekdbServer)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.contains("machine learning"))
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0

            forbidden_phrase = "machine learning"
            print(f"   Testing hybrid_search with $not_contains filter (SeekdbServer)")
            results_not = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.not_contains(forbidden_phrase))
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results_not is not None
            assert "documents" in results_not
            docs_not = results_not["documents"][0] if results_not.get("documents") else []
            for doc in docs_not:
                if doc:
                    assert forbidden_phrase not in doc.lower()
        finally:
            self._cleanup_collection(client, collection_name)

    def test_server_hybrid_search_combined(self):
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with both full-text and vector search (SeekdbServer)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.contains("machine learning"), n_results=10, boost=0.4)
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=10, boost=1.6)
                .rank("rrf", rank_window_size=60, rank_constant=60)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            )

            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
        finally:
            self._cleanup_collection(client, collection_name)

    def test_server_hybrid_search_vector_only(self):
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with vector search only (SeekdbServer)")
            results = self._hs(
                collection,
                lambda hs: hs
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=5)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            )

            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            distances = results["distances"][0]
            assert len(distances) > 0
            for dist in distances:
                assert dist >= 0
            assert min(distances) < 10.0
        finally:
            self._cleanup_collection(client, collection_name)

    def test_server_hybrid_search_with_metadata_filter(self):
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with metadata filter (SeekdbServer)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(
                    DOCUMENT.contains("machine"),
                    K("category") == "AI",
                    K("page") >= 1,
                    K("page") <= 5,
                    n_results=10
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("category") == "AI",
                    K("score") >= 90,
                    n_results=10
                )
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results with metadata filters")
            for metadata in results["metadatas"][0]:
                if metadata:
                    assert metadata.get("category") == "AI"
        finally:
            self._cleanup_collection(client, collection_name)

    def test_server_hybrid_search_with_logical_operators(self):
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with logical operators (SeekdbServer)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(
                    DOCUMENT.contains(["machine", "learning"]),
                    (K("tag") == "ml") | (K("tag") == "python"),
                    n_results=10
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("tag").is_in(["ml", "python"]),
                    n_results=10
                )
                .rank()
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            for metadata in results["metadatas"][0]:
                if metadata and "tag" in metadata:
                    assert metadata["tag"] in ["ml", "python"]
        finally:
            self._cleanup_collection(client, collection_name)

    def test_server_hybrid_search_scalar_in_nin_and_id(self):
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"SeekdbServer connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            inserted_ids = self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            results_in = self._hs(
                collection,
                lambda hs: hs
                .query(K("tag").is_in(["ml", "python"]), n_results=10)
                .limit(5)
                .select(METADATAS)
            )
            assert results_in and results_in.get("metadatas")
            for metadata in results_in["metadatas"][0]:
                if metadata:
                    assert metadata.get("tag") in ["ml", "python"]

            results_nin = self._hs(
                collection,
                lambda hs: hs
                .query(K("tag").not_in(["ml", "python"]), n_results=10)
                .limit(5)
                .select(METADATAS)
            )
            assert results_nin and results_nin.get("metadatas")
            for metadata in results_nin["metadatas"][0]:
                if metadata:
                    assert metadata.get("tag") not in ["ml", "python"]

            target_id = inserted_ids[0]
            results_id = self._hs(
                collection,
                lambda hs: hs
                .query(K("#id").is_in([target_id]), n_results=5)
                .limit(5)
                .select(METADATAS)
            )
            assert results_id and results_id.get("ids") and len(results_id["ids"][0]) > 0
            assert target_id in results_id["ids"][0]
        finally:
            self._cleanup_collection(client, collection_name)

    # -------------------- Embedded --------------------
    def test_embedded_hybrid_search_scalar_in_nin_and_id(self):
        try:
            import pylibseekdb  # noqa: F401
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            inserted_ids = self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            results_in = self._hs(
                collection,
                lambda hs: hs
                .query(K("tag").is_in(["ml", "python"]), n_results=10)
                .limit(5)
                .select(METADATAS)
            )
            assert results_in and results_in.get("metadatas")
            for metadata in results_in["metadatas"][0]:
                if metadata:
                    assert metadata.get("tag") in ["ml", "python"]

            results_nin = self._hs(
                collection,
                lambda hs: hs
                .query(K("tag").not_in(["ml", "python"]), n_results=10)
                .limit(5)
                .select(METADATAS)
            )
            assert results_nin and results_nin.get("metadatas")
            for metadata in results_nin["metadatas"][0]:
                if metadata:
                    assert metadata.get("tag") not in ["ml", "python"]

            target_id = inserted_ids[0]
            results_id = self._hs(
                collection,
                lambda hs: hs
                .query(K("#id").is_in([target_id]), n_results=5)
                .limit(5)
                .select(METADATAS)
            )
            assert results_id and results_id.get("ids") and len(results_id["ids"][0]) > 0
            assert target_id in results_id["ids"][0]
        finally:
            self._cleanup_collection(client, collection_name)

    def test_embedded_hybrid_search_full_text_only(self):
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.SeekdbEmbeddedClient)

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with full-text search only (SeekdbEmbedded)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.contains("machine learning"))
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert "ids" in results
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0

            forbidden_phrase = "machine learning"
            print(f"   Testing hybrid_search with $not_contains filter (SeekdbEmbedded)")
            results_not = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.not_contains(forbidden_phrase))
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results_not is not None
            assert "documents" in results_not
            docs_not = results_not["documents"][0] if results_not.get("documents") else []
            for doc in docs_not:
                if doc:
                    assert forbidden_phrase not in doc.lower()
        finally:
            self._cleanup_collection(client, collection_name)

    def test_embedded_hybrid_search_vector_only(self):
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with vector search only (SeekdbEmbedded)")
            results = self._hs(
                collection,
                lambda hs: hs
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=5)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            )

            assert results is not None
            assert "ids" in results
            assert "distances" in results
            assert len(results["ids"]) > 0
            distances = results["distances"][0]
            assert len(distances) > 0
            for dist in distances:
                assert dist >= 0
            assert min(distances) < 10.0
        finally:
            self._cleanup_collection(client, collection_name)

    def test_embedded_hybrid_search_combined(self):
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with both full-text and vector search (SeekdbEmbedded)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(DOCUMENT.contains("machine learning"), n_results=10, boost=0.4)
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=10, boost=1.6)
                .rank("rrf", rank_window_size=60, rank_constant=60)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            )

            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            print(f"   Found {len(results['ids'][0])} results after RRF ranking")
        finally:
            self._cleanup_collection(client, collection_name)

    def test_embedded_hybrid_search_with_metadata_filter(self):
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with metadata filter (SeekdbEmbedded)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(
                    DOCUMENT.contains("machine"),
                    K("category") == "AI",
                    K("page") >= 1,
                    K("page") <= 5,
                    n_results=10
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("category") == "AI",
                    K("score") >= 90,
                    n_results=10
                )
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            for metadata in results["metadatas"][0]:
                if metadata:
                    assert metadata.get("category") == "AI"
        finally:
            self._cleanup_collection(client, collection_name)

    def test_embedded_hybrid_search_with_logical_operators(self):
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        collection_name = f"test_hybrid_search_{int(time.time())}"
        collection, actual_dimension = self._create_test_collection(client, collection_name)

        try:
            self._insert_test_data(client, collection_name, dimension=actual_dimension)
            time.sleep(1)

            print(f"\n✅ Testing hybrid_search with logical operators (SeekdbEmbedded)")
            results = self._hs(
                collection,
                lambda hs: hs
                .query(
                    DOCUMENT.contains(["machine", "learning"]),
                    (K("tag") == "ml") | (K("tag") == "python"),
                    n_results=10
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("tag").is_in(["ml", "python"]),
                    n_results=10
                )
                .rank()
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            )

            assert results is not None
            assert len(results["ids"]) > 0
            assert len(results["ids"][0]) > 0
            for metadata in results["metadatas"][0]:
                if metadata and "tag" in metadata:
                    assert metadata["tag"] in ["ml", "python"]
        finally:
            self._cleanup_collection(client, collection_name)

