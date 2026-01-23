"""
Collection hybrid search tests using HybridSearch builder with db_client fixture
Mirrors test_collection_hybrid_search.py but passes HybridSearch instances to collection.hybrid_search().
"""

import time
import uuid

import pytest

from pyseekdb import (
    DOCUMENT,
    DOCUMENTS,
    EMBEDDINGS,
    EMBEDDINGS_FIELD,
    METADATAS,
    HybridSearch,
    K,
)


class TestCollectionHybridSearchWithBuilder:
    """Test collection.hybrid_search() using HybridSearch builder with parameterized db_client fixture"""

    def _hs(self, collection, build):
        hs = HybridSearch()
        hs = build(hs)
        return collection.hybrid_search(hs)

    def _create_test_collection(self, client, collection_name: str, dimension: int | None = None):
        """Helper method to create a test collection"""
        from pyseekdb import HNSWConfiguration

        if dimension is not None:
            config = HNSWConfiguration(dimension=dimension, distance="l2")
            collection = client.create_collection(name=collection_name, configuration=config, embedding_function=None)
        else:
            collection = client.create_collection(name=collection_name)
        return collection, collection.dimension

    def _generate_query_vector(self, dimension: int, base_vector: list[float] | None = None) -> list[float]:
        if base_vector is None:
            base_vector = [1.0, 2.0, 3.0]

        if dimension <= len(base_vector):
            return base_vector[:dimension]
        extended = base_vector * ((dimension // len(base_vector)) + 1)
        return extended[:dimension]

    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        collection = client.get_collection(collection_name)
        base_vectors = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1],
            [2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [2.2, 3.2, 4.2],
            [1.4, 2.4, 3.4],
        ]

        test_data = [
            {
                "document": "Machine learning is a subset of artificial intelligence",
                "base_vector": base_vectors[0],
                "metadata": {"category": "AI", "page": 1, "score": 95, "tag": "ml"},
            },
            {
                "document": "Python programming language is widely used in data science",
                "base_vector": base_vectors[1],
                "metadata": {
                    "category": "Programming",
                    "page": 2,
                    "score": 88,
                    "tag": "python",
                },
            },
            {
                "document": "Deep learning algorithms for neural networks",
                "base_vector": base_vectors[2],
                "metadata": {"category": "AI", "page": 3, "score": 92, "tag": "ml"},
            },
            {
                "document": "Data science with Python and machine learning",
                "base_vector": base_vectors[3],
                "metadata": {
                    "category": "Data Science",
                    "page": 4,
                    "score": 90,
                    "tag": "python",
                },
            },
            {
                "document": "Introduction to artificial intelligence and neural networks",
                "base_vector": base_vectors[4],
                "metadata": {"category": "AI", "page": 5, "score": 85, "tag": "neural"},
            },
            {
                "document": "Advanced machine learning techniques and algorithms",
                "base_vector": base_vectors[5],
                "metadata": {"category": "AI", "page": 6, "score": 93, "tag": "ml"},
            },
            {
                "document": "Python tutorial for beginners in programming",
                "base_vector": base_vectors[6],
                "metadata": {
                    "category": "Programming",
                    "page": 7,
                    "score": 87,
                    "tag": "python",
                },
            },
            {
                "document": "Natural language processing with machine learning",
                "base_vector": base_vectors[7],
                "metadata": {"category": "AI", "page": 8, "score": 91, "tag": "nlp"},
            },
        ]

        inserted_ids = [str(uuid.uuid4()) for _ in test_data]
        collection.add(
            ids=inserted_ids,
            embeddings=[data["base_vector"] for data in test_data],
            documents=[data["document"] for data in test_data],
            metadatas=[data["metadata"] for data in test_data],
        )

        print(f"   Inserted {len(test_data)} test records (dimension={dimension})")
        return inserted_ids

    def test_hybrid_search_full_text_only(self, db_client):
        """
        Test hybrid_search with only full-text search using HybridSearch builder.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_builder_ft_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name, dimension=3)

        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)

        print("\n✅ Testing hybrid_search with full-text search only")
        results = self._hs(
            collection,
            lambda hs: hs.query(DOCUMENT.contains("machine learning")).limit(5).select(DOCUMENTS, METADATAS),
        )

        assert results is not None
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results")

        forbidden_phrase = "machine learning"
        print("   Testing hybrid_search with $not_contains filter")
        results_not = self._hs(
            collection,
            lambda hs: hs.query(DOCUMENT.not_contains(forbidden_phrase)).limit(5).select(DOCUMENTS, METADATAS),
        )

        assert results_not is not None
        assert "documents" in results_not
        docs_not = results_not["documents"][0] if results_not.get("documents") else []
        for doc in docs_not:
            if doc:
                assert forbidden_phrase not in doc.lower()

    def test_hybrid_search_vector_only(self, db_client):
        """
        Test hybrid_search with only vector search using HybridSearch builder.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_builder_vec_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name, dimension=3)

        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)

        print("\n✅ Testing hybrid_search with vector search only")
        results = self._hs(
            collection,
            lambda hs: (
                hs
                .knn(EMBEDDINGS(self._generate_query_vector(actual_dimension)), n_results=5)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            ),
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

    def test_hybrid_search_combined(self, db_client):
        """
        Test hybrid_search with both full-text and vector search using HybridSearch builder.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_builder_comb_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name, dimension=3)

        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)

        print("\n✅ Testing hybrid_search with both full-text and vector search")
        results = self._hs(
            collection,
            lambda hs: (
                hs
                .query(DOCUMENT.contains("machine learning"), n_results=10, boost=0.4)
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    n_results=10,
                    boost=1.6,
                )
                .rank("rrf", rank_window_size=60, rank_constant=60)
                .limit(5)
                .select(DOCUMENTS, METADATAS, EMBEDDINGS_FIELD)
            ),
        )

        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results after RRF ranking")

    def test_hybrid_search_with_metadata_filter(self, db_client):
        """
        Test hybrid_search with metadata filters using HybridSearch builder.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_builder_meta_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name, dimension=3)

        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)

        print("\n✅ Testing hybrid_search with metadata filter")
        results = self._hs(
            collection,
            lambda hs: (
                hs
                .query(
                    DOCUMENT.contains("machine"),
                    K("category") == "AI",
                    K("page") >= 1,
                    K("page") <= 5,
                    n_results=10,
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("category") == "AI",
                    K("score") >= 90,
                    n_results=10,
                )
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            ),
        )

        assert results is not None
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results with metadata filters")
        for metadata in results["metadatas"][0]:
            if metadata:
                assert metadata.get("category") == "AI"

    def test_hybrid_search_with_logical_operators(self, db_client):
        """
        Test hybrid_search with logical operators using HybridSearch builder.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_builder_logic_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name, dimension=3)

        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)

        print("\n✅ Testing hybrid_search with logical operators")
        results = self._hs(
            collection,
            lambda hs: (
                hs
                .query(
                    DOCUMENT.contains(["machine", "learning"]),
                    (K("tag") == "ml") | (K("tag") == "python"),
                    n_results=10,
                )
                .knn(
                    EMBEDDINGS(self._generate_query_vector(actual_dimension)),
                    K("tag").is_in(["ml", "python"]),
                    n_results=10,
                )
                .rank()
                .limit(5)
                .select(DOCUMENTS, METADATAS)
            ),
        )

        assert results is not None
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results with logical operators")
        for metadata in results["metadatas"][0]:
            if metadata and "tag" in metadata:
                assert metadata["tag"] in ["ml", "python"]

    def test_hybrid_search_scalar_in_nin_and_id(self, db_client):
        """
        Test hybrid_search with $in/$nin and #id using HybridSearch builder.

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_builder_scalar_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name, dimension=3)

        inserted_ids = self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)

        print("\n✅ Testing hybrid_search with $in operator")
        results_in = self._hs(
            collection,
            lambda hs: hs.query(K("tag").is_in(["ml", "python"]), n_results=10).limit(5).select(METADATAS),
        )
        assert results_in and results_in.get("metadatas")
        for metadata in results_in["metadatas"][0]:
            if metadata:
                assert metadata.get("tag") in ["ml", "python"]
        print(f"   Found {len(results_in['ids'][0])} results with $in")

        print("   Testing hybrid_search with $nin operator")
        results_nin = self._hs(
            collection,
            lambda hs: hs.query(K("tag").not_in(["ml", "python"]), n_results=10).limit(5).select(METADATAS),
        )
        assert results_nin and results_nin.get("metadatas")
        for metadata in results_nin["metadatas"][0]:
            if metadata:
                assert metadata.get("tag") not in ["ml", "python"]
        print(f"   Found {len(results_nin['ids'][0])} results with $nin")

        print("   Testing hybrid_search with #id filter")
        target_id = inserted_ids[0]
        results_id = self._hs(
            collection,
            lambda hs: hs.query(K("#id").is_in([target_id]), n_results=5).limit(5).select(METADATAS),
        )
        assert results_id and results_id.get("ids") and len(results_id["ids"][0]) > 0
        assert target_id in results_id["ids"][0]
        print("   Found target ID successfully")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
