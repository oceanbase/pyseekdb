"""
Collection hybrid search tests - REFACTORED using db_client fixture
Demonstrates how to eliminate 67% code duplication using parameterized fixtures
"""
import pytest
import time
import json
import uuid
from typing import List

import pyseekdb


class TestCollectionHybridSearchRefactored:
    """Test collection.hybrid_search() interface using parameterized db_client fixture"""
    
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
    
    def _generate_query_vector(self, dimension: int, base_vector: List[float] = None) -> List[float]:
        """Generate a query vector with the correct dimension"""
        if base_vector is None:
            base_vector = [1.0, 2.0, 3.0]
        
        if dimension <= len(base_vector):
            return base_vector[:dimension]
        else:
            extended = base_vector * ((dimension // len(base_vector)) + 1)
            return extended[:dimension]
    
    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        """Helper method to insert test data via SQL and return inserted IDs"""
        from pyseekdb.client.meta_info import CollectionNames
        table_name = CollectionNames.table_name(collection_name)
        
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
    
    def test_hybrid_search_full_text_only(self, db_client):
        """
        Test hybrid_search with only full-text search (query).
        
        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_ft_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name)
        
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)
        
        # Test 1: Full-text search only
        print(f"\n✅ Testing hybrid_search with full-text search only")
        results = collection.hybrid_search(
            query={
                "where_document": {
                    "$contains": "machine learning"
                }
            },
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        assert results is not None
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results")
        
        # Verify results contain "machine learning"
        for doc in results["documents"][0]:
            if doc:
                assert "machine" in doc.lower() or "learning" in doc.lower()

        # Test 1b: Full-text search with $not_contains
        print(f"   Testing hybrid_search with $not_contains filter")
        forbidden_phrase = "machine learning"
        results_not = collection.hybrid_search(
            query={
                "where_document": {
                    "$not_contains": forbidden_phrase
                }
            },
            n_results=5,
            include=["documents", "metadatas"]
        )

        assert results_not is not None
        assert "documents" in results_not
        docs_not = results_not["documents"][0] if results_not.get("documents") else []
        for doc in docs_not:
            if doc:
                assert forbidden_phrase not in doc.lower()
    
    def test_hybrid_search_vector_only(self, db_client):
        """
        Test hybrid_search with only vector search (knn).
        
        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_vec_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name)
        
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)
        
        # Test: Vector search only
        print(f"\n✅ Testing hybrid_search with vector search only")
        results = collection.hybrid_search(
            knn={
                "query_embeddings": self._generate_query_vector(actual_dimension),
                "n_results": 5
            },
            n_results=5,
            include=["documents", "metadatas", "embeddings"]
        )
        
        assert results is not None
        assert "ids" in results
        assert "distances" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results")
        
        # Verify distances are reasonable
        distances = results["distances"][0]
        assert len(distances) > 0
        for dist in distances:
            assert dist >= 0, f"Distance should be non-negative, got {dist}"
        min_distance = min(distances)
        assert min_distance < 10.0, f"At least one distance should be reasonable, got min={min_distance}"
    
    def test_hybrid_search_combined(self, db_client):
        """
        Test hybrid_search with both full-text and vector search.
        
        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_comb_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name)
        
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)
        
        # Test: Combined full-text and vector search
        print(f"\n✅ Testing hybrid_search with both full-text and vector search")
        results = collection.hybrid_search(
            query={
                "where_document": {
                    "$contains": "machine learning"
                },
                "n_results": 10,
                "boost": 0.4
            },
            knn={
                "query_embeddings": self._generate_query_vector(actual_dimension),
                "n_results": 10,
                "boost": 1.6
            },
            rank={
                "rrf": {
                    "rank_window_size": 60,
                    "rank_constant": 60
                }
            },
            n_results=5,
            include=["documents", "metadatas", "embeddings"]
        )
        
        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results after RRF ranking")
    
    def test_hybrid_search_with_metadata_filter(self, db_client):
        """
        Test hybrid_search with metadata filters.
        
        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_meta_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name)
        
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)
        
        # Test: Hybrid search with metadata filter
        print(f"\n✅ Testing hybrid_search with metadata filter")
        results = collection.hybrid_search(
            query={
                "where_document": {
                    "$contains": "machine"
                },
                "where": {
                    "$and": [
                        {"category": {"$eq": "AI"}},
                        {"page": {"$gte": 1}},
                        {"page": {"$lte": 5}}
                    ]
                },
                "n_results": 10
            },
            knn={
                "query_embeddings": self._generate_query_vector(actual_dimension),
                "where": {
                    "$and": [
                        {"category": {"$eq": "AI"}},
                        {"score": {"$gte": 90}}
                    ]
                },
                "n_results": 10
            },
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        assert results is not None
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results with metadata filters")
        
        # Verify metadata filters are applied
        for metadata in results["metadatas"][0]:
            if metadata:
                assert metadata.get("category") == "AI"
    
    def test_hybrid_search_with_logical_operators(self, db_client):
        """
        Test hybrid_search with logical operators in metadata filters.
        
        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_logic_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name)
        
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)
        
        # Test: Hybrid search with logical operators ($or, $in)
        print(f"\n✅ Testing hybrid_search with logical operators")
        results = collection.hybrid_search(
            query={
                "where_document": {
                    "$and": [
                        {"$contains": "machine"},
                        {"$contains": "learning"}
                    ]
                },
                "where": {
                    "$or": [
                        {"tag": {"$eq": "ml"}},
                        {"tag": {"$eq": "python"}}
                    ]
                },
                "n_results": 10
            },
            knn={
                "query_embeddings": self._generate_query_vector(actual_dimension),
                "where": {
                    "tag": {"$in": ["ml", "python"]}
                },
                "n_results": 10
            },
            rank={"rrf": {}},
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        assert results is not None
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results with logical operators")
        
        # Verify logical operators are applied
        for metadata in results["metadatas"][0]:
            if metadata and "tag" in metadata:
                assert metadata["tag"] in ["ml", "python"]
    
    def test_hybrid_search_scalar_in_nin_and_id(self, db_client):
        """
        Test hybrid_search scalar filters with $in/$nin and #id support.
        
        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_hybrid_search_scalar_{int(time.time() * 1000)}"
        collection, actual_dimension = self._create_test_collection(db_client, collection_name)
        
        inserted_ids = self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        time.sleep(1)
        
        # Test $in operator
        print(f"\n✅ Testing hybrid_search with $in operator")
        results_in = collection.hybrid_search(
            query={
                "where": {
                    "tag": {"$in": ["ml", "python"]}
                },
                "n_results": 10
            },
            n_results=5,
            include=["metadatas"]
        )
        assert results_in and results_in.get("metadatas")
        for metadata in results_in["metadatas"][0]:
            if metadata:
                assert metadata.get("tag") in ["ml", "python"]
        print(f"   Found {len(results_in['ids'][0])} results with tag in ['ml', 'python']")
        
        # Test $nin operator
        print(f"   Testing hybrid_search with $nin operator")
        results_nin = collection.hybrid_search(
            query={
                "where": {
                    "tag": {"$nin": ["ml", "python"]}
                },
                "n_results": 10
            },
            n_results=5,
            include=["metadatas"]
        )
        assert results_nin and results_nin.get("metadatas")
        for metadata in results_nin["metadatas"][0]:
            if metadata:
                assert metadata.get("tag") not in ["ml", "python"]
        print(f"   Found {len(results_nin['ids'][0])} results with tag not in ['ml', 'python']")
        
        # Test #id filter
        print(f"   Testing hybrid_search with #id filter")
        target_id = inserted_ids[0]
        results_id = collection.hybrid_search(
            query={
                "where": {
                    "#id": {"$in": [target_id]}
                },
                "n_results": 5
            },
            n_results=5,
            include=["metadatas"]
        )
        assert results_id and results_id.get("ids") and len(results_id["ids"][0]) > 0
        assert target_id in results_id["ids"][0]
        print(f"   Found target ID: {target_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

