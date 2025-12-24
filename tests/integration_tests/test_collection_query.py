"""
Collection query tests - REFACTORED using db_client fixture
Demonstrates how to use the new conftest.py fixtures to eliminate code duplication
"""
import pytest
import time
import json
import uuid

import pyseekdb


class TestCollectionQueryRefactored:
    """Test collection.query() interface using parameterized db_client fixture"""
    
    def _insert_test_data(self, client, collection_name: str, dimension: int = 3):
        """Helper method to insert test data using direct SQL
        
        Args:
            client: Client instance
            collection_name: Collection name
            dimension: Actual dimension of the collection (used to generate vectors)
        """
        from pyseekdb.client.meta_info import CollectionNames
        table_name = CollectionNames.table_name(collection_name)
        
        # Base vectors (3D) - will be extended or truncated to match actual dimension
        base_vectors = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1],
            [2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2]
        ]
        
        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "document": "This is a test document about machine learning",
                "base_vector": base_vectors[0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"}
            },
            {
                "document": "Python programming tutorial for beginners",
                "base_vector": base_vectors[1],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"}
            },
            {
                "document": "Advanced machine learning algorithms",
                "base_vector": base_vectors[2],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"}
            },
            {
                "document": "Data science with Python",
                "base_vector": base_vectors[3],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"}
            },
            {
                "document": "Introduction to neural networks",
                "base_vector": base_vectors[4],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"}
            }
        ]
        
        for data in test_data:
            # Generate UUID for _id (use string format directly)
            id_str = str(uuid.uuid4())
            # Escape single quotes in ID
            id_str_escaped = id_str.replace("'", "''")
            
            # Generate vector with correct dimension
            base_vec = data["base_vector"]
            if dimension <= len(base_vec):
                # Truncate if dimension is smaller
                embedding = base_vec[:dimension]
            else:
                # Extend if dimension is larger (repeat pattern)
                embedding = base_vec * ((dimension // len(base_vec)) + 1)
                embedding = embedding[:dimension]
            
            # Convert vector to string format: [1.0,2.0,3.0]
            vector_str = "[" + ",".join(map(str, embedding)) + "]"
            # Convert metadata to JSON string
            metadata_str = json.dumps(data["metadata"], ensure_ascii=False).replace("'", "\\'")
            # Escape single quotes in document
            document_str = data["document"].replace("'", "\\'")
            
            # Use CAST to convert string to binary for varbinary(512) field
            sql = f"""INSERT INTO `{table_name}` (_id, document, embedding, metadata) 
                     VALUES (CAST('{id_str_escaped}' AS BINARY), '{document_str}', '{vector_str}', '{metadata_str}')"""
            client._server._execute(sql)
        
        print(f"   Inserted {len(test_data)} test records (dimension={dimension})")
    
    def test_collection_query(self, db_client):
        """
        Test collection.query() with all three client modes.
        
        This single test function automatically runs 3 times:
        - test_collection_query[embedded]
        - test_collection_query[server]
        - test_collection_query[oceanbase]
        
        No manual client creation or cleanup needed!
        """
        # Create test collection
        collection_name = f"test_query_refactored_{int(time.time() * 1000)}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        collection = db_client.get_or_create_collection(
            name=collection_name, 
            configuration=config, 
            embedding_function=None
        )
        # Get actual dimension (may be different from requested due to default embedding function)
        actual_dimension = collection.dimension
        
        # Insert test data
        self._insert_test_data(db_client, collection_name, dimension=actual_dimension)
        
        # Test 1: Basic vector similarity query
        print(f"\n✅ Testing basic query")
        # Generate query vector with correct dimension
        query_vector = [1.0, 2.0, 3.0] * ((actual_dimension // 3) + 1)
        query_vector = query_vector[:actual_dimension]
        results = collection.query(
            query_embeddings=query_vector,
            n_results=3
        )
        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) > 0
        assert len(results["ids"][0]) > 0
        print(f"   Found {len(results['ids'][0])} results")
        
        # Test 2: Query with metadata filter
        print(f"✅ Testing query with metadata filter")
        results = collection.query(
            query_embeddings=query_vector,
            where={"category": "AI"},
            n_results=5
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results with category='AI'")
        
        # Test 3: Query with document filter
        print(f"✅ Testing query with document filter")
        results = collection.query(
            query_embeddings=query_vector,
            where_document={"$contains": "machine learning"},
            n_results=5
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results containing 'machine learning'")
        
        # Test 4: Query with document filter using regex
        print(f"✅ Testing query with document filter using regex")
        results = collection.query(
            query_embeddings=query_vector,
            where_document={"$regex": ".*machine.*"},
            n_results=5
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results matching regex '.*machine.*'")
        
        # Test 5: Query with include parameter
        print(f"✅ Testing query with include parameter")
        results = collection.query(
            query_embeddings=query_vector,
            include=["documents", "metadatas"],
            n_results=3
        )
        assert results is not None
        assert "ids" in results
        if len(results["ids"][0]) > 0:
            # Check that results have the expected fields
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["ids"][0]) == len(results["documents"][0])
            assert len(results["ids"][0]) == len(results["metadatas"][0])
        
        # Test 6: Query with multiple vectors (should return dict with lists of lists)
        print(f"✅ Testing query with multiple vectors (returns dict with lists of lists)")
        query_vector2 = [2.0, 3.0, 4.0] * ((actual_dimension // 3) + 1)
        query_vector2 = query_vector2[:actual_dimension]
        results = collection.query(
            query_embeddings=[query_vector, query_vector2],
            n_results=2
        )
        assert results is not None
        assert isinstance(results, dict), "Multiple vectors should return dict"
        assert "ids" in results
        assert len(results["ids"]) == 2, f"Expected 2 ID lists, got {len(results['ids'])}"
        for i in range(len(results["ids"])):
            assert len(results["ids"][i]) > 0, f"ID list {i} should have at least one item"
            print(f"   Query {i}: {len(results['ids'][i])} items")
        
        # Test 7: Single vector returns dict with single list
        print(f"✅ Testing single vector returns dict format")
        results = collection.query(
            query_embeddings=query_vector,
            n_results=2
        )
        assert results is not None
        assert isinstance(results, dict), "Single vector should return dict"
        assert "ids" in results
        assert len(results["ids"]) == 1, "Single query should have one ID list"
        assert len(results["ids"][0]) > 0
        print(f"   Single query with {len(results['ids'][0])} items")
        
        # Test 8: Query with $in operator
        print(f"✅ Testing query with $in operator")
        results = collection.query(
            query_embeddings=query_vector,
            where={"tag": {"$in": ["ml", "python"]}},
            n_results=5
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results with tag in ['ml', 'python']")
        
        # Test 9: Query with comparison operators
        print(f"✅ Testing query with comparison operators ($gte)")
        results = collection.query(
            query_embeddings=query_vector,
            where={"score": {"$gte": 90}},
            n_results=5
        )
        assert results is not None
        assert "ids" in results
        print(f"   Found {len(results['ids'][0])} results with score >= 90")
        
        # No cleanup needed - the fixture handles it automatically!
        print(f"   ✅ All tests passed (cleanup will be automatic)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

