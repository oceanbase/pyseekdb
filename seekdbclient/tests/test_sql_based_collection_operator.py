"""
Test cases for SqlBasedCollectionOperator
Tests add, update, upsert, and delete operations with various scenarios
"""
import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Configure logging to DEBUG level for sql_based_collection_operator
# This will show debug logs from sql_based_collection_operator during tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set the logger for sql_based_collection_operator to DEBUG
sql_operator_logger = logging.getLogger('seekdbclient.client.sql_based_collection_operator')
sql_operator_logger.setLevel(logging.DEBUG)
# Also set root logger to DEBUG to see all logs
logging.getLogger().setLevel(logging.DEBUG)

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import directly from module files to avoid triggering __init__.py imports
import importlib.util

# Set up sys.modules to allow relative imports
import sys as _sys

# Import sql_utils first (dependency of sql_based_collection_operator)
sql_utils_path = project_root / "seekdbclient" / "client" / "sql_utils.py"
spec1 = importlib.util.spec_from_file_location("seekdbclient.client.sql_utils", str(sql_utils_path))
sql_utils = importlib.util.module_from_spec(spec1)
_sys.modules['seekdbclient.client.sql_utils'] = sql_utils
spec1.loader.exec_module(sql_utils)

# Import meta_info
meta_info_path = project_root / "seekdbclient" / "client" / "meta_info.py"
spec2 = importlib.util.spec_from_file_location("seekdbclient.client.meta_info", str(meta_info_path))
meta_info = importlib.util.module_from_spec(spec2)
_sys.modules['seekdbclient.client.meta_info'] = meta_info
spec2.loader.exec_module(meta_info)

# Import sql_based_collection_operator (depends on sql_utils and meta_info)
sql_operator_path = project_root / "seekdbclient" / "client" / "sql_based_collection_operator.py"
spec3 = importlib.util.spec_from_file_location("seekdbclient.client.sql_based_collection_operator", str(sql_operator_path))
sql_based_collection_operator = importlib.util.module_from_spec(spec3)
_sys.modules['seekdbclient.client.sql_based_collection_operator'] = sql_based_collection_operator
spec3.loader.exec_module(sql_based_collection_operator)

SqlBasedCollectionOperator = sql_based_collection_operator.SqlBasedCollectionOperator
CollectionFieldNames = meta_info.CollectionFieldNames
CollectionNames = meta_info.CollectionNames


class TestSqlBasedCollectionOperator:
    """Test cases for SqlBasedCollectionOperator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.mock_client.execute = Mock()
        self.mock_client.begin = Mock(return_value=MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=None)))
        self.collection_name = "test_collection"
        self.table_name = CollectionNames.table_name(self.collection_name)
    
    # ==================== ADD Tests ====================
    
    def test_add_with_documents_only(self):
        """Test add operation with documents only"""
        documents = ["doc1", "doc2"]
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            documents=documents
        )
        
        # Verify SQL was executed
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert f"INSERT INTO {self.table_name}" in sql
        assert CollectionFieldNames.DOCUMENT in sql
        assert "doc1" in sql
        assert "doc2" in sql
    
    def test_add_with_vectors_only(self):
        """Test add operation with vectors only"""
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            vectors=vectors
        )
        
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert f"INSERT INTO {self.table_name}" in sql
        assert CollectionFieldNames.EMBEDDING in sql
    
    def test_add_with_metadatas_only(self):
        """Test add operation with metadatas only"""
        metadatas = [{"key1": "value1"}, {"key2": "value2"}]
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            metadatas=metadatas
        )
        
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert f"INSERT INTO {self.table_name}" in sql
        assert CollectionFieldNames.METADATA in sql
    
    def test_add_with_all_fields(self):
        """Test add operation with all fields (ids, documents, vectors, metadatas)"""
        ids = ["id1", "id2"]
        documents = ["doc1", "doc2"]
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        metadatas = [{"key1": "value1"}, {"key2": "value2"}]
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            vectors=vectors,
            metadatas=metadatas
        )
        
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert f"INSERT INTO {self.table_name}" in sql
        assert CollectionFieldNames.ID in sql
        assert CollectionFieldNames.DOCUMENT in sql
        assert CollectionFieldNames.EMBEDDING in sql
        assert CollectionFieldNames.METADATA in sql
    
    def test_add_with_single_string_id(self):
        """Test add operation with single string id (should be converted to list)"""
        documents = ["doc1"]
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids="id1",
            documents=documents
        )
        
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert CollectionFieldNames.ID in sql
        assert "id1" in sql
    
    def test_add_with_single_document(self):
        """Test add operation with single document string (should be converted to list)"""
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            documents="doc1"
        )
        
        assert self.mock_client.execute.called
    
    def test_add_with_single_vector(self):
        """Test add operation with single vector list (should be converted to nested list)"""
        vector = [1.0, 2.0, 3.0]
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            vectors=vector
        )
        
        assert self.mock_client.execute.called
    
    def test_add_with_single_metadata(self):
        """Test add operation with single metadata dict (should be converted to list)"""
        metadata = {"key": "value"}
        
        SqlBasedCollectionOperator.add(
            client=self.mock_client,
            collection_name=self.collection_name,
            metadatas=metadata
        )
        
        assert self.mock_client.execute.called
    
    def test_add_raises_error_when_no_data(self):
        """Test that add raises ValueError when no documents, vectors, or metadatas provided"""
        with pytest.raises(ValueError, match="at least one of documents, embeddings, or metadatas must be provided"):
            SqlBasedCollectionOperator.add(
                client=self.mock_client,
                collection_name=self.collection_name
            )
    
    def test_add_raises_error_when_ids_length_mismatch(self):
        """Test that add raises ValueError when ids length doesn't match items"""
        # When ids determines num_item (1), documents mismatch (2) should be detected
        with pytest.raises(ValueError, match="number of documents"):
            SqlBasedCollectionOperator.add(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1"],
                documents=["doc1", "doc2"]
            )
    
    def test_add_raises_error_when_documents_length_mismatch(self):
        """Test that add raises ValueError when documents length doesn't match items"""
        with pytest.raises(ValueError, match="number of documents"):
            SqlBasedCollectionOperator.add(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                documents=["doc1"]
            )
    
    def test_add_raises_error_when_vectors_length_mismatch(self):
        """Test that add raises ValueError when vectors length doesn't match items"""
        with pytest.raises(ValueError, match="number of vectors"):
            SqlBasedCollectionOperator.add(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                vectors=[[1.0, 2.0]]
            )
    
    def test_add_raises_error_when_metadatas_length_mismatch(self):
        """Test that add raises ValueError when metadatas length doesn't match items"""
        with pytest.raises(ValueError, match="number of metadatas"):
            SqlBasedCollectionOperator.add(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                metadatas=[{"key": "value"}]
            )
    
    def test_add_raises_error_when_empty_items(self):
        """Test that add raises ValueError when empty items provided"""
        # Empty list is falsy, so it triggers the "must be provided" check first
        with pytest.raises(ValueError, match="at least one of documents, embeddings, or metadatas must be provided"):
            SqlBasedCollectionOperator.add(
                client=self.mock_client,
                collection_name=self.collection_name,
                documents=[]
            )
    
    # ==================== UPDATE Tests ====================
    
    def test_update_with_documents(self):
        """Test update operation with documents"""
        ids = ["id1", "id2"]
        documents = ["new_doc1", "new_doc2"]
        
        SqlBasedCollectionOperator.update(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.call_count == len(ids)
        
        # Check first update SQL
        sql1 = self.mock_client.execute.call_args_list[0][0][0]
        assert f"UPDATE {self.table_name}" in sql1
        assert CollectionFieldNames.DOCUMENT in sql1
        assert "new_doc1" in sql1
        assert "id1" in sql1
    
    def test_update_with_vectors(self):
        """Test update operation with vectors"""
        ids = ["id1"]
        vectors = [[1.0, 2.0, 3.0]]
        
        SqlBasedCollectionOperator.update(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            vectors=vectors
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_update_with_metadatas(self):
        """Test update operation with metadatas"""
        ids = ["id1"]
        metadatas = [{"key": "value"}]
        
        SqlBasedCollectionOperator.update(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            metadatas=metadatas
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_update_with_all_fields(self):
        """Test update operation with all fields"""
        ids = ["id1"]
        documents = ["doc1"]
        vectors = [[1.0, 2.0]]
        metadatas = [{"key": "value"}]
        
        SqlBasedCollectionOperator.update(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            vectors=vectors,
            metadatas=metadatas
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert CollectionFieldNames.DOCUMENT in sql
        assert CollectionFieldNames.EMBEDDING in sql
        assert CollectionFieldNames.METADATA in sql
    
    def test_update_with_single_string_id(self):
        """Test update with single string id (should be converted to list)"""
        SqlBasedCollectionOperator.update(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids="id1",
            documents=["doc1"]
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_update_raises_error_when_no_ids(self):
        """Test that update raises ValueError when ids is empty"""
        with pytest.raises(ValueError, match="ids must not be empty"):
            SqlBasedCollectionOperator.update(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=[],
                documents=["doc1"]
            )
    
    def test_update_raises_error_when_no_update_fields(self):
        """Test that update raises ValueError when no fields to update"""
        with pytest.raises(ValueError, match="You must specify at least one column to update"):
            SqlBasedCollectionOperator.update(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1"]
            )
    
    def test_update_raises_error_when_documents_length_mismatch(self):
        """Test that update raises ValueError when documents length doesn't match ids"""
        with pytest.raises(ValueError, match="number of documents"):
            SqlBasedCollectionOperator.update(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                documents=["doc1"]
            )
    
    def test_update_raises_error_when_vectors_length_mismatch(self):
        """Test that update raises ValueError when vectors length doesn't match ids"""
        with pytest.raises(ValueError, match="number of vectors"):
            SqlBasedCollectionOperator.update(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                vectors=[[1.0, 2.0]]
            )
    
    def test_update_raises_error_when_metadatas_length_mismatch(self):
        """Test that update raises ValueError when metadatas length doesn't match ids"""
        with pytest.raises(ValueError, match="number of metadatas"):
            SqlBasedCollectionOperator.update(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                metadatas=[{"key": "value"}]
            )
    
    # ==================== UPSERT Tests ====================
    
    def test_upsert_with_documents(self):
        """Test upsert operation with documents"""
        ids = ["id1", "id2"]
        documents = ["doc1", "doc2"]
        
        SqlBasedCollectionOperator.upsert(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.call_count == len(ids)
        
        # Check first upsert SQL
        sql1 = self.mock_client.execute.call_args_list[0][0][0]
        assert f"INSERT INTO {self.table_name}" in sql1
        assert "ON DUPLICATE KEY UPDATE" in sql1
        assert CollectionFieldNames.DOCUMENT in sql1
    
    def test_upsert_with_vectors(self):
        """Test upsert operation with vectors"""
        ids = ["id1"]
        vectors = [[1.0, 2.0, 3.0]]
        
        SqlBasedCollectionOperator.upsert(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            vectors=vectors
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_upsert_with_metadatas(self):
        """Test upsert operation with metadatas"""
        ids = ["id1"]
        metadatas = [{"key": "value"}]
        
        SqlBasedCollectionOperator.upsert(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            metadatas=metadatas
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_upsert_with_all_fields(self):
        """Test upsert operation with all fields"""
        ids = ["id1"]
        documents = ["doc1"]
        vectors = [[1.0, 2.0]]
        metadatas = [{"key": "value"}]
        
        SqlBasedCollectionOperator.upsert(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            vectors=vectors,
            metadatas=metadatas
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert CollectionFieldNames.ID in sql
        assert CollectionFieldNames.DOCUMENT in sql
        assert CollectionFieldNames.EMBEDDING in sql
        assert CollectionFieldNames.METADATA in sql
        assert "ON DUPLICATE KEY UPDATE" in sql
    
    def test_upsert_with_none_values(self):
        """Test upsert with None values in documents/metadatas/vectors"""
        ids = ["id1"]
        documents = [None]
        metadatas = [None]
        
        SqlBasedCollectionOperator.upsert(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_upsert_raises_error_when_no_ids(self):
        """Test that upsert raises ValueError when ids is empty"""
        with pytest.raises(ValueError, match="ids must not be empty"):
            SqlBasedCollectionOperator.upsert(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=[],
                documents=["doc1"]
            )
    
    def test_upsert_raises_error_when_no_update_fields(self):
        """Test that upsert raises ValueError when no fields to update"""
        with pytest.raises(ValueError, match="You must specify at least one column to update"):
            SqlBasedCollectionOperator.upsert(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1"]
            )
    
    def test_upsert_raises_error_when_documents_length_mismatch(self):
        """Test that upsert raises ValueError when documents length doesn't match ids"""
        with pytest.raises(ValueError, match="number of documents"):
            SqlBasedCollectionOperator.upsert(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                documents=["doc1"]
            )
    
    def test_upsert_raises_error_when_vectors_length_mismatch(self):
        """Test that upsert raises ValueError when vectors length doesn't match ids"""
        with pytest.raises(ValueError, match="number of vectors"):
            SqlBasedCollectionOperator.upsert(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                vectors=[[1.0, 2.0]]
            )
    
    def test_upsert_raises_error_when_metadatas_length_mismatch(self):
        """Test that upsert raises ValueError when metadatas length doesn't match ids"""
        with pytest.raises(ValueError, match="number of metadatas"):
            SqlBasedCollectionOperator.upsert(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1", "id2"],
                metadatas=[{"key": "value"}]
            )
    
    # ==================== DELETE Tests ====================
    
    def test_delete_with_ids(self):
        """Test delete operation with ids"""
        ids = ["id1", "id2"]
        
        # Mock _collection_get to return ids
        self.mock_client._collection_get = Mock(return_value={
            CollectionFieldNames.ID: ids
        })
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
        sql = self.mock_client.execute.call_args[0][0]
        assert f"DELETE FROM {self.table_name}" in sql
        assert "id1" in sql
        assert "id2" in sql
    
    def test_delete_with_single_id(self):
        """Test delete operation with single id string"""
        ids = "id1"
        
        self.mock_client._collection_get = Mock(return_value={
            CollectionFieldNames.ID: [ids]
        })
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids
        )
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_delete_with_where(self):
        """Test delete operation with where clause"""
        where = {"key": "value"}
        ids_to_delete = ["id1", "id2"]
        
        self.mock_client._collection_get = Mock(return_value={
            CollectionFieldNames.ID: ids_to_delete
        })
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            where=where
        )
        
        # Verify _collection_get was called with where parameter
        self.mock_client._collection_get.assert_called_once()
        call_kwargs = self.mock_client._collection_get.call_args[1]
        assert call_kwargs["where"] == where
        assert call_kwargs["include"] == [CollectionFieldNames.ID]
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_delete_with_where_document(self):
        """Test delete operation with where_document clause"""
        where_document = {"$contains": "test"}
        ids_to_delete = ["id1"]
        
        self.mock_client._collection_get = Mock(return_value={
            CollectionFieldNames.ID: ids_to_delete
        })
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            where_document=where_document
        )
        
        call_kwargs = self.mock_client._collection_get.call_args[1]
        assert call_kwargs["where_document"] == where_document
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called
    
    def test_delete_with_no_results(self):
        """Test delete operation when no results to delete"""
        self.mock_client._collection_get = Mock(return_value=None)
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=["id1"]
        )
        
        # Should not execute DELETE SQL
        assert not self.mock_client.execute.called
    
    def test_delete_with_empty_ids_list(self):
        """Test delete operation when ids list is empty"""
        self.mock_client._collection_get = Mock(return_value={
            CollectionFieldNames.ID: []
        })
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=["id1"]
        )
        
        # Should not execute DELETE SQL
        assert not self.mock_client.execute.called
    
    def test_delete_raises_error_when_id_not_in_results(self):
        """Test that delete raises ValueError when ID field not in results"""
        self.mock_client._collection_get = Mock(return_value={
            "other_field": ["value"]
        })
        
        with pytest.raises(ValueError, match="Internal Error"):
            SqlBasedCollectionOperator.delete(
                client=self.mock_client,
                collection_name=self.collection_name,
                ids=["id1"]
            )
    
    def test_delete_with_multiple_conditions(self):
        """Test delete operation with both ids and where conditions"""
        ids = ["id1"]
        where = {"key": "value"}
        ids_to_delete = ["id1", "id2"]
        
        self.mock_client._collection_get = Mock(return_value={
            CollectionFieldNames.ID: ids_to_delete
        })
        
        SqlBasedCollectionOperator.delete(
            client=self.mock_client,
            collection_name=self.collection_name,
            ids=ids,
            where=where
        )
        
        # Verify _collection_get was called with both parameters
        call_kwargs = self.mock_client._collection_get.call_args[1]
        assert call_kwargs["ids"] == ids
        assert call_kwargs["where"] == where
        
        assert self.mock_client.begin.called
        assert self.mock_client.execute.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

