import json
import unittest
from unittest.mock import MagicMock

from pymysql.converters import escape_string

from pyseekdb.client.client_base import BaseClient
from pyseekdb.client.collection import Collection


class MockClient(BaseClient):
    """Mock client for testing SQL generation without actual DB connection."""

    def __init__(self):
        """Initialize MockClient with a mock executor."""
        self._executor = MagicMock()

    def _execute(self, sql):
        """Execute SQL statement using mock executor.

        Args:
            sql: SQL statement to execute.

        Returns:
            Result from mock executor.
        """
        return self._executor(sql)

    @property
    def mode(self):
        """Return the client mode.

        Returns:
            str: Mode identifier ('mock').
        """
        return "mock"

    def is_connected(self) -> bool:
        """Check if client is connected.

        Returns:
            bool: Always True for mock client.
        """
        return True

    def get_raw_connection(self):
        """Get raw database connection.

        Returns:
            MagicMock: Mock connection object.
        """
        return MagicMock()

    def _ensure_connection(self):
        """Ensure connection is established.

        Returns:
            MagicMock: Mock connection.
        """
        return MagicMock()

    def _cleanup(self):
        """Clean up resources."""
        pass

    # Implement abstract methods with dummies
    def create_collection(self, name, configuration=None, embedding_function=None, **kwargs):
        """Create a collection (not implemented for mock)."""
        pass

    def get_collection(self, name, embedding_function=None):
        """Get a collection (not implemented for mock)."""
        pass

    def delete_collection(self, name):
        """Delete a collection (not implemented for mock)."""
        pass

    def list_collections(self):
        """List collections (not implemented for mock)."""
        pass

    def has_collection(self, name):
        """Check if collection exists (not implemented for mock)."""
        pass


class TestSpecialCharacters(unittest.TestCase):
    """Tests for special-character escaping in generated collection SQL."""

    def setUp(self):
        self.client = MockClient()
        # Mock connection check
        self.client._ensure_connection = MagicMock()
        self.client._use_context_manager_for_cursor = MagicMock(return_value=False)

        # Create a dummy collection
        self.collection_name = "test_collection"
        self.collection = Collection(client=self.client, name=self.collection_name, collection_id="test_id_123")

        # Define special test cases
        self.special_chars = [
            # SQL Injection attempts
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "admin' --",
            '"',
            "`",
            "%",
            "%%",
            "100%_complete",
            # Special syntax characters
            "\\",
            "\\\\",
            "\n",
            "\r",
            "\t",
            "\0",
            # Unicode and Languages
            "中文测试",
            "ñandú",
            "München",
            "עִבְרִית",  # Hebrew
            "العربية",  # Arabic
            # Emojis
            "😀",
            "👨‍👩‍👧‍👦",
            "🔥",
            # Whitespace
            "   ",
            " ",
        ]

    def test_ids_special_characters(self):
        """Test that IDs with special characters are correctly escaped and cast to BINARY."""
        for special_str in self.special_chars:
            # We explicitly test the internal SQL conversion method for IDs
            sql = self.client._convert_id_to_sql(special_str)

            expected_id_sql = f"CAST('{escape_string(special_str)}' AS BINARY)"
            self.assertEqual(sql, expected_id_sql)

            # Now try adding it to collection via internal method
            self.client._executor.reset_mock()
            self.client._collection_add(
                collection_id=self.collection.id,
                collection_name=self.collection.name,
                ids=[special_str],
                embeddings=[[0.1, 0.2]],  # Dummy embedding
            )

            # Check the executed SQL
            call_args = self.client._executor.call_args
            self.assertIsNotNone(call_args)
            executed_sql = call_args[0][0]
            self.assertIn("INSERT INTO", executed_sql)
            self.assertIn("_id", executed_sql)
            self.assertIn(self.collection.id, executed_sql)

            # Verify the complete ID expression is in the INSERT statement.
            self.assertIn(expected_id_sql, executed_sql)

    def test_documents_special_characters(self):
        """Test that documents with special characters are correctly escaped."""
        for special_str in self.special_chars:
            self.client._executor.reset_mock()

            self.client._collection_add(
                collection_id=self.collection.id,
                collection_name=self.collection.name,
                ids=["id_1"],
                documents=[special_str],
                embeddings=[[0.1, 0.2]],
            )

            call_args = self.client._executor.call_args
            executed_sql = call_args[0][0]
            self.assertIn("INSERT INTO", executed_sql)
            self.assertIn("_id", executed_sql)
            self.assertIn(self.collection.id, executed_sql)

            expected_doc_sql = f"'{escape_string(special_str)}'"
            self.assertIn(expected_doc_sql, executed_sql)

    def test_metadata_special_characters(self):
        """Test that metadata keys and values with special characters are correctly handled."""
        for special_str in self.special_chars:
            self.client._executor.reset_mock()

            # Test as value: {key: special_str}
            metadata_value = {"key": special_str}
            self.client._collection_add(
                collection_id=self.collection.id,
                collection_name=self.collection.name,
                ids=["id_val"],
                embeddings=[[0.1, 0.2]],
                metadatas=[metadata_value],
            )
            executed_sql = self.client._executor.call_args[0][0]
            self.assertIn("INSERT INTO", executed_sql)
            self.assertIn(self.collection.id, executed_sql)
            expected_value_sql = f"'{escape_string(json.dumps(metadata_value, ensure_ascii=False))}'"
            self.assertIn(expected_value_sql, executed_sql)

            # Test as key: {special_str: "value"}
            # JSON keys must be strings, so we still need to verify the key is
            # JSON-serialized and SQL-escaped alongside the document payload.
            metadata_key = {special_str: "value"}
            self.client._executor.reset_mock()
            self.client._collection_add(
                collection_id=self.collection.id,
                collection_name=self.collection.name,
                ids=["id_key"],
                documents=["doc"],
                embeddings=[[0.1, 0.2]],
                metadatas=[metadata_key],
            )
            executed_sql = self.client._executor.call_args[0][0]
            self.assertIn("INSERT INTO", executed_sql)
            expected_key_sql = f"'{escape_string(json.dumps(metadata_key, ensure_ascii=False))}'"
            self.assertIn(expected_key_sql, executed_sql)


if __name__ == "__main__":
    unittest.main()
