"""
Security test for SQL injection prevention (Issue #62)

Tests the escape_string fixes across all database operations to ensure:
1. SQL injection attacks are safely neutralized
2. Special characters are correctly handled
3. Data integrity is preserved
4. All escape_string fixes work correctly

This test covers ADD, UPDATE, UPSERT, QUERY, and GET operations using db_client fixture.
"""

from typing import Any

import pytest


class TestSecuritySQLInjection:
    """Security test class for SQL injection prevention"""

    def get_security_test_cases(self) -> list[dict[str, Any]]:
        """Get test cases with various security attack vectors"""
        return [
            {
                "id": "test_quotes",
                "document": "It's a test with 'single' and \"double\" quotes",
                "metadata": {"type": "quotes", "note": "Testing 'quotes'"},
                "description": "Single and double quotes",
            },
            {
                "id": "test_backslash",
                "document": "Path: C:\\Users\\test\\file.txt with \\backslashes",
                "metadata": {"type": "backslash", "path": "C:\\Windows\\System32"},
                "description": "Backslash characters",
            },
            {
                "id": "test_sql_injection_1",
                "document": "'; DROP TABLE users; --",
                "metadata": {"type": "injection", "attack": "'; DELETE FROM *; --"},
                "description": "SQL injection - table drop attempt",
            },
            {
                "id": "test_sql_injection_2",
                "document": "1' OR '1'='1",
                "metadata": {"type": "injection", "condition": "1' OR 1=1 --"},
                "description": "SQL injection - condition bypass",
            },
            {
                "id": "test_special_chars",
                "document": "Mixed: 'quotes', \\backslash, \nnewline, \ttab",
                "metadata": {"type": "special", "chars": "\\n\\t\\r'\""},
                "description": "Mixed special characters",
            },
            {
                "id": "test_unicode",
                "document": "中文测试 🚀 emoji with 'quotes'",
                "metadata": {"type": "unicode", "lang": "中文", "emoji": "🚀"},
                "description": "Unicode and emoji characters",
            },
        ]

    def verify_data_integrity(self, collection, test_case: dict[str, Any]) -> bool:
        """Verify that data was stored and retrieved correctly"""
        try:
            result = collection.get(ids=[test_case["id"]], include=["documents", "metadatas"])

            if not result or len(result["ids"]) == 0:
                return False

            retrieved_doc = result["documents"][0] if result.get("documents") else None
            retrieved_meta = result["metadatas"][0] if result.get("metadatas") else None

            doc_match = retrieved_doc == test_case["document"]
            meta_match = retrieved_meta == test_case["metadata"]
        except Exception:
            return False
        else:
            return doc_match and meta_match

    def test_security_sql_injection(self, db_client):
        """
        Security SQL injection tests for all client modes.

        Tests all database operations (ADD, UPDATE, UPSERT, QUERY, GET) with:
        - Single and double quotes
        - Backslash characters
        - SQL injection attempts
        - Special characters (newlines, tabs)
        - Unicode and emoji

        Automatically runs for: embedded, server, oceanbase
        """
        print("\n🔒 Running security SQL injection tests")

        # Test each operation separately with its own collection to avoid ID conflicts

        # Test 1: ADD operation
        self._run_single_test(db_client, "add_test", self._test_add_operation_security)

        # Test 2: UPDATE operation
        self._run_single_test(db_client, "update_test", self._test_update_operation_security)

        # Test 3: UPSERT operation
        self._run_single_test(db_client, "upsert_test", self._test_upsert_operation_security)

        # Test 4: QUERY operation
        self._run_single_test(db_client, "query_test", self._test_query_operation_security)

        # Test 5: GET operation
        self._run_single_test(db_client, "get_test", self._test_get_operation_security)

        # Test 6: Comprehensive workflow
        self._run_single_test(db_client, "comprehensive_test", self._test_comprehensive_security_workflow)

        print("✅ All security SQL injection tests passed")

    def _run_single_test(self, client, test_name, test_method):
        """Run a single test with its own collection"""
        collection_name = f"security_{test_name}_collection"

        # Create new collection
        collection = client.get_or_create_collection(name=collection_name)

        try:
            # Run the test method
            test_method(collection)
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
            except Exception as cleanup_error:
                print(f"Warning: failed to cleanup collection '{collection_name}': {cleanup_error}")

    def _test_add_operation_security(self, collection):
        """Test ADD operation with security attack vectors"""
        print("\n  🧪 Testing ADD operation security")
        test_cases = self.get_security_test_cases()

        for test_case in test_cases:
            # Test ADD operation
            collection.add(
                ids=[test_case["id"]],
                documents=[test_case["document"]],
                metadatas=[test_case["metadata"]],
            )

            # Verify data integrity
            assert self.verify_data_integrity(collection, test_case), (
                f"Data integrity failed for ADD operation: {test_case['description']}"
            )

        print(f"     ✓ ADD operation passed ({len(test_cases)} test cases)")

    def _test_update_operation_security(self, collection):
        """Test UPDATE operation with security attack vectors"""
        print("\n  🧪 Testing UPDATE operation security")
        test_cases = self.get_security_test_cases()

        # First add the data
        for test_case in test_cases:
            collection.add(
                ids=[test_case["id"]],
                documents=[test_case["document"]],
                metadatas=[test_case["metadata"]],
            )

        # Then update with modified security payloads
        for test_case in test_cases:
            updated_doc = f"UPDATED: {test_case['document']}"
            updated_meta = {
                **test_case["metadata"],
                "updated": True,
                "note": "Updated with 'quotes'",
            }

            # Test UPDATE operation
            collection.update(ids=[test_case["id"]], documents=[updated_doc], metadatas=[updated_meta])

            # Verify updated data integrity
            updated_test_case = {
                "id": test_case["id"],
                "document": updated_doc,
                "metadata": updated_meta,
            }
            assert self.verify_data_integrity(collection, updated_test_case), (
                f"Data integrity failed for UPDATE operation: {test_case['description']}"
            )

        print(f"     ✓ UPDATE operation passed ({len(test_cases)} test cases)")

    def _test_upsert_operation_security(self, collection):
        """Test UPSERT operation with security attack vectors"""
        print("\n  🧪 Testing UPSERT operation security")
        test_cases = self.get_security_test_cases()

        # Test UPSERT for new records
        for test_case in test_cases:
            collection.upsert(
                ids=[test_case["id"]],
                documents=[test_case["document"]],
                metadatas=[test_case["metadata"]],
            )

            # Verify data integrity
            assert self.verify_data_integrity(collection, test_case), (
                f"Data integrity failed for UPSERT (new) operation: {test_case['description']}"
            )

        # Test UPSERT for existing records
        for test_case in test_cases:
            upserted_doc = f"UPSERTED: {test_case['document']}"
            upserted_meta = {
                **test_case["metadata"],
                "upserted": True,
                "note": "Upserted with \\backslash",
            }

            collection.upsert(
                ids=[test_case["id"]],
                documents=[upserted_doc],
                metadatas=[upserted_meta],
            )

            # Verify upserted data integrity
            upserted_test_case = {
                "id": test_case["id"],
                "document": upserted_doc,
                "metadata": upserted_meta,
            }
            assert self.verify_data_integrity(collection, upserted_test_case), (
                f"Data integrity failed for UPSERT (existing) operation: {test_case['description']}"
            )

        print(f"     ✓ UPSERT operation passed ({len(test_cases)} test cases)")

    def _test_query_operation_security(self, collection):
        """Test QUERY operation with security attack vectors"""
        print("\n  🧪 Testing QUERY operation security")
        test_cases = self.get_security_test_cases()

        # First add test data
        for test_case in test_cases:
            collection.add(
                ids=[test_case["id"]],
                documents=[test_case["document"]],
                metadatas=[test_case["metadata"]],
            )

        # Test queries with special characters
        query_tests = [
            {"name": "Query with single quote", "text": "it's"},
            {"name": "Query with quotes", "text": "'quotes'"},
            {"name": "Query with backslash", "text": "backslash"},
            {"name": "Query with injection attempt", "text": "'; DROP TABLE"},
        ]

        for query_test in query_tests:
            # This should not raise any SQL errors
            try:
                results = collection.query(
                    query_texts=[query_test["text"]],
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )

                # Results can be empty, but query should execute successfully
                assert isinstance(results, dict), f"Query failed for: {query_test['name']}"
                assert "ids" in results, f"Query result missing 'ids' for: {query_test['name']}"

            except Exception as e:
                pytest.fail(f"Query operation failed with security payload '{query_test['text']}': {e}")

        print(f"     ✓ QUERY operation passed ({len(query_tests)} test cases)")

    def _test_get_operation_security(self, collection):
        """Test GET operation with security attack vectors in IDs"""
        print("\n  🧪 Testing GET operation security")

        # Test with special character IDs
        special_ids = [
            "id_with_'quote",
            "id_with_\\backslash",
            'id_with_"double_quote',
            "id'; DROP TABLE users; --",
        ]

        # Add data with special IDs
        for i, special_id in enumerate(special_ids):
            collection.add(
                ids=[special_id],
                documents=[f"Document for {special_id}"],
                metadatas=[{"id_type": "special", "index": i}],
            )

        # Test GET operations with special IDs
        for special_id in special_ids:
            try:
                result = collection.get(ids=[special_id], include=["documents", "metadatas"])

                # Should successfully retrieve the data
                assert result is not None, f"GET failed for special ID: {special_id}"
                assert len(result["ids"]) > 0, f"No data retrieved for special ID: {special_id}"

            except Exception as e:
                pytest.fail(f"GET operation failed with special ID '{special_id}': {e}")

        print(f"     ✓ GET operation passed ({len(special_ids)} test cases)")

    def _test_comprehensive_security_workflow(self, collection):
        """Test a comprehensive workflow with all operations and security payloads"""
        print("\n  🧪 Testing comprehensive security workflow")

        # This is a comprehensive test that combines all operations
        test_case = {
            "id": "comprehensive_'; DROP TABLE test; --",
            "document": "Comprehensive test: It's got 'quotes', \\backslashes, and 1' OR '1'='1",
            "metadata": {
                "type": "comprehensive",
                "attack_vectors": ["quotes", "backslashes", "sql_injection"],
                "payload": "'; DELETE FROM users WHERE '1'='1'; --",
                "path": "C:\\Windows\\System32\\evil.exe",
            },
        }

        # 1. ADD with security payload
        collection.add(
            ids=[test_case["id"]],
            documents=[test_case["document"]],
            metadatas=[test_case["metadata"]],
        )
        assert self.verify_data_integrity(collection, test_case)

        # 2. UPDATE with security payload
        updated_doc = f"UPDATED: {test_case['document']} with more 'attacks'"
        updated_meta = {
            **test_case["metadata"],
            "updated": True,
            "new_attack": "1' UNION SELECT * FROM users --",
        }

        collection.update(ids=[test_case["id"]], documents=[updated_doc], metadatas=[updated_meta])

        updated_test_case = {
            "id": test_case["id"],
            "document": updated_doc,
            "metadata": updated_meta,
        }
        assert self.verify_data_integrity(collection, updated_test_case)

        # 3. QUERY with security payload
        results = collection.query(
            query_texts=["'; DROP TABLE"],
            n_results=1,
            include=["documents", "metadatas"],
        )
        assert isinstance(results, dict)

        # 4. GET with security payload ID
        result = collection.get(ids=[test_case["id"]], include=["documents", "metadatas"])
        assert result is not None
        assert len(result["ids"]) > 0

        print("     ✓ Comprehensive workflow passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
