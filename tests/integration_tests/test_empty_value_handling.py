"""
Empty value handling tests - testing upsert operations with empty strings, empty lists, and None values
Tests the fixes for falsy value handling bugs in _collection_upsert method using db_client fixture
"""

import time

import pytest

import pyseekdb


class TestEmptyValueHandling:
    """Test empty value handling in upsert operations for all three modes"""

    def test_empty_value_handling(self, db_client):
        """
        Test empty value handling with all client modes.

        Tests upsert operations with:
        - Empty strings
        - Whitespace strings
        - String "0" and "false"
        - Empty metadata
        - None values

        Automatically runs for: embedded, server, oceanbase
        """
        collection_name = f"test_empty_values_{int(time.time() * 1000)}"
        collection = db_client.get_or_create_collection(
            name=collection_name, embedding_function=pyseekdb.DefaultEmbeddingFunction()
        )

        try:
            # Test 1: Upsert update path - empty string document (Line 1072 fix)
            self._test_upsert_update_empty_document(collection)

            # Test 2: Upsert insert path - empty string document (Line 1093 fix)
            self._test_upsert_insert_empty_document(collection)

            # Test 3: Upsert update path - empty metadata (should work correctly)
            self._test_upsert_empty_metadata(collection)

            # Test 4: Add method baseline - should work correctly
            self._test_add_empty_values_baseline(collection)

            # Test 5: Mixed empty and non-empty values in batch operations
            self._test_mixed_empty_values_batch(collection)

            print("✅ All empty value tests passed")
        finally:
            # Cleanup
            try:
                db_client.delete_collection(name=collection_name)
            except Exception as cleanup_error:
                print(f"Warning: failed to cleanup collection '{collection_name}': {cleanup_error}")

    def _test_upsert_update_empty_document(self, collection):
        """Test upsert update path with empty string document (Line 1072 fix)"""
        print("\n🔍 Testing upsert update path - empty document")

        # Add initial document
        test_id = f"update_test_{int(time.time() * 1000)}"
        collection.add(
            ids=[test_id],
            documents=["original document"],
            metadatas=[{"test": "update_path"}],
        )

        # Test cases for empty document values
        test_cases = [
            ("", "empty string"),
            ("   ", "whitespace string"),
            ("0", "string zero"),
            ("false", "string false"),
        ]

        for doc_value, description in test_cases:
            print(f"   Testing {description}: {doc_value!r}")

            # Upsert with empty/falsy document value (triggers Line 1072 fix)
            collection.upsert(
                ids=[test_id],
                documents=[doc_value],
                metadatas=[{"test": "updated", "case": description}],
            )

            # Verify the document was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should find exactly one result for {description}"

            actual_doc = results["documents"][0]
            assert actual_doc == doc_value, f"Expected {doc_value!r}, got {actual_doc!r} for {description}"

            # Ensure it's not the literal string 'NULL'
            assert actual_doc != "NULL", f"Document should not be literal 'NULL' string for {description}"

            print(f"   ✅ {description} correctly stored as {actual_doc!r}")

    def _test_upsert_insert_empty_document(self, collection):
        """Test upsert insert path with empty string document (Line 1093 fix)"""
        print("\n🔍 Testing upsert insert path - empty document")

        # Test cases for empty document values in insert path
        test_cases = [
            ("", "empty string"),
            ("   ", "whitespace string"),
            ("0", "string zero"),
            ("false", "string false"),
        ]

        for i, (doc_value, description) in enumerate(test_cases):
            test_id = f"insert_test_{i}_{int(time.time() * 1000)}"
            print(f"   Testing {description}: {doc_value!r}")

            # Upsert non-existing record (triggers Line 1093 fix)
            collection.upsert(
                ids=[test_id],
                documents=[doc_value],
                metadatas=[{"test": "insert_path", "case": description}],
            )

            # Verify the document was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should create exactly one result for {description}"

            actual_doc = results["documents"][0]
            assert actual_doc == doc_value, f"Expected {doc_value!r}, got {actual_doc!r} for {description}"

            # Ensure it's not None/NULL
            assert actual_doc is not None, f"Document should not be None for {description}"

            print(f"   ✅ {description} correctly stored as {actual_doc!r}")

    def _test_upsert_empty_metadata(self, collection):
        """Test upsert with empty metadata (should work correctly)"""
        print("\n🔍 Testing upsert empty metadata")

        test_id = f"meta_test_{int(time.time() * 1000)}"

        # Add initial document
        collection.add(ids=[test_id], documents=["test document"], metadatas=[{"initial": "value"}])

        # Test cases for metadata values
        test_cases = [
            ({}, "empty dict"),
            ({"": ""}, "dict with empty string key/value"),
            ({"key": ""}, "dict with empty string value"),
            ({"key": None}, "dict with None value"),
            ({"key": 0}, "dict with zero value"),
            ({"key": False}, "dict with False value"),
        ]

        for meta_value, description in test_cases:
            print(f"   Testing {description}: {meta_value}")

            # Upsert with test metadata
            collection.upsert(ids=[test_id], documents=["updated document"], metadatas=[meta_value])

            # Verify the metadata was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should find exactly one result for {description}"

            actual_meta = results["metadatas"][0]
            assert actual_meta == meta_value, f"Expected {meta_value}, got {actual_meta} for {description}"

            print(f"   ✅ {description} correctly stored as {actual_meta}")

    def _test_add_empty_values_baseline(self, collection):
        """Test add method with empty values as baseline (should work correctly)"""
        print("\n🔍 Testing add method baseline - empty values")

        # Test cases for add method with empty values
        test_cases = [
            ("", "empty string"),
            ("   ", "whitespace string"),
            ("0", "string zero"),
        ]

        for i, (doc_value, description) in enumerate(test_cases):
            test_id = f"add_baseline_{i}_{int(time.time() * 1000)}"
            print(f"   Testing {description}: {doc_value!r}")

            # Add with empty document value (baseline test)
            collection.add(
                ids=[test_id],
                documents=[doc_value],
                metadatas=[{"test": "add_baseline", "case": description}],
            )

            # Verify the document was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should find exactly one result for {description}"

            actual_doc = results["documents"][0]
            assert actual_doc == doc_value, f"Expected {doc_value!r}, got {actual_doc!r} for {description}"

            print(f"   ✅ {description} correctly stored as {actual_doc!r}")

    def _test_mixed_empty_values_batch(self, collection):
        """Test mixed empty and non-empty values in batch operations"""
        print("🔍 Testing mixed empty and non-empty values in batch")

        test_ids = [f"mixed_{i}_{int(time.time() * 1000)}" for i in range(4)]
        test_docs = ["normal doc", "", "   ", "another normal doc"]
        test_metas = [{"type": "normal"}, {}, {"empty": ""}, {"type": "normal2"}]

        # Add initial batch
        collection.add(ids=test_ids, documents=test_docs, metadatas=test_metas)

        # Upsert with mixed empty values
        updated_docs = ["", "updated normal", "", "final doc"]
        updated_metas = [{"updated": True}, {}, {"key": ""}, {"final": True}]

        collection.upsert(ids=test_ids, documents=updated_docs, metadatas=updated_metas)

        # Verify all values were stored correctly
        results = collection.get(ids=test_ids, include=["documents", "metadatas"])
        assert len(results["ids"]) == 4, "Should find all 4 results"

        for i, (expected_doc, expected_meta) in enumerate(zip(updated_docs, updated_metas, strict=True)):
            actual_doc = results["documents"][i]
            actual_meta = results["metadatas"][i]

            assert actual_doc == expected_doc, f"Document {i}: expected {expected_doc!r}, got {actual_doc!r}"
            assert actual_meta == expected_meta, f"Metadata {i}: expected {expected_meta}, got {actual_meta}"

        print("✅ Mixed empty and non-empty values handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
