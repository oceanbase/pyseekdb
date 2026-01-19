"""
Integration tests to reproduce bugs with special characters in insert operations.

Bug reports:
1. If there's '\' character in the documents parameter, it failed to insert data into database.
2. If there's '%' in the id parameter, it failed.
3. If there's '"' character in the metadata parameter, it failed to insert data into database.
"""

import pytest
import time
import uuid

import pyseekdb
from pyseekdb import HNSWConfiguration


class TestSpecialCharactersBugs:
    """Test special characters handling in insert operations"""

    def test_backslash_in_documents(self, db_client):
        """
        Bug reproduction: If there's '\' character in the documents parameter,
        it failed to insert data into database.
        """
        # Create test collection
        collection_name = f"test_backslash_doc_{int(time.time() * 1000)}"
        dimension = 3

        # Create collection using proper API
        config = HNSWConfiguration(dimension=dimension, distance="cosine")
        collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Test cases with backslashes in documents
        test_cases = [
            # Single backslash
            "Single\\backslash",
            "\\Backslash at start",
            "Backslash at end\\",

            # Multiple consecutive backslashes
            "Double\\\\backslash",
            "Triple\\\\\\backslash",
            "Four\\\\\\\\backslashes",
            "Five\\\\\\\\\\backslashes",
            "Many\\\\\\\\\\\\backslashes",

            # Backslashes in paths (common use case)
            "Path: C:\\Users\\Documents\\file.txt",
            "Unix path: /home/user\\file.txt",
            "Network path: \\\\server\\share\\file.txt",
            "Relative path: ..\\..\\parent\\file.txt",
            "Deep path: C:\\Users\\Documents\\Projects\\2024\\file.txt",

            # Multiple backslashes in different positions
            "Multiple\\backslashes\\here",
            "Start\\middle\\end",
            "\\a\\b\\c\\d\\e",
            "Text\\with\\many\\separated\\backslashes",

            # Backslashes with escape sequences
            "Escaped\\nnewline",
            "Escaped\\ttab",
            "Escaped\\rreturn",
            "Mixed\\t\\r\\ncharacters",
            "All\\n\\t\\r\\v\\f",
            "test single quote: \\'",

            # Consecutive backslashes in escape sequences
            "Double\\\\nnewline",
            "Triple\\\\\\nnewline",
            "Mixed\\\\t\\n\\r",

            # Edge cases
            "Only\\\\backslashes",
            "\\",
            "\\\\",
            "\\\\\\",
            "\\\\\\\\",
            "\\\\\\\\\\",

            # Backslashes with other special characters
            "Backslash\\and%percent",
            "Backslash\\and\"quote",
            "Backslash\\and'apostrophe",
            "Backslash\\and\\backslash",

            # Long strings with many backslashes
            "\\".join(["part1", "part2", "part3", "part4", "part5"]),
            "C:\\" + "\\".join([f"folder{i}" for i in range(10)]),

            # Real-world scenarios
            "Windows path: C:\\Program Files\\MyApp\\config\\settings.ini",
            "Regex pattern: \\d+\\s+\\w+",
            "JSON string: {\"path\": \"C:\\\\Users\\\\file.txt\"}",
            "Command: cd C:\\Users\\Documents && dir",
        ]

        print(f"\n🔍 Testing backslash in documents parameter")
        for i, doc_with_backslash in enumerate(test_cases):
            test_id = f"test_backslash_{i}_{int(time.time() * 1000)}"
            print(f"  Testing: {repr(doc_with_backslash)}")

            try:
                # Attempt to add document with backslash
                collection.add(
                    ids=test_id,
                    embeddings=[1.0, 2.0, 3.0],
                    documents=doc_with_backslash,
                    metadatas={"test": "backslash"},
                )

                # Verify insertion succeeded
                results = collection.get(ids=test_id)
                assert len(results["ids"]) == 1, f"Failed to insert document with backslash: {doc_with_backslash}"
                assert results["documents"][0] == doc_with_backslash, f"Document content mismatch: expected {repr(doc_with_backslash)}, got {repr(results['documents'][0])}"
                print(f"    ✅ Successfully inserted and verified: {repr(doc_with_backslash)}")
            except Exception as e:
                print(f"    ❌ FAILED to insert document with backslash: {repr(doc_with_backslash)}")
                print(f"       Error: {e}")
                raise

    def test_percent_in_id(self, db_client):
        """
        Bug reproduction: If there's '%' in the id parameter, it failed.
        """
        # Create test collection
        collection_name = f"test_percent_id_{int(time.time() * 1000)}"
        dimension = 3

        # Create collection using proper API
        config = HNSWConfiguration(dimension=dimension, distance="cosine")
        collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Test cases with percent signs in IDs
        test_cases = [
            "id_with_%_percent",
            "100%_complete",
            "%percent_at_start",
            "percent_at_end%",
            "multiple%percent%signs",
            "50%_off_sale",
            "%",
            "%%",
            "%%%",
        ]

        print(f"\n🔍 Testing percent sign in id parameter")
        for i, id_with_percent in enumerate(test_cases):
            print(f"  Testing ID: {repr(id_with_percent)}")

            try:
                # Attempt to add with ID containing percent
                collection.add(
                    ids=id_with_percent,
                    embeddings=[1.0, 2.0, 3.0],
                    documents="Test document",
                    metadatas={"test": "percent"},
                )

                # Verify insertion succeeded
                results = collection.get(ids=id_with_percent)
                assert len(results["ids"]) == 1, f"Failed to insert with ID containing percent: {id_with_percent}"
                assert results["ids"][0] == id_with_percent, f"ID mismatch: expected {repr(id_with_percent)}, got {repr(results['ids'][0])}"
                print(f"    ✅ Successfully inserted and verified ID: {repr(id_with_percent)}")
            except ValueError as e:
                # This is the expected bug - ValueError about unsupported format character
                error_msg = str(e)
                if "unsupported format character" in error_msg or "%" in error_msg:
                    print(f"    ❌ BUG REPRODUCED: Failed to insert with ID containing percent: {repr(id_with_percent)}")
                    print(f"       Error type: {type(e).__name__}")
                    print(f"       Error message: {error_msg}")
                    print(f"       This confirms the bug - percent signs in IDs cause formatting errors")
                    raise
                else:
                    # Different ValueError, re-raise
                    raise
            except Exception as e:
                print(f"    ❌ FAILED to insert with ID containing percent: {repr(id_with_percent)}")
                print(f"       Error type: {type(e).__name__}")
                print(f"       Error message: {e}")
                raise

    def test_double_quote_in_metadata(self, db_client):
        """
        Bug reproduction: If there's '"' character in the metadata parameter,
        it failed to insert data into database.
        """
        # Create test collection
        collection_name = f"test_quote_meta_{int(time.time() * 1000)}"
        dimension = 3

        # Create collection using proper API
        config = HNSWConfiguration(dimension=dimension, distance="cosine")
        collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        # Test cases with double quotes in metadata
        test_cases = [
            {"title": 'Book "The Great Gatsby"'},
            {"description": 'He said "Hello World"'},
            {"quote": '"To be or not to be"'},
            {"text": 'Multiple "quotes" in "one" string'},
            {"value": '"'},
            {"nested": {"inner": '"quoted"'}},
            {"mixed": 'Start "middle" end'},
            {"json_like": '{"key": "value"}'},
        ]

        print(f"\n🔍 Testing double quote in metadata parameter")
        for i, metadata_with_quote in enumerate(test_cases):
            test_id = f"test_quote_{i}_{int(time.time() * 1000)}"
            print(f"  Testing metadata: {repr(metadata_with_quote)}")

            try:
                # Attempt to add with metadata containing double quotes
                collection.add(
                    ids=test_id,
                    embeddings=[1.0, 2.0, 3.0],
                    documents="Test document",
                    metadatas=metadata_with_quote,
                )

                # Verify insertion succeeded
                results = collection.get(ids=test_id)
                assert len(results["ids"]) == 1, f"Failed to insert with metadata containing double quote: {metadata_with_quote}"
                assert results["metadatas"][0] == metadata_with_quote, f"Metadata mismatch: expected {repr(metadata_with_quote)}, got {repr(results['metadatas'][0])}"
                print(f"    ✅ Successfully inserted and verified metadata: {repr(metadata_with_quote)}")
            except Exception as e:
                print(f"    ❌ FAILED to insert with metadata containing double quote: {repr(metadata_with_quote)}")
                print(f"       Error: {e}")
                raise

    def test_all_special_characters_combined(self, db_client):
        """
        Combined test: Test all three special characters together
        """
        # Create test collection
        collection_name = f"test_all_special_{int(time.time() * 1000)}"
        dimension = 3

        # Create collection using proper API
        config = HNSWConfiguration(dimension=dimension, distance="cosine")
        collection = db_client.create_collection(
            name=collection_name, configuration=config, embedding_function=None
        )

        print(f"\n🔍 Testing all special characters combined")
        test_id = "id_with_%_percent"
        test_document = "Path: C:\\Users\\Documents\\file.txt"
        test_metadata = {"title": 'Book "The Great Gatsby"', "path": "C:\\Users\\Documents"}

        try:
            collection.add(
                ids=test_id,
                embeddings=[1.0, 2.0, 3.0],
                documents=test_document,
                metadatas=test_metadata,
            )

            # Verify insertion succeeded
            results = collection.get(ids=test_id)
            assert len(results["ids"]) == 1, "Failed to insert with all special characters"
            assert results["ids"][0] == test_id, f"ID mismatch"
            assert results["documents"][0] == test_document, f"Document mismatch"
            assert results["metadatas"][0] == test_metadata, f"Metadata mismatch"
            print(f"    ✅ Successfully inserted and verified all special characters combined")
        except Exception as e:
            print(f"    ❌ FAILED to insert with all special characters combined")
            print(f"       Error: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
