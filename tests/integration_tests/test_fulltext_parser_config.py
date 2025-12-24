"""
Tests for fulltext parser configuration in collection creation - REFACTORED using db_client fixture
"""
import pytest
import time

import pyseekdb
from pyseekdb import (
    Configuration,
    HNSWConfiguration,
    FulltextParserConfig
)


class TestFulltextParserConfigRefactored:
    """Test fulltext parser configuration using parameterized db_client fixture"""

    def _test_fulltext_parser_config(self, client, parser_name: str, params: dict = None):
        """
        Test creating a collection with a specific fulltext parser configuration

        Args:
            client: Client proxy object
            parser_name: Parser name ('ik', 'space', 'ngram', 'ngram2', 'beng')
            params: Optional parser parameters
        """
        test_collection_name = f"test_fulltext_{parser_name}_{int(time.time() * 1000)}"
        test_dimension = 128

        # Create configuration with fulltext parser
        fulltext_config = FulltextParserConfig(parser=parser_name, params=params)
        config = Configuration(
            hnsw=HNSWConfiguration(dimension=test_dimension, distance='cosine'),
            fulltext_config=fulltext_config
        )

        # Create collection
        collection = client.create_collection(
            name=test_collection_name,
            configuration=config,
            embedding_function=None
        )

        # Verify collection was created
        assert collection is not None
        assert collection.name == test_collection_name
        assert collection.dimension == test_dimension

        # Verify the fulltext index was created with correct parser
        from pyseekdb.client.meta_info import CollectionNames
        table_name = CollectionNames.table_name(test_collection_name)
        try:
            # Get CREATE TABLE statement
            create_table_result = client._server._execute(f"SHOW CREATE TABLE `{table_name}`")
            assert create_table_result is not None
            assert len(create_table_result) > 0

            # Extract CREATE TABLE statement
            if isinstance(create_table_result[0], dict):
                create_stmt = create_table_result[0].get('Create Table', create_table_result[0].get('create table', ''))
            elif isinstance(create_table_result[0], (tuple, list)):
                create_stmt = create_table_result[0][1] if len(create_table_result[0]) > 1 else ''
            else:
                create_stmt = str(create_table_result[0])

            # Verify FULLTEXT INDEX contains the parser
            assert 'FULLTEXT KEY' in create_stmt.upper()
            assert f'PARSER {parser_name}'.upper() in create_stmt.upper()

            # If params are provided, verify they're in the SQL
            if params:
                for key, value in params.items():
                    # Check if parameter appears in SQL (may be formatted differently)
                    param_pattern = f"{key}={value}" if not isinstance(value, str) else f"{key}='{value}'"
                    # Also check without quotes for string values
                    if isinstance(value, str):
                        assert key in create_stmt or param_pattern in create_stmt
                    else:
                        assert str(value) in create_stmt

            print(f"\n✅ Collection '{test_collection_name}' created with parser '{parser_name}'")
            print(f"   CREATE TABLE statement contains: PARSER {parser_name}")

        except Exception as e:
            # Clean up and fail
            try:
                client._server._execute(f"DROP TABLE IF EXISTS `{table_name}`")
            except Exception:
                pass
            pytest.fail(f"Failed to verify fulltext parser configuration: {e}")

        # Clean up
        try:
            client.delete_collection(test_collection_name)
        except Exception:
            pass

    def _test_default_parser(self, client):
        """Test that default parser (ik) is used when not specified"""
        test_collection_name = f"test_default_parser_{int(time.time() * 1000)}"
        test_dimension = 128

        # Create collection without fulltext_config (should default to ik)
        config = Configuration(
            hnsw=HNSWConfiguration(dimension=test_dimension, distance='cosine')
            # fulltext_config not specified, should default to ik
        )

        collection = client.create_collection(
            name=test_collection_name,
            configuration=config,
            embedding_function=None
        )

        assert collection is not None

        # Verify default parser (ik) is used  
        table_name = CollectionNames.table_name(test_collection_name)
        try:
            create_table_result = client._server._execute(f"SHOW CREATE TABLE `{table_name}`")
            create_stmt = create_table_result[0][1] if isinstance(create_table_result[0], (tuple, list)) else \
                create_table_result[0].get('Create Table', create_table_result[0].get('create table', ''))

            assert 'PARSER IK' in create_stmt.upper()
            print(f"\n✅ Default parser (ik) used correctly")

        except Exception as e:
            try:
                client._server._execute(f"DROP TABLE IF EXISTS `{table_name}`")
            except Exception:
                pass
            pytest.fail(f"Failed to verify default parser: {e}")

        # Clean up
        try:
            client.delete_collection(test_collection_name)
        except Exception:
            pass

    def _test_backward_compatibility(self, client):
        """Test backward compatibility with HNSWConfiguration (no fulltext config)"""
        test_collection_name = f"test_backward_compat_{int(time.time() * 1000)}"
        test_dimension = 128

        # Create collection with HNSWConfiguration only (old style)
        config = HNSWConfiguration(dimension=test_dimension, distance='cosine')

        collection = client.create_collection(
            name=test_collection_name,
            configuration=config,
            embedding_function=None
        )

        assert collection is not None

        # Verify default parser (ik) is used for backward compatibility
        table_name = CollectionNames.table_name(test_collection_name)
        try:
            create_table_result = client._server._execute(f"SHOW CREATE TABLE `{table_name}`")
            create_stmt = create_table_result[0][1] if isinstance(create_table_result[0], (tuple, list)) else \
                create_table_result[0].get('Create Table', create_table_result[0].get('create table', ''))

            assert 'PARSER IK' in create_stmt.upper()
            print(f"\n✅ Backward compatibility: HNSWConfiguration defaults to ik parser")

        except Exception as e:
            try:
                client._server._execute(f"DROP TABLE IF EXISTS `{table_name}`")
            except Exception:
                pass
            pytest.fail(f"Failed to verify backward compatibility: {e}")

        # Clean up
        try:
            client.delete_collection(test_collection_name)
        except Exception:
            pass

    def test_fulltext_parser(self, db_client):
        """
        Test all fulltext parsers with various configurations.
        
        Automatically runs for: embedded, server, oceanbase
        """
        # Test all supported parsers
        parsers = ['ik', 'space', 'ngram', 'ngram2', 'beng']
        for parser in parsers:
            self._test_fulltext_parser_config(db_client, parser)

        # Test parser with parameters
        self._test_fulltext_parser_config(db_client, 'ngram', params={'ngram_token_size': 3})

        # Test default parser
        self._test_default_parser(db_client)

        # Test backward compatibility
        self._test_backward_compatibility(db_client)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

