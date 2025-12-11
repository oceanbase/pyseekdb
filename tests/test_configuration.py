"""
Unit tests for configuration classes
"""
import pytest
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyseekdb import (
    Configuration,
    HNSWConfiguration,
    FulltextParserConfig
)


class TestHNSWConfiguration:
    """Test HNSWConfiguration class"""

    def test_valid_configuration(self):
        """Test creating valid HNSWConfiguration"""
        config = HNSWConfiguration(dimension=128, distance='cosine')
        assert config.dimension == 128
        assert config.distance == 'cosine'

    def test_default_distance(self):
        """Test default distance metric"""
        config = HNSWConfiguration(dimension=128)
        assert config.distance == 'l2'

    def test_invalid_dimension(self):
        """Test that invalid dimension raises ValueError"""
        with pytest.raises(ValueError, match="dimension must be positive"):
            HNSWConfiguration(dimension=0)

        with pytest.raises(ValueError, match="dimension must be positive"):
            HNSWConfiguration(dimension=-1)

    def test_invalid_distance(self):
        """Test that invalid distance raises ValueError"""
        with pytest.raises(ValueError, match="distance must be one of"):
            HNSWConfiguration(dimension=128, distance='invalid')


class TestFulltextParserConfig:
    """Test FulltextParserConfig class"""

    def test_valid_parsers(self):
        """Test creating FulltextParserConfig with valid parsers"""
        valid_parsers = ['ik', 'space', 'ngram', 'ngram2', 'beng']
        for parser in valid_parsers:
            config = FulltextParserConfig(parser=parser)
            assert config.parser == parser
            assert config.params is None

    def test_default_parser(self):
        """Test default parser is 'ik'"""
        config = FulltextParserConfig()
        assert config.parser == 'ik'

    def test_parser_with_params(self):
        """Test parser with parameters"""
        config = FulltextParserConfig(parser='ngram', params={'size': 2})
        assert config.parser == 'ngram'
        assert config.params == {'size': 2}

    def test_parser_with_multiple_params(self):
        """Test parser with multiple parameters"""
        config = FulltextParserConfig(
            parser='ngram',
            params={'size': 3, 'min_size': 1, 'max_size': 5}
        )
        assert config.parser == 'ngram'
        assert config.params['size'] == 3
        assert config.params['min_size'] == 1
        assert config.params['max_size'] == 5

    def test_params_with_different_types(self):
        """Test params with different primitive types"""
        config = FulltextParserConfig(
            parser='ik',
            params={
                'string_param': 'value',
                'int_param': 42,
                'float_param': 3.14,
                'bool_param': True
            }
        )
        assert config.params['string_param'] == 'value'
        assert config.params['int_param'] == 42
        assert config.params['float_param'] == 3.14
        assert config.params['bool_param'] is True


class TestConfiguration:
    """Test Configuration class"""

    def test_configuration_with_hnsw_only(self):
        """Test Configuration with only HNSW config"""
        hnsw_config = HNSWConfiguration(dimension=128, distance='cosine')
        config = Configuration(hnsw=hnsw_config)
        assert config.hnsw == hnsw_config
        assert config.fulltext_config is None

    def test_configuration_with_fulltext_only(self):
        """Test Configuration with only fulltext config"""
        fulltext_config = FulltextParserConfig(parser='ik')
        config = Configuration(fulltext_config=fulltext_config)
        assert config.hnsw is None
        assert config.fulltext_config == fulltext_config

    def test_configuration_with_both(self):
        """Test Configuration with both HNSW and fulltext config"""
        hnsw_config = HNSWConfiguration(dimension=128, distance='cosine')
        fulltext_config = FulltextParserConfig(parser='space')
        config = Configuration(
            hnsw=hnsw_config,
            fulltext_config=fulltext_config
        )
        assert config.hnsw == hnsw_config
        assert config.fulltext_config == fulltext_config

    def test_configuration_empty(self):
        """Test Configuration with no parameters"""
        config = Configuration()
        assert config.hnsw is None
        assert config.fulltext_config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
