"""
Unit tests for configuration classes
"""

import sys
from pathlib import Path

import pytest

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyseekdb import Configuration, FulltextAnalyzerConfig, HNSWConfiguration  # noqa: E402


class TestHNSWConfiguration:
    """Test HNSWConfiguration class"""

    def test_valid_configuration(self):
        """Test creating valid HNSWConfiguration"""
        config = HNSWConfiguration(dimension=128, distance="cosine")
        assert config.dimension == 128
        assert config.distance == "cosine"

    def test_default_distance(self):
        """Test default distance metric"""
        config = HNSWConfiguration(dimension=128)
        assert config.distance == "l2"

    def test_invalid_dimension(self):
        """Test that invalid dimension raises ValueError"""
        with pytest.raises(ValueError, match="dimension must be positive"):
            HNSWConfiguration(dimension=0)

        with pytest.raises(ValueError, match="dimension must be positive"):
            HNSWConfiguration(dimension=-1)

    def test_invalid_distance(self):
        """Test that invalid distance raises ValueError"""
        with pytest.raises(ValueError, match="distance must be one of"):
            HNSWConfiguration(dimension=128, distance="invalid")


class TestFulltextAnalyzerConfig:
    """Test FulltextAnalyzerConfig class"""

    def test_valid_parsers(self):
        """Test creating FulltextAnalyzerConfig with valid parsers"""
        valid_parsers = ["ik", "space", "ngram", "ngram2", "beng"]
        for parser in valid_parsers:
            config = FulltextAnalyzerConfig(analyzer=parser)
            assert config.analyzer == parser
            assert config.properties is None

    def test_default_parser(self):
        """Test default parser is 'ik'"""
        config = FulltextAnalyzerConfig()
        assert config.analyzer == "ik"

    def test_parser_with_params(self):
        """Test parser with parameters"""
        config = FulltextAnalyzerConfig(analyzer="ngram", properties={"size": 2})
        assert config.analyzer == "ngram"
        assert config.properties == {"size": 2}

    def test_parser_with_multiple_params(self):
        """Test parser with multiple parameters"""
        config = FulltextAnalyzerConfig(analyzer="ngram", properties={"size": 3, "min_size": 1, "max_size": 5})
        assert config.analyzer == "ngram"
        assert config.properties["size"] == 3
        assert config.properties["min_size"] == 1
        assert config.properties["max_size"] == 5

    def test_params_with_different_types(self):
        """Test params with different primitive types"""
        config = FulltextAnalyzerConfig(
            analyzer="ik",
            properties={
                "string_param": "value",
                "int_param": 42,
                "float_param": 3.14,
                "bool_param": True,
            },
        )
        assert config.properties["string_param"] == "value"
        assert config.properties["int_param"] == 42
        assert config.properties["float_param"] == 3.14
        assert config.properties["bool_param"] is True


class TestConfiguration:
    """Test Configuration class"""

    def test_configuration_with_hnsw_only(self):
        """Test Configuration with only HNSW config"""
        hnsw_config = HNSWConfiguration(dimension=128, distance="cosine")
        config = Configuration(hnsw=hnsw_config)
        assert config.hnsw == hnsw_config
        assert config.fulltext_config is None

    def test_configuration_with_fulltext_only(self):
        """Test Configuration with only fulltext config"""
        fulltext_config = FulltextAnalyzerConfig(analyzer="ik")
        config = Configuration(fulltext_config=fulltext_config)
        assert config.hnsw is None
        assert config.fulltext_config == fulltext_config

    def test_configuration_with_both(self):
        """Test Configuration with both HNSW and fulltext config"""
        hnsw_config = HNSWConfiguration(dimension=128, distance="cosine")
        fulltext_config = FulltextAnalyzerConfig(analyzer="space")
        config = Configuration(hnsw=hnsw_config, fulltext_config=fulltext_config)
        assert config.hnsw == hnsw_config
        assert config.fulltext_config == fulltext_config

    def test_configuration_empty(self):
        """Test Configuration with no parameters"""
        config = Configuration()
        assert config.hnsw is None
        assert config.fulltext_config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
