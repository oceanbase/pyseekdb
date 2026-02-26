"""
Unit tests for configuration classes
"""

import sys
from pathlib import Path

import pytest

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyseekdb import (  # noqa: E402
    BqHNSWConfiguration,
    Configuration,
    FulltextIndexConfig,
    HNSWConfiguration,
    IKFulltextIndexConfig,
    Ngram2FulltextIndexConfig,
    NgramFulltextIndexConfig,
    SpaceFulltextIndexConfig,
)
from pyseekdb.client.client_base import _get_vector_index_sql  # noqa: E402


class TestHNSWConfiguration:
    """Test HNSWConfiguration class"""

    def test_valid_configuration(self):
        """Test creating valid HNSWConfiguration"""
        config = HNSWConfiguration(dimension=128, distance="cosine", type="hnsw", lib="vsag")
        assert config.dimension == 128
        assert config.distance == "cosine"
        assert config.type == "hnsw"
        assert config.lib == "vsag"

    def test_default_distance(self):
        """Test default distance metric"""
        config = HNSWConfiguration()
        assert config.distance == "l2"
        assert config.dimension == 384

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

    def test_invalid_type(self):
        """Test that invalid type raises ValueError"""
        with pytest.raises(ValueError, match="type must be one of"):
            HNSWConfiguration(type="invalid")

    def test_invalid_lib(self):
        """Test that invalid lib raises ValueError"""
        with pytest.raises(ValueError, match="lib must be one of"):
            HNSWConfiguration(lib="invalid")

    def test_properties_with_primitive_types(self):
        """Test properties with primitive value types"""
        config = HNSWConfiguration(
            dimension=128,
            properties={
                "M": 16,
                "ef_search": 200,
                "ef_construction": 400,
                "extra_info_max_size": 100,
                "normalize": True,
                "quantization": "pq",
                "alpha": 0.75,
            },
        )
        assert config.properties["m"] == 16
        assert config.properties["ef_search"] == 200
        assert config.properties["ef_construction"] == 400
        assert config.properties["extra_info_max_size"] == 100
        assert config.properties["normalize"] is True
        assert config.properties["quantization"] == "pq"
        assert config.properties["alpha"] == 0.75

    def test_properties_invalid_type(self):
        """Test properties with invalid value types"""
        with pytest.raises(TypeError, match="properties must be a dictionary of string, int, float, or bool"):
            HNSWConfiguration(
                dimension=128,
                properties={
                    "m": 16,
                    "invalid": {"nested": "dict"},
                },
            )

    def test_properties_reserved_keywords(self):
        """Test that top-level keys in properties are removed with warning"""
        with pytest.warns(UserWarning, match="reserved keyword"):
            config = HNSWConfiguration(
                dimension=128,
                properties={
                    "distance": "cosine",
                    "type": "hnsw",
                    "lib": "vsag",
                    "M": 32,
                },
            )
        assert "distance" not in {key.lower() for key in config.properties}
        assert "type" not in {key.lower() for key in config.properties}
        assert "lib" not in {key.lower() for key in config.properties}
        assert config.properties["m"] == 32

    def test_hnsw_numeric_ranges(self):
        with pytest.raises(ValueError, match="m must be between 5 and 128"):
            HNSWConfiguration(properties={"m": 3})
        with pytest.raises(ValueError, match="ef_construction must be between 5 and 1000"):
            HNSWConfiguration(properties={"ef_construction": 1001})
        with pytest.raises(ValueError, match="ef_search must be between 1 and 1000"):
            HNSWConfiguration(properties={"ef_search": 0})
        with pytest.raises(ValueError, match="extra_info_max_size must be between 0 and 16384"):
            HNSWConfiguration(properties={"extra_info_max_size": 17000})

    def test_hnsw_bq_properties(self):
        config = BqHNSWConfiguration(refine_k=4.0, refine_type="sq8", bq_bits_query=32, bq_use_fht=True)
        assert config.type == "hnsw_bq"
        assert config.properties["refine_k"] == 4.0
        assert config.properties["refine_type"] == "sq8"
        assert config.properties["bq_bits_query"] == 32
        assert config.properties["bq_use_fht"] is True

    def test_vector_index_sql_with_properties(self):
        """Test SQL generation includes properties"""
        config = HNSWConfiguration(
            dimension=128,
            distance="cosine",
            type="hnsw_sq",
            lib="vsag",
            properties={
                "M": 16,
                "ef_search": 200,
                "quantization": "pq",
            },
        )
        sql = _get_vector_index_sql(config)
        assert "DISTANCE=cosine" in sql
        assert "TYPE=hnsw_sq" in sql
        assert "LIB=vsag" in sql
        assert "m=16" in sql
        assert "ef_search=200" in sql
        assert "quantization='pq'" in sql


class TestFulltextIndexConfig:
    """Test FulltextIndexConfig class"""

    def test_valid_parsers(self):
        """Test creating FulltextIndexConfig with valid parsers"""
        valid_parsers = ["ik", "space", "ngram", "ngram2", "beng"]
        for parser in valid_parsers:
            config = FulltextIndexConfig(analyzer=parser)
            assert config.analyzer == parser
            assert config.properties is None

    def test_default_parser(self):
        """Test default parser is 'ik'"""
        config = FulltextIndexConfig()
        assert config.analyzer == "ik"

    def test_parser_with_params(self):
        """Test parser with parameters"""
        config = FulltextIndexConfig(analyzer="ngram", properties={"size": 2})
        assert config.analyzer == "ngram"
        assert config.properties == {"size": 2}

    def test_parser_with_multiple_params(self):
        """Test parser with multiple parameters"""
        config = FulltextIndexConfig(analyzer="ngram", properties={"size": 3, "min_size": 1, "max_size": 5})
        assert config.analyzer == "ngram"
        assert config.properties["size"] == 3
        assert config.properties["min_size"] == 1
        assert config.properties["max_size"] == 5

    def test_unknown_analyzer_warns(self):
        with pytest.warns(UserWarning, match="Unknown analyzer"):
            config = FulltextIndexConfig(analyzer="jieba", properties={"token_size": 4})
        assert config.analyzer == "jieba"
        assert config.properties["token_size"] == 4

    def test_space_analyzer_param_validation(self):
        with pytest.raises(ValueError, match="max_token_size should not be less than min_token_size"):
            SpaceFulltextIndexConfig(min_token_size=16, max_token_size=10)

    def test_ngram_analyzer_param_validation(self):
        with pytest.raises(ValueError, match="ngram_token_size must be between 1 and 10"):
            NgramFulltextIndexConfig(ngram_token_size=11)

    def test_ngram2_analyzer_param_validation(self):
        with pytest.raises(ValueError, match="max_ngram_size should not be less than min_ngram_size"):
            Ngram2FulltextIndexConfig(min_ngram_size=10, max_ngram_size=2)

    def test_ik_mode_validation(self):
        with pytest.raises(ValueError, match="ik_mode should be one of"):
            IKFulltextIndexConfig(ik_mode="invalid")

    def test_params_with_different_types(self):
        """Test params with different primitive types"""
        config = FulltextIndexConfig(
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
        fulltext_config = FulltextIndexConfig(analyzer="ik")
        config = Configuration(fulltext_config=fulltext_config)
        assert config.hnsw is None
        assert config.fulltext_config == fulltext_config

    def test_configuration_with_both(self):
        """Test Configuration with both HNSW and fulltext config"""
        hnsw_config = HNSWConfiguration(dimension=128, distance="cosine")
        fulltext_config = FulltextIndexConfig(analyzer="space")
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
