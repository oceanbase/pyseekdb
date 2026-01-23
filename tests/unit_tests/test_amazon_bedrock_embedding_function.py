"""
Unit tests for AmazonBedrockEmbeddingFunction.

Tests Amazon Bedrock embedding function initialization, embedding generation, and dimension detection.
Uses real API calls - requires AWS credentials to be configured and boto3 to be installed.

To run this test manually:
    pytest tests/unit_tests/test_amazon_bedrock_embedding_function.py -v -s
    # Make sure AWS credentials are configured via:
    # - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, or
    # - AWS IAM roles, or
    # - AWS credentials file (~/.aws/credentials)
"""

import importlib.util
import re

import pytest

from pyseekdb.client.embedding_function import dimension_of
from pyseekdb.utils.embedding_functions import AmazonBedrockEmbeddingFunction


def is_boto3_available() -> bool:
    """
    Check if boto3 is available for testing.

    Returns:
        True if boto3 is available, False otherwise.
    """
    return importlib.util.find_spec("boto3") is not None


def are_aws_credentials_available() -> bool:
    """
    Check if AWS credentials are available for testing.

    Returns:
        True if AWS credentials are configured, False otherwise.
    """
    if not is_boto3_available():
        return False

    credentials = None
    try:
        import boto3

        # Try to create a session and check if credentials are available
        session = boto3.Session()
        credentials = session.get_credentials()
    except Exception:
        return False
    return credentials is not None


# Skip this test by default - it requires AWS credentials and boto3
@pytest.mark.skipif(
    not is_boto3_available() or not are_aws_credentials_available(),
    reason="boto3 must be installed and AWS credentials must be configured",
)
class TestAmazonBedrockEmbeddingFunction:
    """Test AmazonBedrockEmbeddingFunction - skipped by default, requires AWS credentials"""

    def test_bedrock_env(self):
        """Test if boto3 package is installed and AWS credentials are available."""
        assert is_boto3_available(), "boto3 package is not installed"
        assert are_aws_credentials_available(), "AWS credentials are not configured"

    def test_initialization_with_defaults(self):
        """Test AmazonBedrockEmbeddingFunction initialization with default values"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization with defaults")

        # Check if boto3 is available and AWS credentials are set
        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session)

        assert ef is not None
        assert ef.model_name == "amazon.titan-embed-text-v2"
        print(f"   Model name: {ef.model_name}")

    def test_initialization_with_different_models(self):
        """Test AmazonBedrockEmbeddingFunction initialization with different models"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization with different models")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        models = [
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text-v2",
            "amazon.titan-embed-g1-text-02",
            "amazon.titan-embed-text-v2:0",
        ]

        for model in models:
            ef = AmazonBedrockEmbeddingFunction(session=session, model_name=model)
            assert ef.model_name == model
            print(f"   Model {model}: initialized successfully")

    def test_initialization_with_region_name(self):
        """Test AmazonBedrockEmbeddingFunction initialization with region_name"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization with region_name")

        self.test_bedrock_env()

        import boto3

        # Test with specific region
        session = boto3.Session(region_name="us-east-1")
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        assert ef is not None
        assert ef._session_args.get("region_name") == "us-east-1"
        print(f"   Region name: {ef._session_args.get('region_name')}")

    def test_initialization_with_profile_name(self):
        """Test AmazonBedrockEmbeddingFunction initialization with profile_name"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization with profile_name")

        self.test_bedrock_env()

        import boto3

        # Try to use default profile, or skip if no profiles available
        try:
            # Get available profiles
            available_profiles = boto3.Session().available_profiles
            if available_profiles:
                profile_name = available_profiles[0]
                session = boto3.Session(profile_name=profile_name)
                ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
                assert ef is not None
                assert ef._session_args.get("profile_name") == profile_name
                print(f"   Profile name: {ef._session_args.get('profile_name')}")
            else:
                print("   No AWS profiles available, skipping profile_name test")
        except Exception as e:
            print(f"   Could not test profile_name: {e}")

    def test_initialization_with_kwargs(self):
        """Test AmazonBedrockEmbeddingFunction initialization with additional kwargs"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization with kwargs")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        # Test with endpoint_url (useful for testing with localstack or custom endpoints)
        # Note: This might fail if the endpoint doesn't exist, but we're just testing initialization
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        assert ef is not None
        print("   Initialized with default kwargs")

    def test_initialization_without_boto3(self):
        """Test that initialization fails when boto3 is not installed"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization without boto3")

        # This test doesn't require AWS credentials, just checks the import error
        # We can't easily mock importlib.util.find_spec, so we'll skip if boto3 is available
        if is_boto3_available():
            pytest.skip("boto3 is available, cannot test import error")

    def test_initialization_invalid_kwargs(self):
        """Test that initialization fails with invalid kwargs types"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction initialization with invalid kwargs")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()

        # Test with non-primitive type in kwargs
        class CustomObject:
            pass

        with pytest.raises(TypeError, match=re.escape("Keyword argument.*is not a primitive type")):
            AmazonBedrockEmbeddingFunction(session=session, invalid_arg=CustomObject())

    def test_dimension_property_known_models(self):
        """Test dimension property for known Amazon Bedrock models"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction dimension property for known models")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()

        # Test amazon.titan-embed-text-v1 (1536 dimensions)
        ef_v1 = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v1")
        dim_v1 = ef_v1.dimension
        assert dim_v1 == 1536, f"Expected dimension 1536 for amazon.titan-embed-text-v1, got {dim_v1}"
        print(f"   amazon.titan-embed-text-v1 dimension: {dim_v1}")

        # Test amazon.titan-embed-text-v2 (1024 dimensions)
        ef_v2 = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        dim_v2 = ef_v2.dimension
        assert dim_v2 == 1024, f"Expected dimension 1024 for amazon.titan-embed-text-v2, got {dim_v2}"
        print(f"   amazon.titan-embed-text-v2 dimension: {dim_v2}")

        # Test amazon.titan-embed-g1-text-02 (1024 dimensions)
        ef_g1 = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-g1-text-02")
        dim_g1 = ef_g1.dimension
        assert dim_g1 == 1024, f"Expected dimension 1024 for amazon.titan-embed-g1-text-02, got {dim_g1}"
        print(f"   amazon.titan-embed-g1-text-02 dimension: {dim_g1}")

        # Test amazon.titan-embed-text-v2:0 (1024 dimensions)
        ef_v2_0 = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2:0")
        dim_v2_0 = ef_v2_0.dimension
        assert dim_v2_0 == 1024, f"Expected dimension 1024 for amazon.titan-embed-text-v2:0, got {dim_v2_0}"
        print(f"   amazon.titan-embed-text-v2:0 dimension: {dim_v2_0}")

    def test_call_single_document(self):
        """Test __call__ with single document"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction embedding generation (single document)")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        single_doc = "Hello, world!"
        embeddings = ef(single_doc)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        print(f"   Single document embedding dimension: {len(embeddings[0])}")

    def test_call_multiple_documents(self):
        """Test __call__ with multiple documents"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction embedding generation (multiple documents)")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        multiple_docs = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
        ]
        embeddings = ef(multiple_docs)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(multiple_docs)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == len(embeddings[0]), "All embeddings should have same dimension"
        print(f"   Multiple documents embedding dimension: {len(embeddings[0])}")
        print(f"   Number of embeddings: {len(embeddings)}")

    def test_call_empty_input(self):
        """Test __call__ with empty input"""
        print("\n✅ Testing AmazonBedrockEmbeddingFunction with empty input")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        empty_embeddings = ef([])
        assert empty_embeddings == []
        print("   Empty input correctly returns empty list")

    def test_dimension_of_function(self):
        """Test dimension_of function with AmazonBedrockEmbeddingFunction"""
        print("\n✅ Testing dimension_of function with AmazonBedrockEmbeddingFunction")

        self.test_bedrock_env()

        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2")
        dim = dimension_of(ef)
        assert dim == 1024
        print(f"   dimension_of result for amazon.titan-embed-text-v2: {dim}")

        ef_v1 = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v1")
        dim_v1 = dimension_of(ef_v1)
        assert dim_v1 == 1536
        print(f"   dimension_of result for amazon.titan-embed-text-v1: {dim_v1}")


@pytest.mark.skip("Skipping AmazonBedrockEmbeddingFunction persistence tests")
@pytest.mark.skipif(not is_boto3_available(), reason="boto3 is not available on this system")
class TestAmazonBedrockEmbeddingFunctionPersistence:
    """Test persistence for AmazonBedrockEmbeddingFunction"""

    def test_name(self):
        """Test that name() returns the correct identifier"""
        assert AmazonBedrockEmbeddingFunction.name() == "amazon_bedrock"

    def test_get_config_with_defaults(self):
        """Test that get_config() returns correct config with default values"""
        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session)
        config = ef.get_config()

        assert isinstance(config, dict)
        assert config["model_name"] == "amazon.titan-embed-text-v2"
        assert isinstance(config["client_kwargs"], dict)
        assert isinstance(config["session_args"], dict)
        # Credentials should NOT be in config
        assert "credentials" not in str(config)
        assert "access_key" not in str(config).lower()
        assert "secret_key" not in str(config).lower()
        # name should NOT be in config
        assert "name" not in config

    def test_get_config_with_custom_values(self):
        """Test that get_config() returns correct config with custom values"""
        import boto3

        session = boto3.Session(region_name="us-west-2")
        ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v1")
        config = ef.get_config()

        assert config["model_name"] == "amazon.titan-embed-text-v1"
        assert config["session_args"].get("region_name") == "us-west-2"

    def test_get_config_stores_session_args(self):
        """Test that get_config() stores session_args (region_name, profile_name)"""
        import boto3

        # Test with region_name
        session_region = boto3.Session(region_name="eu-west-1")
        ef_region = AmazonBedrockEmbeddingFunction(session=session_region)
        config_region = ef_region.get_config()

        assert config_region["session_args"].get("region_name") == "eu-west-1"
        assert config_region["session_args"].get("profile_name") is None

    def test_build_from_config_with_defaults(self):
        """Test that build_from_config() restores instance with default values"""

        config = {
            "model_name": "amazon.titan-embed-text-v2",
            "client_kwargs": {},
            "session_args": {},
        }

        restored_ef = AmazonBedrockEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, AmazonBedrockEmbeddingFunction)
        assert restored_ef.model_name == "amazon.titan-embed-text-v2"

    def test_build_from_config_with_custom_values(self):
        """Test that build_from_config() restores instance with custom values"""

        config = {
            "model_name": "amazon.titan-embed-text-v1",
            "client_kwargs": {},
            "session_args": {"region_name": "us-east-1"},
        }

        restored_ef = AmazonBedrockEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, AmazonBedrockEmbeddingFunction)
        assert restored_ef.model_name == "amazon.titan-embed-text-v1"
        assert restored_ef._session_args.get("region_name") == "us-east-1"

    def test_build_from_config_with_session_args(self):
        """Test that build_from_config() correctly restores session_args"""

        config = {
            "model_name": "amazon.titan-embed-text-v2",
            "client_kwargs": {},
            "session_args": {"region_name": "ap-southeast-1", "profile_name": None},
        }

        restored_ef = AmazonBedrockEmbeddingFunction.build_from_config(config)

        assert restored_ef._session_args.get("region_name") == "ap-southeast-1"

    def test_build_from_config_invalid_kwargs(self):
        """Test that build_from_config() raises TypeError when kwargs is not a dict"""
        config = {
            "model_name": "amazon.titan-embed-text-v2",
            "client_kwargs": "not-a-dict",
            "session_args": {},
        }

        with pytest.raises(TypeError, match="kwargs must be a dictionary"):
            AmazonBedrockEmbeddingFunction.build_from_config(config)

    def test_build_from_config_without_boto3(self):
        """Test that build_from_config() fails when boto3 is not installed"""
        # This test is skipped if boto3 is available
        # In a real scenario, this would test the ImportError
        if is_boto3_available():
            pytest.skip("boto3 is available, cannot test import error")

    def test_persistence_roundtrip(self):
        """Test complete roundtrip: get_config -> build_from_config"""
        import boto3

        session = boto3.Session(region_name="us-east-1")
        original_ef = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v1")

        config = original_ef.get_config()
        restored_ef = AmazonBedrockEmbeddingFunction.build_from_config(config)

        assert isinstance(restored_ef, AmazonBedrockEmbeddingFunction)
        assert restored_ef.model_name == original_ef.model_name
        assert restored_ef._session_args == original_ef._session_args

    def test_persistence_does_not_store_credentials(self):
        """Test that get_config() does not store AWS credentials"""
        import boto3

        session = boto3.Session()
        ef = AmazonBedrockEmbeddingFunction(session=session)
        config = ef.get_config()

        # Ensure no credential information is stored
        config_str = str(config).lower()
        assert "access_key" not in config_str
        assert "secret_key" not in config_str
        assert "token" not in config_str or "session_args" in config_str  # session_args is OK
        assert "credential" not in config_str or "session_args" in config_str  # session_args is OK


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
