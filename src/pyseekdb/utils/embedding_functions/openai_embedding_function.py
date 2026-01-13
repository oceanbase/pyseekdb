from pyseekdb.utils.embedding_functions.openai_base_embedding_function import OpenAIBaseEmbeddingFunction
from typing import Any, Optional

# Known OpenAI embedding model dimensions
# Source: https://platform.openai.com/docs/guides/embeddings
_OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbeddingFunction(OpenAIBaseEmbeddingFunction):
    """
    A convenient embedding function for OpenAI embedding models.

    This class provides a simplified interface to OpenAI embedding models using the OpenAI API.

    For more information about OpenAI models, see https://platform.openai.com/docs/guides/embeddings

    Example:
        pip install pyseekdb openai

    .. code-block:: python
        import pyseekdb
        from pyseekdb.utils.embedding_functions import OpenAIEmbeddingFunction

        # Using default model (text-embedding-ada-002)
        # Set OPENAI_API_KEY environment variable first
        ef = OpenAIEmbeddingFunction()

        # Using a specific OpenAI model
        ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

        # Using with additional parameters
        ef = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-large",
            timeout=30,
            max_retries=3
        )

        # Using text-embedding-3 with custom dimensions
        ef = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            dimensions=512  # Reduce from default 1536 to 512
        )

        db = pyseekdb.Client(path="./seekdb.db")
        collection = db.create_collection(name="my_collection", embedding_function=ef)
        # Add documents
        collection.add(ids=["1", "2"], documents=["Hello world", "How are you?"], metadatas=[{"id": 1}, {"id": 2}])
        # Query using semantic search
        results = collection.query("How are you?", top_k=1)
        print(results)

    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key_env: Optional[str] = None,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize OpenAIEmbeddingFunction.

        Args:
            model_name (str, optional): Name of the OpenAI embedding model.
                Defaults to "text-embedding-ada-002".
                Other options include:
                - "text-embedding-ada-002" (1536 dimensions, default)
                - "text-embedding-3-small" (1536 dimensions by default, can be reduced via dimensions parameter)
                - "text-embedding-3-large" (3072 dimensions by default, can be reduced via dimensions parameter)
            api_key_env (str, optional): Name of the environment variable containing the OpenAI API key.
                Defaults to "OPENAI_API_KEY" if not provided.
            api_base (str, optional): Base URL for the API endpoint.
                Defaults to "https://api.openai.com/v1" if not provided.
                Useful for OpenAI-compatible proxies or custom endpoints.
            dimensions (int, optional): The number of dimensions the resulting embeddings should have.
                Only supported for text-embedding-3 models. Can reduce dimensions from
                default (1536 for text-embedding-3-small, 3072 for text-embedding-3-large).
            **kwargs: Additional arguments to pass to the OpenAI client.
                Common options include:
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
                - See https://github.com/openai/openai-python for more options
        """
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            api_base=api_base,
            dimensions=dimensions,
            **kwargs
        )

    def _get_default_api_base(self) -> str:
        """Get the default API base URL for OpenAI.

        Returns:
            str: Default OpenAI API base URL
        """
        return "https://api.openai.com/v1"

    def _get_default_api_key_env(self) -> str:
        """Get the default API key environment variable name for OpenAI.

        Returns:
            str: Default OpenAI API key environment variable name
        """
        return "OPENAI_API_KEY"

    def _get_model_dimensions(self) -> dict[str, int]:
        """Get a dictionary mapping OpenAI model names to their default dimensions.

        Returns:
            dict[str, int]: Dictionary mapping model names to dimensions
        """
        return _OPENAI_MODEL_DIMENSIONS
