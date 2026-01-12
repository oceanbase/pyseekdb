from pyseekdb.utils.embedding_functions.litellm_embedding_function import LiteLLMEmbeddingFunction
from typing import Any, Optional


class OpenAIEmbeddingFunction(LiteLLMEmbeddingFunction):
    """
    A convenient embedding function for OpenAI embedding models.

    This class provides a simplified interface to OpenAI embedding models using LiteLLM.
    It sets default values for OpenAI-specific configurations.

    Example:
        pip install pyseekdb litellm

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
        **kwargs: Any,
    ):
        """Initialize OpenAIEmbeddingFunction.

        Args:
            model_name (str, optional): Name of the OpenAI embedding model.
                Defaults to "text-embedding-ada-002".
                Other options include:
                - "text-embedding-ada-002" (1536 dimensions, default)
                - "text-embedding-3-small" (1536 dimensions)
                - "text-embedding-3-large" (3072 dimensions)
            api_key_env (str, optional): Name of the environment variable containing the OpenAI API key.
                Defaults to "OPENAI_API_KEY" if not provided.
            **kwargs: Additional arguments to pass to the LiteLLM embedding function.
                Common options include:
                - api_base: Base URL for the API endpoint (useful for OpenAI-compatible proxies)
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
                - api_version: API version
                - user: User identifier for usage tracking
                - See https://docs.litellm.ai/docs/embedding for more options
        """
        # Set default api_key_env if not provided
        if api_key_env is None:
            api_key_env = "OPENAI_API_KEY"

        # Initialize parent class with OpenAI-specific defaults
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            **kwargs
        )
