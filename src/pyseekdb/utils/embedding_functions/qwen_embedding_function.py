from pyseekdb.utils.embedding_functions.litellm_embedding_function import LiteLLMEmbeddingFunction
from typing import Any, Optional


class QwenEmbeddingFunction(LiteLLMEmbeddingFunction):
    """
    A convenient embedding function for Qwen (Alibaba Cloud) embedding models.

    This class provides a simplified interface to Qwen embedding models using LiteLLM.
    Qwen provides OpenAI-compatible API endpoints for embedding generation.

    Example:
        pip install pyseekdb litellm

    .. code-block:: python
        import pyseekdb
        from pyseekdb.utils.embedding_functions import QwenEmbeddingFunction

        # Using Qwen embedding model
        # Set QWEN_API_KEY environment variable first
        ef = QwenEmbeddingFunction(
            model_name="text-embedding-v1"
        )

        # Using with custom api_key_env and additional parameters
        ef = QwenEmbeddingFunction(
            model_name="text-embedding-v1",
            api_key_env="QWEN_API_KEY",
            timeout=30
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
        model_name: str,
        api_key_env: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize QwenEmbeddingFunction.

        Args:
            model_name (str): Name of the Qwen embedding model.
                Examples:
                - "text-embedding-v1" (common Qwen embedding model)
                - "text-embedding-v2" (if available)
                - See Qwen documentation for available models
            api_key_env (str, optional): Name of the environment variable containing the Qwen API key.
                Defaults to "QWEN_API_KEY" if not provided.
            **kwargs: Additional arguments to pass to the LiteLLM embedding function.
                Common options include:
                - api_base: Base URL for the Qwen API endpoint.
                    Defaults to "https://dashscope.aliyuncs.com/compatible-mode/v1" if not provided.
                - encoding_format: Encoding format for embeddings. Defaults to "float".
                    Qwen API supports "float" or "base64".
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
                - See https://docs.litellm.ai/docs/embedding for more options
        """
        # Set default api_key_env if not provided
        if api_key_env is None:
            api_key_env = "QWEN_API_KEY"

        # Set default api_base if not provided
        api_base_provided = "api_base" in kwargs
        if not api_base_provided:
            kwargs["api_base"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # For OpenAI-compatible APIs, LiteLLM needs to know the provider
        # Use custom_llm_provider parameter instead of modifying model name
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"

        # Qwen API requires encoding_format to be explicitly set to "float" or "base64"
        # We use "float" since we need float embeddings
        if "encoding_format" not in kwargs:
            kwargs["encoding_format"] = "float"

        # Initialize parent class with Qwen-specific defaults
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            **kwargs
        )
