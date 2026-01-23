import os
from typing import Any

from pyseekdb.client.embedding_function import Documents, EmbeddingFunction, Embeddings


class LiteLLMBaseEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using LiteLLM to access various embedding models.

    LiteLLM provides a unified interface to access embedding models from multiple providers
    including OpenAI, Hugging Face, Cohere, and many others.

    Example:
        pip install pyseekdb litellm

    .. code-block:: python
        import pyseekdb
        from pyseekdb.utils.embedding_functions import LiteLLMEmbeddingFunction

        # Using OpenAI embeddings (API key from environment variable)
        # Set OPENAI_API_KEY environment variable first
        ef = LiteLLMEmbeddingFunction(
            model_name="text-embedding-ada-002",
            api_key_env="OPENAI_API_KEY"
        )

        # Using Hugging Face embeddings (API key from environment variable)
        # Set HUGGINGFACE_API_KEY environment variable first
        ef = LiteLLMEmbeddingFunction(
            model_name="huggingface/sentence-transformers/all-MiniLM-L6-v2",
            api_key_env="HUGGINGFACE_API_KEY"
        )

        # Using with additional parameters via kwargs
        ef = LiteLLMEmbeddingFunction(
            model_name="text-embedding-ada-002",
            api_key_env="OPENAI_API_KEY",
            api_base="https://api.openai.com/v1",
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
        model_name: str,
        api_key_env: str | None = None,
        **kwargs: Any,
    ):
        """Initialize.

        Args:
            model_name (str): Identifier of the embedding model.
                Examples:
                - "text-embedding-ada-002" (OpenAI)
                - "text-embedding-3-small" (OpenAI)
                - "huggingface/sentence-transformers/all-MiniLM-L6-v2" (Hugging Face)
                - "cohere/embed-english-v3.0" (Cohere)
                - See https://docs.litellm.ai/docs/embedding/supported_embedding for full list
            api_key_env (str, optional): Name of the environment variable containing the API key.
                If not provided, LiteLLM will try to use default environment variables based on the provider.
                For example, "OPENAI_API_KEY" for OpenAI, "HUGGINGFACE_API_KEY" for Hugging Face.
                See https://docs.litellm.ai/docs/set_keys for a complete list of default environment variable names.
            **kwargs: Additional arguments to pass to the LiteLLM embedding function.
                Common options include:
                - api_base: Base URL for the API endpoint (useful for LiteLLM proxy or custom endpoints)
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
                - api_version: API version (provider-specific)
                - user: User identifier for usage tracking
                - See https://docs.litellm.ai/docs/embedding/supported_embedding for more options
        """
        try:
            from litellm import embedding
        except ImportError as exc:
            raise ValueError(
                "The litellm python package is not installed. Please install it with `pip install litellm`"
            ) from exc

        self.model_name = model_name
        self.api_key_env = api_key_env
        for key, value in kwargs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                raise TypeError(f"Keyword argument {key} is not a primitive type")
        self.kwargs = kwargs
        self._embedding_func = embedding

    def __call__(self, documents: Documents) -> Embeddings:  # noqa: C901
        """Generate embeddings for the given documents.

        Args:
            documents: Documents to generate embeddings for. Can be a single string or list of strings.

        Returns:
            Embeddings for the documents as a list of lists of floats.
        """
        # Handle single string input
        if isinstance(documents, str):
            documents = [documents]

        # Handle empty input
        if not documents:
            return []

        # Prepare arguments for LiteLLM embedding function
        embedding_kwargs = {"model": self.model_name, "input": documents, **self.kwargs}

        # Read API key from environment variable if specified
        if self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if api_key is None:
                raise ValueError(
                    f"API key environment variable '{self.api_key_env}' is not set. "
                    f"Please set it before using LiteLLMEmbeddingFunction."
                )
            embedding_kwargs["api_key"] = api_key

        # Call LiteLLM embedding function
        response = self._embedding_func(**embedding_kwargs)

        # Extract embeddings from response
        # LiteLLM returns an EmbeddingResponse object with a 'data' attribute
        # Each item in data has an 'embedding' field with the actual vector
        embeddings = []

        # Handle EmbeddingResponse object (most common case)
        if hasattr(response, "data"):
            for item in response.data:
                if hasattr(item, "embedding"):
                    embeddings.append(item.embedding)
                elif isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                elif isinstance(item, list):
                    # Some providers might return the embedding vector directly
                    embeddings.append(item)
                else:
                    raise ValueError(f"Unexpected item format in LiteLLM response: {type(item)}")
        # Handle dict response (backward compatibility)
        elif isinstance(response, dict) and "data" in response:
            for item in response["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                elif isinstance(item, list):
                    # Some providers might return the embedding vector directly
                    embeddings.append(item)
                else:
                    raise ValueError(f"Unexpected item format in LiteLLM response: {type(item)}")
        # Handle list response (backward compatibility)
        elif isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                elif isinstance(item, list):
                    embeddings.append(item)
                else:
                    raise ValueError(f"Unexpected item format in LiteLLM response: {type(item)}")
        else:
            raise ValueError(f"Unexpected response format from LiteLLM: {type(response)}")

        # Validate that we got the expected number of embeddings
        if len(embeddings) != len(documents):
            raise ValueError(f"Expected {len(documents)} embeddings but got {len(embeddings)} from LiteLLM")

        return embeddings
