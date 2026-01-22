from typing import Any

from pyseekdb.utils.embedding_functions.openai_base_embedding_function import (
    OpenAIBaseEmbeddingFunction,
)

# Known SiliconFlow embedding model dimensions
# Source: https://docs.siliconflow.cn/en/api-reference/embeddings/create-embeddings
_SILICONFLOW_MODEL_DIMENSIONS = {
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "netease-youdao/bce-embedding-base_v1": 768,
    "BAAI/bge-m3": 1024,
    "Pro/BAAI/bge-m3": 1024,
    # Qwen models support variable dimensions, default values listed below
    "Qwen/Qwen3-Embedding-8B": 4096,  # default, supports [64,128,256,512,768,1024,1536,2048,2560,4096]
    "Qwen/Qwen3-Embedding-4B": 2560,  # default, supports [64,128,256,512,768,1024,1536,2048,2560]
    "Qwen/Qwen3-Embedding-0.6B": 1024,  # default, supports [64,128,256,512,768,1024]
}


class SiliconflowEmbeddingFunction(OpenAIBaseEmbeddingFunction):
    """
    A convenient embedding function for SiliconFlow embedding models.

    This class provides a simplified interface to SiliconFlow embedding models using the OpenAI-compatible API.
    SiliconFlow provides OpenAI-compatible API endpoints for embedding generation.

    For more information about SiliconFlow models, see https://docs.siliconflow.cn/en/api-reference/embeddings/create-embeddings

    Example:
        pip install pyseekdb openai

    .. code-block:: python
        import pyseekdb
        from pyseekdb.utils.embedding_functions import SiliconflowEmbeddingFunction

        # Using SiliconFlow embedding model
        # Set SILICONFLOW_API_KEY environment variable first
        ef = SiliconflowEmbeddingFunction(
            model_name="BAAI/bge-large-zh-v1.5"
        )

        # Using with custom api_key_env and additional parameters
        ef = SiliconflowEmbeddingFunction(
            model_name="BAAI/bge-m3",
            api_key_env="SILICONFLOW_API_KEY",
            timeout=30
        )

        # Using Qwen models with custom dimensions
        ef = SiliconflowEmbeddingFunction(
            model_name="Qwen/Qwen3-Embedding-8B",
            dimensions=1024  # Reduce from default 4096 to 1024
        )

        db = pyseekdb.Client(path="./seekdb.db")
        collection = db.create_collection(name="my_collection", embedding_function=ef)
        # Add documents
        collection.add(ids=["1", "2"], documents=["Hello world", "How are you?"], metadatas=[{"id": 1}, {"id": 2}])
        # Query using semantic search
        results = collection.query("How are you?", n_results=1)
        print(results)

    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        api_key_env: str | None = None,
        api_base: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ):
        """Initialize SiliconflowEmbeddingFunction.

        Args:
            model_name (str, optional): Name of the SiliconFlow embedding model.
                Defaults to "BAAI/bge-large-zh-v1.5".
                Common options include:
                - "BAAI/bge-large-zh-v1.5" (1024 dimensions, 512 token limit)
                - "BAAI/bge-large-en-v1.5" (1024 dimensions, 512 token limit)
                - "BAAI/bge-m3" (1024 dimensions, 8192 token limit)
                - "Pro/BAAI/bge-m3" (1024 dimensions, 8192 token limit)
                - "Qwen/Qwen3-Embedding-8B" (4096 dimensions by default, 32768 token limit, supports variable dimensions)
                - "Qwen/Qwen3-Embedding-4B" (2560 dimensions by default, 32768 token limit, supports variable dimensions)
                - "Qwen/Qwen3-Embedding-0.6B" (1024 dimensions by default, 32768 token limit, supports variable dimensions)
                - "netease-youdao/bce-embedding-base_v1" (768 dimensions, 512 token limit)
                - See SiliconFlow documentation for available models
            api_key_env (str, optional): Name of the environment variable containing the SiliconFlow API key.
                Defaults to "SILICONFLOW_API_KEY" if not provided.
            api_base (str, optional): Base URL for the SiliconFlow API endpoint.
                Defaults to "https://api.siliconflow.cn/v1" if not provided.
            dimensions (int, optional): The number of dimensions the resulting embeddings should have.
                Only supported for Qwen/Qwen3 series models. Can reduce dimensions from default.
                For Qwen/Qwen3-Embedding-8B: [64,128,256,512,768,1024,1536,2048,2560,4096]
                For Qwen/Qwen3-Embedding-4B: [64,128,256,512,768,1024,1536,2048,2560]
                For Qwen/Qwen3-Embedding-0.6B: [64,128,256,512,768,1024]
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
            **kwargs,
        )

    def _get_default_api_base(self) -> str:
        """Get the default API base URL for SiliconFlow.

        Returns:
            str: Default SiliconFlow API base URL
        """
        return "https://api.siliconflow.cn/v1"

    def _get_default_api_key_env(self) -> str:
        """Get the default API key environment variable name for SiliconFlow.

        Returns:
            str: Default SiliconFlow API key environment variable name
        """
        return "SILICONFLOW_API_KEY"

    def _get_model_dimensions(self) -> dict[str, int]:
        """Get a dictionary mapping SiliconFlow model names to their default dimensions.

        Returns:
            dict[str, int]: Dictionary mapping model names to dimensions
        """
        return _SILICONFLOW_MODEL_DIMENSIONS

    @staticmethod
    def name() -> str:
        """Get the unique name identifier for SiliconflowEmbeddingFunction.

        Returns:
            The name identifier for this embedding function type
        """
        return "siliconflow"

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for the SiliconflowEmbeddingFunction.

        Returns:
            Dictionary containing configuration needed to restore this embedding function
        """
        return super().get_config()

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "SiliconflowEmbeddingFunction":
        """Build a SiliconflowEmbeddingFunction from its configuration dictionary.

        Args:
            config: Dictionary containing the embedding function's configuration

        Returns:
            Restored SiliconflowEmbeddingFunction instance

        Raises:
            ValueError: If the configuration is invalid or missing required fields
        """
        model_name = config.get("model_name")
        if model_name is None:
            raise ValueError("Missing required field 'model_name' in configuration")

        api_key_env = config.get("api_key_env")
        api_base = config.get("api_base")
        dimensions = config.get("dimensions")
        client_kwargs = config.get("client_kwargs", {})
        if not isinstance(client_kwargs, dict):
            raise TypeError(f"client_kwargs must be a dictionary, but got {client_kwargs}")

        return SiliconflowEmbeddingFunction(
            model_name=model_name,
            api_key_env=api_key_env,
            api_base=api_base,
            dimensions=dimensions,
            **client_kwargs,
        )
