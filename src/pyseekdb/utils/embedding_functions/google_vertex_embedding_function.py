from pyseekdb.utils.embedding_functions.litellm_base_embedding_function import (
    LiteLLMBaseEmbeddingFunction,
)
from typing import Any, Dict, Optional

# Known Google Vertex AI embedding model dimensions
# Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
# Note: Some models support flexible dimensions via outputDimensionality parameter
_GOOGLE_VERTEX_MODEL_DIMENSIONS = {
    "textembedding-gecko@003": 768,  # supports outputDimensionality: 128, 256, 512, 768
    "textembedding-gecko@002": 768,  # supports outputDimensionality: 128, 256, 512, 768
    "text-multilingual-embedding-002": 768,  # supports outputDimensionality: 128, 256, 512, 768
    "gemini-embedding-001": 768,  # supports outputDimensionality: 128, 256, 512, 768
}


class GoogleVertexEmbeddingFunction(LiteLLMBaseEmbeddingFunction):
    """
    A convenient embedding function for Google Vertex AI embedding models.

    This class provides a simplified interface to Google Vertex AI embedding models using LiteLLM.

    For more information about Google Vertex AI models, see
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api

    For LiteLLM documentation, see https://docs.litellm.ai/docs/embedding/supported_embedding

    Authentication:
        This function uses Application Default Credentials (ADC). Set up authentication by:
        1. Setting the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your
           service account key file, or
        2. Running `gcloud auth application-default login` for local development

    Example:
        pip install pyseekdb litellm

    .. code-block:: python
        import pyseekdb
        from pyseekdb.utils.embedding_functions import GoogleVertexEmbeddingFunction

        # Using Google Vertex AI embedding model
        # Set up authentication first (see Authentication section above)
        ef = GoogleVertexEmbeddingFunction(
            project_id="your-project-id",
            model_name="textembedding-gecko@003"
        )

        # Using with task_type for better retrieval performance
        ef = GoogleVertexEmbeddingFunction(
            project_id="your-project-id",
            model_name="textembedding-gecko@003",
            task_type="RETRIEVAL_DOCUMENT"  # or "RETRIEVAL_QUERY" for queries
        )

        # Using with custom output_dimensionality
        ef = GoogleVertexEmbeddingFunction(
            project_id="your-project-id",
            model_name="textembedding-gecko@003",
            output_dimensionality=512  # Reduce from default 768 to 512
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
        project_id: str,
        model_name: str = "textembedding-gecko@003",
        location: str = "us-central1",
        task_type: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        api_key_env: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize GoogleVertexEmbeddingFunction.

        Args:
            project_id (str): Your Google Cloud project ID.
            model_name (str, optional): Name of the Vertex AI embedding model.
                Defaults to "textembedding-gecko@003".
                Available options:
                - "textembedding-gecko@003" (768 dimensions by default, supports 128, 256, 512, 768)
                - "textembedding-gecko@002" (768 dimensions by default, supports 128, 256, 512, 768)
                - "text-multilingual-embedding-002" (768 dimensions by default, supports 128, 256, 512, 768)
                - "gemini-embedding-001" (768 dimensions by default, supports 128, 256, 512, 768)
            location (str, optional): The region where the model is deployed.
                Defaults to "us-central1".
            task_type (str, optional): The task type for the embedding. Options:
                - "RETRIEVAL_QUERY": For queries in retrieval/search tasks
                - "RETRIEVAL_DOCUMENT": For documents in retrieval/search tasks
                - "SEMANTIC_SIMILARITY": For semantic similarity tasks
                - "CLASSIFICATION": For classification tasks
                - "CLUSTERING": For clustering tasks
                - "QUESTION_ANSWERING": For question answering tasks
                Defaults to None (no specific task type).
            output_dimensionality (int, optional): The number of dimensions for resulting embeddings.
                Supported values: 128, 256, 512, 768 (default).
                Only supported for certain models. If None, uses the model's default dimension.
            api_key_env (str, optional): Name of the environment variable containing the API key.
                For Google Vertex AI, this is typically not needed as it uses Application Default Credentials.
                If provided, will be passed to LiteLLM.
            **kwargs: Additional arguments to pass to LiteLLM.
                Common options include:
                - vertex_project: Google Cloud project ID (alternative to project_id parameter)
                - vertex_location: Region (alternative to location parameter)
                - See https://docs.litellm.ai/docs/embedding/supported_embedding for more options
        """
        # Construct LiteLLM model name format: vertex_ai/<model-name>
        litellm_model_name = f"vertex_ai/{model_name}"

        # Prepare kwargs for LiteLLM
        litellm_kwargs = {
            "vertex_project": project_id,
            "vertex_location": location,
            **kwargs,
        }

        # Add task_type and output_dimensionality if provided
        if task_type is not None:
            litellm_kwargs["task_type"] = task_type
        if output_dimensionality is not None:
            litellm_kwargs["output_dimensionality"] = output_dimensionality

        # Initialize the base class
        super().__init__(
            model_name=litellm_model_name,
            api_key_env=api_key_env,
            **litellm_kwargs,
        )

        # Store additional configuration for get_config
        self.project_id = project_id
        self._base_model_name = model_name  # Store original model name without prefix
        self.location = location
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality

        # Store dimension for quick access (will be calculated if needed)
        self._dimension = output_dimensionality
        if self._dimension is None:
            # Use default dimension from model dimensions dict if known
            model_dims = _GOOGLE_VERTEX_MODEL_DIMENSIONS
            if model_name in model_dims:
                self._dimension = model_dims[model_name]
            else:
                # Will be calculated on first access via dimension property
                self._dimension = None

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function.

        Returns the known dimension for models without making an API call.
        If the output_dimensionality parameter is specified, that value is returned.
        Otherwise, the default dimension for the model is returned.

        If the model is not in the known dimensions list, falls back to making
        an API call to get the embedding and infer the dimension.

        Returns:
            int: The dimension of embeddings for this model.
        """
        # If output_dimensionality is explicitly set, use it
        if self._dimension is not None:
            return self._dimension

        # Fallback: make an API call to get the embedding and infer the dimension
        # This is done by actually generating an embedding for a dummy sentence
        test_input = "dimension probing"
        try:
            embeddings = self([test_input])
        except Exception as e:
            raise RuntimeError(
                f"Failed to determine embedding dimension via API call: {e}"
            )
        if (
            not embeddings
            or not isinstance(embeddings, list)
            or not isinstance(embeddings[0], list)
        ):
            raise RuntimeError("Could not get embedding dimension from API response")

        # Cache the dimension for future use
        self._dimension = len(embeddings[0])
        return self._dimension

    @staticmethod
    def name() -> str:
        """Get the unique name identifier for GoogleVertexEmbeddingFunction.

        Returns:
            The name identifier for this embedding function type
        """
        return "google_vertex"

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for the GoogleVertexEmbeddingFunction.

        Returns:
            Dictionary containing configuration needed to restore this embedding function
        """
        # Get base config from parent
        base_config = super().get_config()

        # Add Google Vertex AI specific configuration
        return {
            "project_id": self.project_id,
            "model_name": self._base_model_name,
            "location": self.location,
            "task_type": self.task_type,
            "output_dimensionality": self.output_dimensionality,
            "api_key_env": self.api_key_env,
            "kwargs": base_config.get("kwargs", {}),
        }

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "GoogleVertexEmbeddingFunction":
        """Build a GoogleVertexEmbeddingFunction from its configuration dictionary.

        Args:
            config: Dictionary containing the embedding function's configuration

        Returns:
            Restored GoogleVertexEmbeddingFunction instance

        Raises:
            ValueError: If the configuration is invalid or missing required fields
        """
        project_id = config.get("project_id")
        if project_id is None:
            raise ValueError("Missing required field 'project_id' in configuration")

        model_name = config.get("model_name", "textembedding-gecko@003")
        location = config.get("location", "us-central1")
        task_type = config.get("task_type")
        output_dimensionality = config.get("output_dimensionality")
        api_key_env = config.get("api_key_env")
        kwargs = config.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(f"kwargs must be a dictionary, but got {kwargs}")

        return GoogleVertexEmbeddingFunction(
            project_id=project_id,
            model_name=model_name,
            location=location,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            api_key_env=api_key_env,
            **kwargs,
        )
