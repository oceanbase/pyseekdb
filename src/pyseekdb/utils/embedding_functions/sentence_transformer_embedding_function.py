from pyseekdb.client.embedding_function import EmbeddingFunction, Space, Embeddings, Documents
from typing import Dict, Any


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using sentence-transformers with a specific model.

    Example:
        pip install pyseekdb sentence-transformers

    .. code-block:: python
        import pyseekdb
        from pyseekdb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        db = pyseekdb.Client(
            path="./mydb"
        )
        collection = db.create_collection(name="my_collection", embedding_function=ef)
        # Add documents
        collection.add(ids=["1", "2"], documents=["Hello world", "How are you?"], metadatas=[{"id": 1}, {"id": 2}])
        # Query using semantic search
        results = collection.query("How are you?", top_k=1)
        print(results)

    """
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # for a full list of options: https://huggingface.co/sentence-transformers,
    # https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name (str, optional): Identifier of the SentenceTransformer model, defaults to "all-MiniLM-L6-v2"
            device (str, optional): Device used for computation, defaults to "cpu"
            normalize_embeddings (bool, optional): Whether to normalize returned vectors, defaults to False
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence-transformers python package is not installed. Please install it with `pip install sentence-transformers`"
            )

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        for key, value in kwargs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple)):
                raise ValueError(f"Keyword argument {key} is not a primitive type")
        self.kwargs = kwargs

        if model_name not in self.models:
            self.models[model_name] = SentenceTransformer(
                model_name_or_path=model_name, device=device, **kwargs
            )
        self._model = self.models[model_name]

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents.

        Args:
            input: Documents to generate embeddings for.

        Returns:
            Embeddings for the documents.
        """
        embeddings = self._model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return [embedding.tolist() for embedding in embeddings]
