from pyseekdb import EmbeddingFunction, DefaultEmbeddingFunction
from typing import List, Union
import os
from openai import OpenAI

Documents = Union[str, List[str]]
Embeddings = List[List[float]]

class SentenceTransformerCustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using sentence-transformers with a specific model.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "cpu"): # TODO: your own model name and device
        """
        Initialize the sentence-transformer embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name or os.environ.get('SENTENCE_TRANSFORMERS_MODEL_NAME')
        self.device = device or os.environ.get('SENTENCE_TRANSFORMERS_DEVICE')
        self._model = None
        self._dimension = None
    
    def _ensure_model_loaded(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Get dimension from model
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                self._dimension = len(test_embedding[0])
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it with: pip install sentence-transformers"
                )
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        self._ensure_model_loaded()
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors
        """
        self._ensure_model_loaded()
        
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Generate embeddings
        embeddings = self._model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]



class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using Embedding API.
    """
    
    def __init__(self, model_name: str = "", api_key: str = "", base_url: str = ""):
        """
        Initialize the Embedding API embedding function.
        
        Args:
            model_name: Name of the Embedding API embedding model
            api_key: Embedding API key (if not provided, uses EMBEDDING_API_KEY env var)
        """
        self.model_name = model_name or os.environ.get('EMBEDDING_MODEL_NAME')
        self.api_key = api_key or os.environ.get('EMBEDDING_API_KEY')
        self.base_url = base_url or os.environ.get('EMBEDDING_BASE_URL')
        self._dimension = None
        if not self.api_key:
            raise ValueError("Embedding API key is required")


    def _ensure_model_loaded(self):
        """Lazy load the Embedding API model"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            response = client.embeddings.create(
                model=self.model_name,
                input=["test"]
            )
            self._dimension = len(response.data[0].embedding)
        except Exception as e:
            raise ValueError(f"Failed to load Embedding API model: {e}")

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        self._ensure_model_loaded()
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings using Embedding API.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors
        """
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Call Embedding API
        client = OpenAI(
            api_key=self.api_key,  
            base_url=self.base_url
        )
        response = client.embeddings.create(
            model=self.model_name,
            input=input
        )
        
        # Extract Embedding API embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings


def create_embedding_function() -> EmbeddingFunction:
    embedding_function_type = os.environ.get('EMBEDDING_FUNCTION_TYPE')
    if embedding_function_type == "api":
        print("Using OpenAI Embedding API embedding function")
        return OpenAIEmbeddingFunction()
    elif embedding_function_type == "local":
        print("Using SentenceTransformer embedding function")
        return SentenceTransformerCustomEmbeddingFunction()
    elif embedding_function_type == "default":
        print("Using Default embedding function")
        return DefaultEmbeddingFunction()
    else:
        raise ValueError(f"Unsupported embedding function type: {embedding_function_type}")
