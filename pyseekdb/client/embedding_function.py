"""
Embedding function interface and implementations

This module provides the EmbeddingFunction protocol and default implementations
for converting text documents to vector embeddings.
"""
import os
from typing import List, Protocol, Union, runtime_checkable, Optional, TypeVar

# Set Hugging Face mirror endpoint for better download speed in China
# Users can override this by setting HF_ENDPOINT environment variable
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Type variable for input types
D = TypeVar('D')

# Type aliases
Documents = Union[str, List[str]]
Embeddings = List[List[float]]
Embedding = List[float]


@runtime_checkable
class EmbeddingFunction(Protocol[D]):
    """
    Protocol for embedding functions that convert documents to vectors.
    
    This is similar to Chroma's EmbeddingFunction interface.
    Implementations should convert text documents to vector embeddings.
    
    Example:
        >>> class MyEmbeddingFunction:
        ...     def __call__(self, input: Documents) -> Embeddings:
        ...         # Convert documents to embeddings
        ...         return [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        >>> 
        >>> ef = MyEmbeddingFunction()
        >>> embeddings = ef(["Hello", "World"])
    """
    
    def __call__(self, input: D) -> Embeddings:
        """
        Convert input documents to embeddings.
        
        Args:
            input: Documents to embed (can be a single string or list of strings)
            
        Returns:
            List of embedding vectors (list of floats)
        """
        ...


class DefaultEmbeddingFunction:
    """
    Default embedding function using sentence-transformers.
    
    Uses the 'all-MiniLM-L6-v2' model by default, which produces 384-dimensional embeddings.
    This is a lightweight, fast model suitable for general-purpose text embeddings.
    
    Example:
        >>> ef = DefaultEmbeddingFunction()
        >>> embeddings = ef(["Hello world", "How are you?"])
        >>> print(len(embeddings[0]))  # 384
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the default embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (384 dimensions).
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    def _ensure_model_loaded(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get dimension from model
                # Create a dummy embedding to get the dimension
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
            
        Example:
            >>> ef = DefaultEmbeddingFunction()
            >>> # Single document
            >>> embedding = ef("Hello world")
            >>> # Multiple documents
            >>> embeddings = ef(["Hello", "World"])
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
    
    def __repr__(self) -> str:
        return f"DefaultEmbeddingFunction(model_name='{self.model_name}')"


# Global default embedding function instance
_default_embedding_function: Optional[DefaultEmbeddingFunction] = None


def get_default_embedding_function() -> DefaultEmbeddingFunction:
    """
    Get or create the default embedding function instance.
    
    Returns:
        DefaultEmbeddingFunction instance
    """
    global _default_embedding_function
    if _default_embedding_function is None:
        _default_embedding_function = DefaultEmbeddingFunction()
    return _default_embedding_function

