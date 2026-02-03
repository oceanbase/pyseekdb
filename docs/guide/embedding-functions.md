# 6. Embedding Functions

Embedding functions convert text documents into vector embeddings for similarity search. pyseekdb supports both built-in and custom embedding functions.

## 6.1 Default Embedding Function

The `DefaultEmbeddingFunction` uses all-MiniLM-L6-v2 and is the default embedding function if none is specified.

```python
from pyseekdb import DefaultEmbeddingFunction

# Use default model (all-MiniLM-L6-v2, 384 dimensions)
ef = DefaultEmbeddingFunction()

# Use custom model
ef = DefaultEmbeddingFunction(model_name='all-MiniLM-L6-v2')

# Get embedding dimension
print(f"Dimension: {ef.dimension}")  # 384

# Generate embeddings
embeddings = ef(["Hello world", "How are you?"])
print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
```

## 6.2 Creating Custom Embedding Functions

You can create custom embedding functions by implementing the `EmbeddingFunction` protocol. The function must:

1. Implement `__call__` method that accepts `Documents` (str or List[str]) and returns `Embeddings` (List[List[float]])
2. Optionally implement a `dimension` property to return the vector dimension

### Example: Sentence-Transformer Custom Embedding Function

```python
from typing import List, Union
from pyseekdb import EmbeddingFunction

Documents = Union[str, List[str]]
Embeddings = List[List[float]]
Embedding = List[float]

class SentenceTransformerCustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using sentence-transformers with a specific model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the sentence-transformer embedding function.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
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
            List of embedding embeddings
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

# Use the custom embedding function
from pyseekdb import Configuration, HNSWConfiguration
ef = SentenceTransformerCustomEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device='cpu'
)
collection = client.create_collection(
    name="my_collection",
    configuration=Configuration(
        hnsw=HNSWConfiguration(dimension=384, distance='cosine')
    ),
    embedding_function=ef
)
```

### Example: OpenAI Embedding Function

```python
from typing import List, Union
import os
import openai
from pyseekdb import EmbeddingFunction

Documents = Union[str, List[str]]
Embeddings = List[List[float]]
Embedding = List[float]

class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function using OpenAI's embedding API.
    """

    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        """
        Initialize the OpenAI embedding function.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Dimension for text-embedding-ada-002 is 1536
        self._dimension = 1536 if "ada-002" in model_name else None

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        if self._dimension is None:
            # Call API to get dimension (or use known values)
            raise ValueError("Dimension not set for this model")
        return self._dimension

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings using OpenAI API.

        Args:
            input: Single document (str) or list of documents (List[str])

        Returns:
            List of embedding embeddings
        """
        # Handle single string input
        if isinstance(input, str):
            input = [input]

        # Handle empty input
        if not input:
            return []

        # Call OpenAI API
        response = openai.embeddings.create(
            model=self.model_name,
            input=input,
            api_key=self.api_key
        )

        # Extract embeddings
        embeddings = [item['embedding'] for item in response['data']]
        return embeddings

# Use the custom embedding function
from pyseekdb import Configuration, HNSWConfiguration
ef = OpenAIEmbeddingFunction(
    model_name='text-embedding-ada-002',
    api_key='your-api-key'
)
collection = client.create_collection(
    name="my_collection",
    configuration=Configuration(
        hnsw=HNSWConfiguration(dimension=1536, distance='cosine')
    ),
    embedding_function=ef
)
```

## 6.3 Embedding Function Requirements

When creating a custom embedding function, ensure:

1. **Implement `__call__` method:**
   - Accepts: `str` or `List[str]` (single document or list of documents)
   - Returns: `List[List[float]]` (list of embeddings)
   - Each vector must have the same dimension

2. **Implement `dimension` property (recommended):**
   - Returns: `int` (the dimension of embeddings produced by this function)
   - This helps validate dimension consistency when creating collections

3. **Handle edge cases:**
   - Single string input should be converted to list
   - Empty input should return empty list
   - All embeddings in the output must have the same dimension

## 6.4 Using Custom Embedding Functions

Once you've created a custom embedding function, use it when creating or getting collections:

```python
from pyseekdb import Configuration, HNSWConfiguration

# Create collection with custom embedding function
ef = MyCustomEmbeddingFunction()
collection = client.create_collection(
    name="my_collection",
    configuration=Configuration(
        hnsw=HNSWConfiguration(dimension=ef.dimension, distance='cosine')
    ),
    embedding_function=ef
)

# Get collection with custom embedding function
collection = client.get_collection("my_collection", embedding_function=ef)

# Use the collection - documents will be automatically embedded
collection.add(
    ids=["doc1", "doc2"],
    documents=["Document 1", "Document 2"],  # Embeddings auto-generated
    metadatas=[{"tag": "A"}, {"tag": "B"}]
)

# Query with texts - query embeddings auto-generated
results = collection.query(
    query_texts=["my query"],
    n_results=10
)
```
