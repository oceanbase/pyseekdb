"""
Embedding function interface and implementations

This module provides the EmbeddingFunction protocol and default implementations
for converting text documents to vector embeddings.
"""

import contextlib
import logging
import os
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

import httpx
import numpy as np
import numpy.typing as npt

# Set Hugging Face mirror endpoint for better download speed in China
# Users can override this by setting HF_ENDPOINT environment variable
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

# Type variable for input types
D = TypeVar("D")

# Type aliases
Documents = str | list[str]
Embeddings = list[list[float]]
Embedding = list[float]


@runtime_checkable
class EmbeddingFunction(Protocol[D]):
    """
    Protocol for embedding functions that convert documents to vectors.

    This is similar to Chroma's EmbeddingFunction interface.
    Implementations should convert text documents to vector embeddings.

    Implementations should also provide:
    - `name()`: Static method that returns a unique name identifier for routing (not persisted in config)
    - `get_config()`: Instance method that returns a configuration dictionary
    - `build_from_config(config)`: Static method that restores an instance from config

    Example:
        >>> class MyEmbeddingFunction(EmbeddingFunction[Documents]):
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_embedding_function"
        ...     def __call__(self, documents: Documents) -> Embeddings:
        ...         # Convert documents to embeddings
        ...         return [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {...}  # Note: 'name' is not included
        ...     @staticmethod
        ...     def build_from_config(config: Dict[str, Any]) -> "MyEmbeddingFunction":
        ...         return MyEmbeddingFunction(...)
        >>>
        >>> ef = MyEmbeddingFunction()
        >>> embeddings = ef(["Hello", "World"])
        >>> config = ef.get_config()
        >>> restored_ef = MyEmbeddingFunction.build_from_config(config)
    """

    @abstractmethod
    def __call__(self, documents: D) -> Embeddings:
        """
        Convert input documents to embeddings.

        Args:
            documents: Documents to embed (can be a single string or list of strings)

        Returns:
            List of embedding vectors (list of floats)
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary for the embedding function.

        This method should return a dictionary that contains all the information
        needed to restore the embedding function after restart.

        Returns:
            Dictionary containing the embedding function's configuration.
            Note: The 'name' field is not included as it's handled by the upper layer for routing.
        """
        return NotImplemented

    @staticmethod
    def support_persistence(embedding_function: Any) -> bool:
        """
        Check if the embedding function supports persistence.
        """
        if embedding_function is None:
            return False
        if (
            not hasattr(embedding_function, "name")
            or not hasattr(embedding_function, "build_from_config")
            or not hasattr(embedding_function, "get_config")
        ):
            return False
        try:
            if embedding_function.get_config() is NotImplemented:
                return False
        except Exception:
            return False
        return True


def dimension_of(embedding_function: EmbeddingFunction[D]) -> int:
    """
    Get the dimension of the embeddings produced by the embedding function.
    """
    if hasattr(embedding_function, "dimension") and callable(getattr(embedding_function, "dimension", None)):
        return embedding_function.dimension()
    elif hasattr(embedding_function, "dimension"):
        return embedding_function.dimension
    else:
        # Fallback: if no dimension attribute, call the function to calculate dimension
        # This may trigger model initialization, but is necessary for custom embedding functions
        test_embeddings = embedding_function.__call__("seekdb")
        if test_embeddings and len(test_embeddings) > 0:
            return len(test_embeddings[0])
        else:
            raise ValueError("Embedding function returned empty result when called with 'seekdb'")


class DefaultEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Default embedding function using ONNX runtime.

    Uses the 'all-MiniLM-L6-v2' model via ONNX, which produces 384-dimensional embeddings.
    This is a lightweight, fast model suitable for general-purpose text embeddings.

    Example:
        >>> ef = DefaultEmbeddingFunction()
        >>> embeddings = ef(["Hello world", "How are you?"])
        >>> print(len(embeddings[0]))  # 384
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face model ID
    DOWNLOAD_PATH = Path.home() / ".cache" / "pyseekdb" / "onnx_models" / MODEL_NAME
    EXTRACTED_FOLDER_NAME = "onnx"
    ARCHIVE_FILENAME = "onnx.tar.gz"
    _DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        preferred_providers: list[str] | None = None,
    ):
        """
        Initialize the default embedding function.

        Args:
            model_name: Name of the model (currently only 'all-MiniLM-L6-v2' is supported).
                       Default is 'all-MiniLM-L6-v2' (384 dimensions).
            preferred_providers: The preferred ONNX runtime providers.
                                Defaults to None (uses available providers).
        """
        if model_name != "all-MiniLM-L6-v2":
            raise ValueError(f"Currently only 'all-MiniLM-L6-v2' is supported, got '{model_name}'")
        self.model_name = model_name

        # Validate preferred_providers
        if preferred_providers and not all(isinstance(i, str) for i in preferred_providers):
            raise ValueError("Preferred providers must be a list of strings")
        if preferred_providers and len(preferred_providers) != len(set(preferred_providers)):
            raise ValueError("Preferred providers must be unique")

        self._preferred_providers = preferred_providers

        # Import required modules
        import onnxruntime as ort_module
        import tokenizers
        import tqdm

        self.ort = ort_module
        self.tokenizers = tokenizers  # Store the module
        self.tqdm = tqdm.tqdm

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        return self._DIMENSION

    def _download(self, url: str, fname: str, chunk_size: int = 8192) -> None:
        """
        Download a file from the URL and save it to the file path.

        Args:
            url: The URL to download the file from.
            fname: The path to save the file to.
            chunk_size: The chunk size to use when downloading (default: 8192 for better speed).
        """
        logger.info(f"Downloading from {url}")
        # Use Client to ensure correct handling of redirects
        with httpx.Client(timeout=600.0, follow_redirects=True) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                open(fname, "wb") as file,
                self.tqdm(
                    desc=os.path.basename(fname),
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in resp.iter_bytes(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

    def _get_hf_endpoint(self) -> str:
        """Get Hugging Face endpoint URL, using HF_ENDPOINT environment variable if set."""
        return os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    def _download_from_huggingface(self) -> bool:  # noqa: C901
        """
        Download model files from Hugging Face (supports mirror acceleration).

        Returns:
            True if download successful, False otherwise.
        """
        try:
            hf_endpoint = self._get_hf_endpoint()
            # Remove trailing slash
            hf_endpoint = hf_endpoint.rstrip("/")

            # List of files to download
            # ONNX model files are in the onnx/ subdirectory, other files in the root directory
            files_to_download = {
                "onnx/model.onnx": "model.onnx",  # ONNX file in onnx subdirectory
                "tokenizer.json": "tokenizer.json",
                "config.json": "config.json",
                "special_tokens_map.json": "special_tokens_map.json",
                "tokenizer_config.json": "tokenizer_config.json",
                "vocab.txt": "vocab.txt",
            }

            extracted_folder = os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME)
            os.makedirs(extracted_folder, exist_ok=True)

            logger.info(f"Downloading model from Hugging Face (endpoint: {hf_endpoint})")

            # Download each file
            for hf_filename, local_filename in files_to_download.items():
                local_path = os.path.join(extracted_folder, local_filename)

                # Skip if file already exists
                if os.path.exists(local_path):
                    continue

                # Construct Hugging Face download URL
                # Format: https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
                # Or: https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
                url = f"{hf_endpoint}/{self.HF_MODEL_ID}/resolve/main/{hf_filename}"

                try:
                    # First check if file exists (HEAD request)
                    with contextlib.suppress(Exception):
                        head_resp = httpx.head(url, timeout=10.0, follow_redirects=True)
                        if head_resp.status_code == 404:
                            logger.warning(f"File {hf_filename} not found on Hugging Face (404), will try fallback")
                            return False

                    self._download(url, local_path, chunk_size=8192)
                    logger.info(f"Successfully downloaded {local_filename}")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        logger.warning(f"File {hf_filename} not found on Hugging Face (404), will try fallback")
                        return False
                    logger.warning(f"HTTP error downloading {hf_filename} from Hugging Face: {e}")
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
                except Exception as e:
                    logger.warning(f"Failed to download {hf_filename} from Hugging Face: {e}")
                    # If download fails, try to delete partially downloaded file
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False

            # Verify critical files exist
            if not os.path.exists(os.path.join(extracted_folder, "model.onnx")):
                logger.error("model.onnx not found after download")
                return False
            if not os.path.exists(os.path.join(extracted_folder, "tokenizer.json")):
                logger.error("tokenizer.json not found after download")
                return False

            logger.info("Successfully downloaded all model files from Hugging Face")
            return True  # noqa: TRY300

        except Exception:
            logger.exception("Error downloading from Hugging Face")
            return False

    def _forward(self, documents: list[str], batch_size: int = 32) -> npt.NDArray[np.float32]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: The documents to generate embeddings for.
            batch_size: The batch size to use when generating embeddings.

        Returns:
            The embeddings for the documents.
        """
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Encode each document separately
            encoded = [self.tokenizer.encode(d) for d in batch]

            # Check if any document exceeds the max tokens
            for doc_tokens in encoded:
                if len(doc_tokens.ids) > self.max_tokens():
                    raise ValueError(
                        f"Document length {len(doc_tokens.ids)} is greater than the max tokens {self.max_tokens()}"
                    )

            # Create input arrays exactly like the working standalone script
            # Create input arrays, ensuring int64 type
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            # Ensure 2D arrays (batch_size, seq_length)
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.reshape(1, -1)

            # Use zeros_like to create token_type_ids, ensuring exact shape match
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            # Ensure all arrays are contiguous, which is important for onnxruntime 1.19.0
            input_ids = np.ascontiguousarray(input_ids, dtype=np.int64)
            attention_mask = np.ascontiguousarray(attention_mask, dtype=np.int64)
            token_type_ids = np.ascontiguousarray(token_type_ids, dtype=np.int64)

            onnx_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]

            # Mean pooling (exactly as in the code)
            # Note: attention_mask needs to be converted to float type for floating point operations
            attention_mask_float = attention_mask.astype(np.float32)
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask_float, -1), last_hidden_state.shape)
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )

            embeddings = embeddings.astype(np.float32)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings)

    @cached_property
    def tokenizer(self) -> Any:
        """
        Get the tokenizer for the model.

        Returns:
            The tokenizer for the model.
        """
        tokenizer = self.tokenizers.Tokenizer.from_file(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "tokenizer.json")
        )
        # max_seq_length = 256, for some reason sentence-transformers uses 256
        # even though the HF config has a max length of 128
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)  # noqa: S106
        return tokenizer

    @cached_property
    def model(self) -> Any:
        """
        Get the model.

        Returns:
            The model.
        """
        if self._preferred_providers is None or len(self._preferred_providers) == 0:
            if len(self.ort.get_available_providers()) > 0:
                logger.debug(
                    f"WARNING: No ONNX providers provided, defaulting to available providers: "
                    f"{self.ort.get_available_providers()}"
                )
            self._preferred_providers = self.ort.get_available_providers()
        elif not set(self._preferred_providers).issubset(set(self.ort.get_available_providers())):
            raise ValueError(
                f"Preferred providers must be subset of available providers: {self.ort.get_available_providers()}"
            )

        # Create minimal session options to avoid issues
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        # Disable all optimizations that might cause issues
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        if self._preferred_providers and "CoreMLExecutionProvider" in self._preferred_providers:
            # remove CoreMLExecutionProvider from the list, it is not as well optimized as CPU.
            self._preferred_providers.remove("CoreMLExecutionProvider")

        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            # Force CPU execution provider to avoid provider issues
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )

    def _download_model_if_not_exists(self) -> None:
        """
        Download from Hugging Face with image mirror if the model doesn't exist.
        """
        onnx_files = [
            "config.json",
            "model.onnx",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        ]
        extracted_folder = os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME)
        onnx_files_exist = True
        for f in onnx_files:
            if not os.path.exists(os.path.join(extracted_folder, f)):
                onnx_files_exist = False
                break

        # Model is not downloaded yet
        if not onnx_files_exist:
            os.makedirs(self.DOWNLOAD_PATH, exist_ok=True)

            logger.info("Attempting to download model from Hugging Face...")
            hf_endpoint = self._get_hf_endpoint()
            if not self._download_from_huggingface():
                raise RuntimeError(
                    f"Failed to download model from Hugging Face (endpoint: {hf_endpoint}). "
                    f"Please check your network connection or set HF_ENDPOINT environment variable "
                    f"to use a mirror site (e.g., export HF_ENDPOINT='https://hf-mirror.com'). "
                    f"Model ID: {self.HF_MODEL_ID}"
                )
            logger.info("Model downloaded successfully from Hugging Face")

    def max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model."""
        return 256

    def __call__(self, documents: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Args:
            documents: Single document (str) or list of documents (List[str])

        Returns:
            List of embedding vectors

        Example:
            >>> ef = DefaultEmbeddingFunction()
            >>> # Single document
            >>> embedding = ef("Hello world")
            >>> # Multiple documents
            >>> embeddings = ef(["Hello", "World"])
        """
        # Handle single string input
        if isinstance(documents, str):
            documents = [documents]

        # Handle empty input
        if not documents:
            return []

        # Only download the model when it is actually used
        self._download_model_if_not_exists()

        # Generate embeddings
        embeddings = self._forward(documents)

        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]

    @staticmethod
    def name() -> str:
        return "default"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(_config: dict[str, Any]) -> Self:
        return DefaultEmbeddingFunction()

    def __repr__(self) -> str:
        return f"DefaultEmbeddingFunction(model_name='{self.model_name}')"


# Global default embedding function instance
_default_embedding_function: DefaultEmbeddingFunction | None = None


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


class EmbeddingFunctionRegistry:
    """
    Registry for embedding function classes.

    This registry maps embedding function names (returned by their name() method)
    to their corresponding classes, allowing dynamic instantiation from persisted configurations.

    To register a custom embedding function, you have two options:

    Option 1 (Recommended): Use the @register_embedding_function decorator:
       >>> @register_embedding_function
       ... class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
       ...     # ... implementation ...

    Option 2: Manually register the class:
       >>> EmbeddingFunctionRegistry.register(MyCustomEmbeddingFunction)

    Your embedding function class must implement:
       - __call__() to convert documents to embeddings
       - A static name() method that returns a unique identifier
       - get_config() to return configuration dictionary
       - A static build_from_config() to restore from configuration

    Example:
        >>> from pyseekdb.client.embedding_function import (
        ...     EmbeddingFunction, Documents, Embeddings, EmbeddingFunctionRegistry
        ... )
        >>> from typing import Dict, Any
        >>>
        >>> class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
        ...     def __init__(self, model_name: str = "my-model", dimension: int = 128):
        ...         self.model_name = model_name
        ...         self._dimension = dimension
        ...
        ...     def __call__(self, input: Documents) -> Embeddings:
        ...         # Your embedding logic here
        ...         if isinstance(input, str):
        ...             input = [input]
        ...         # Return list of embedding vectors
        ...         return [[0.1] * self._dimension for _ in input]
        ...
        ...     @property
        ...     def dimension(self) -> int:
        ...         return self._dimension
        ...
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_custom_embedding"
        ...
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {
        ...             "model_name": self.model_name,
        ...             "dimension": self._dimension,
        ...         }
        ...
        ...     @staticmethod
        ...     def build_from_config(config: Dict[str, Any]) -> "MyCustomEmbeddingFunction":
        ...         return MyCustomEmbeddingFunction(
        ...             model_name=config.get("model_name", "my-model"),
        ...             dimension=config.get("dimension", 128),
        ...         )
        >>>
        >>> # Register your custom embedding function
        >>> EmbeddingFunctionRegistry.register(MyCustomEmbeddingFunction)
        >>>
        >>> # Now you can use it when creating collections
        >>> import pyseekdb
        >>> client = pyseekdb.Client(path="./db")
        >>> ef = MyCustomEmbeddingFunction()
        >>> collection = client.create_collection("my_collection", embedding_function=ef)
        >>>
        >>> # When the collection is retrieved later, it will automatically restore
        >>> # the embedding function using the registry
        >>> collection2 = client.get_collection("my_collection")
    """

    _registry: ClassVar[dict[str, type]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the registry with built-in embedding functions."""
        if cls._initialized:
            return

        # Register DefaultEmbeddingFunction
        cls._registry["default"] = DefaultEmbeddingFunction

        # Try to register optional embedding functions (may not be installed)
        try:
            from pyseekdb.utils.embedding_functions import (
                AmazonBedrockEmbeddingFunction,
                CohereEmbeddingFunction,
                GoogleVertexEmbeddingFunction,
                JinaEmbeddingFunction,
                OllamaEmbeddingFunction,
                OpenAIEmbeddingFunction,
                QwenEmbeddingFunction,
                SentenceTransformerEmbeddingFunction,
                SiliconflowEmbeddingFunction,
                TencentHunyuanEmbeddingFunction,
                VoyageaiEmbeddingFunction,
            )

            cls._registry["sentence_transformer"] = SentenceTransformerEmbeddingFunction
            cls._registry["openai"] = OpenAIEmbeddingFunction
            cls._registry["qwen"] = QwenEmbeddingFunction
            cls._registry["siliconflow"] = SiliconflowEmbeddingFunction
            cls._registry["tencent_hunyuan"] = TencentHunyuanEmbeddingFunction
            cls._registry["ollama"] = OllamaEmbeddingFunction
            cls._registry["voyageai"] = VoyageaiEmbeddingFunction
            cls._registry["google_vertex"] = GoogleVertexEmbeddingFunction
            cls._registry["cohere"] = CohereEmbeddingFunction
            cls._registry["jina"] = JinaEmbeddingFunction
            cls._registry["amazon_bedrock"] = AmazonBedrockEmbeddingFunction
        except ImportError as e:
            # Optional dependencies not installed, skip registration
            logger.warning(f"Failed to register some embedding function classes: {e}")

        cls._initialized = True

    @classmethod
    def register(cls, embedding_function_class: type) -> None:
        """
        Register an embedding function class.

        This method should be called before creating collections that use the custom
        embedding function. Once registered, the embedding function can be automatically
        restored from persisted collection metadata.

        Args:
            embedding_function_class: The embedding function class to register.
                                    Must implement:
                                    - A static name() method that returns a unique identifier
                                    - A get_config() instance method that returns configuration dict
                                    - A static build_from_config(config) method to restore instances

        Raises:
            ValueError: If the class doesn't have the required methods or if the name
                       is already registered to a different class.

        Example:
            >>> from pyseekdb.client.embedding_function import EmbeddingFunctionRegistry
            >>> EmbeddingFunctionRegistry.register(MyCustomEmbeddingFunction)
            >>>
            >>> # Verify registration
            >>> assert "my_custom_embedding" in EmbeddingFunctionRegistry.list_registered()
        """
        cls._initialize()

        if not hasattr(embedding_function_class, "name") or not hasattr(embedding_function_class, "build_from_config"):
            raise ValueError(
                f"Embedding function class {embedding_function_class.__name__} "
                f"must have a static name() method, static build_from_config() method"
            )

        name = embedding_function_class.name()
        if name in cls._registry and cls._registry[name] != embedding_function_class:
            raise ValueError(
                f"Embedding function name '{name}' is already registered to {cls._registry[name].__name__}"
            )

        cls._registry[name] = embedding_function_class
        logger.debug(f"Registered embedding function '{name}' -> {embedding_function_class.__name__}")

    @classmethod
    def get_class(cls, name: str) -> type | None:
        """
        Get an embedding function class by name.

        Args:
            name: The name identifier of the embedding function (as returned by its name() method).

        Returns:
            The embedding function class if found, None otherwise.
        """
        cls._initialize()
        return cls._registry.get(name)

    @classmethod
    def list_registered(cls) -> list[str]:
        """
        List all registered embedding function names.

        Returns:
            List of registered embedding function names.
        """
        cls._initialize()
        return list(cls._registry.keys())


T = TypeVar("T", bound=type)


def register_embedding_function(embedding_function_class: type[T]) -> type[T]:
    """
    Decorator to automatically register an embedding function class.

    This decorator can be used as a class decorator to automatically register
    an embedding function when the class is defined, eliminating the need to
    manually call EmbeddingFunctionRegistry.register().

    Args:
        embedding_function_class: The embedding function class to register.
                                Must implement:
                                - A static name() method that returns a unique identifier
                                - A get_config() instance method that returns configuration dict
                                - A static build_from_config(config) method to restore instances

    Returns:
        The same class (for use as a decorator).

    Raises:
        ValueError: If the class doesn't have the required methods or if the name
                   is already registered to a different class.

    Example:
        >>> from pyseekdb.client.embedding_function import (
        ...     EmbeddingFunction, Documents, Embeddings, register_embedding_function
        ... )
        >>> from typing import Dict, Any
        >>>
        >>> @register_embedding_function
        ... class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
        ...     def __init__(self, model_name: str = "my-model"):
        ...         self.model_name = model_name
        ...
        ...     def __call__(self, input: list[str]|str) -> list[list[float]]:
        ...         # Your embedding logic
        ...         return [[0.1, 0.2, 0.3] for _ in (input if isinstance(input, list) else [input])]
        ...
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_custom_embedding"
        ...
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {"model_name": self.model_name}
        ...
        ...     @staticmethod
        ...     def build_from_config(config: Dict[str, Any]) -> "MyCustomEmbeddingFunction":
        ...         return MyCustomEmbeddingFunction(model_name=config.get("model_name", "my-model"))
        >>>
        >>> # The class is now automatically registered!
        >>> # You can use it immediately when creating collections
        >>> import pyseekdb
        >>> client = pyseekdb.Client(path="./seekdb.db")
        >>> ef = MyCustomEmbeddingFunction()
        >>> collection = client.create_collection("my_collection", embedding_function=ef)
    """
    EmbeddingFunctionRegistry.register(embedding_function_class)
    return embedding_function_class
