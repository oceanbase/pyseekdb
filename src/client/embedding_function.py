"""
Embedding function interface and implementations

This module provides the EmbeddingFunction protocol and default implementations
for converting text documents to vector embeddings.
"""
import importlib
import logging
import os
import sys
import tarfile
from functools import cached_property
from pathlib import Path
from typing import List, Protocol, Union, runtime_checkable, Optional, TypeVar, cast, Any

import numpy as np
import numpy.typing as npt
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random

# Set Hugging Face mirror endpoint for better download speed in China
# Users can override this by setting HF_ENDPOINT environment variable
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

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
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", preferred_providers: Optional[List[str]] = None):
        """
        Initialize the default embedding function.
        
        Args:
            model_name: Name of the model (currently only 'all-MiniLM-L6-v2' is supported).
                       Default is 'all-MiniLM-L6-v2' (384 dimensions).
            preferred_providers: The preferred ONNX runtime providers.
                                Defaults to None (uses available providers).
        """
        if model_name != "all-MiniLM-L6-v2":
            raise ValueError(
                f"Currently only 'all-MiniLM-L6-v2' is supported, got '{model_name}'"
            )
        self.model_name = model_name
        
        # Validate preferred_providers
        if preferred_providers and not all(
            [isinstance(i, str) for i in preferred_providers]
        ):
            raise ValueError("Preferred providers must be a list of strings")
        if preferred_providers and len(preferred_providers) != len(
            set(preferred_providers)
        ):
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
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(fname, "wb") as file, self.tqdm(
                    desc=os.path.basename(fname),
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in resp.iter_bytes(chunk_size=chunk_size):
                        size = file.write(data)
                        bar.update(size)

    def _get_hf_endpoint(self) -> str:
        """Get Hugging Face endpoint URL, using HF_ENDPOINT environment variable if set."""
        return os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    
    def _download_from_huggingface(self) -> bool:
        """
        Download model files from Hugging Face (supports mirror acceleration).

        Returns:
            True if download successful, False otherwise.
        """
        try:
            hf_endpoint = self._get_hf_endpoint()
            # Remove trailing slash
            hf_endpoint = hf_endpoint.rstrip('/')
            
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
                    try:
                        head_resp = httpx.head(url, timeout=10.0, follow_redirects=True)
                        if head_resp.status_code == 404:
                            logger.warning(f"File {hf_filename} not found on Hugging Face (404), will try fallback")
                            return False
                    except Exception:
                        # If HEAD request fails, continue with GET request
                        pass
                    
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
                    # 如果下载失败，尝试删除部分下载的文件
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
            
            # 验证关键文件是否存在
            if not os.path.exists(os.path.join(extracted_folder, "model.onnx")):
                logger.error("model.onnx not found after download")
                return False
            if not os.path.exists(os.path.join(extracted_folder, "tokenizer.json")):
                logger.error("tokenizer.json not found after download")
                return False
            
            logger.info("Successfully downloaded all model files from Hugging Face")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading from Hugging Face: {e}")
            return False
    
    def _forward(
        self, documents: List[str], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
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
                        f"Document length {len(doc_tokens.ids)} is greater than "
                        f"the max tokens {self.max_tokens()}"
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
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }

            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]

            # Mean pooling (exactly as in the code)
            # Note: attention_mask needs to be converted to float type for floating point operations
            attention_mask_float = attention_mask.astype(np.float32)
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask_float, -1), last_hidden_state.shape
            )
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
            os.path.join(
                self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "tokenizer.json"
            )
        )
        # max_seq_length = 256, for some reason sentence-transformers uses 256
        # even though the HF config has a max length of 128
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
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
        elif not set(self._preferred_providers).issubset(
            set(self.ort.get_available_providers())
        ):
            raise ValueError(
                f"Preferred providers must be subset of available providers: "
                f"{self.ort.get_available_providers()}"
            )

        # Create minimal session options to avoid issues
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        # Disable all optimizations that might cause issues
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        if (
            self._preferred_providers
            and "CoreMLExecutionProvider" in self._preferred_providers
        ):
            # remove CoreMLExecutionProvider from the list, it is not as well optimized as CPU.
            self._preferred_providers.remove("CoreMLExecutionProvider")

        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            # Force CPU execution provider to avoid provider issues
            providers=['CPUExecutionProvider'],
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
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Only download the model when it is actually used
        self._download_model_if_not_exists()
        
        # Generate embeddings
        embeddings = self._forward(input)
        
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