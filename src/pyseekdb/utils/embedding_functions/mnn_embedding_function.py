"""
MNN-based embedding function implementation.

The default embedding model is downloaded in ONNX format from Hugging Face on
first use and converted to MNN format locally.  Keeping the source ONNX model
in the cache makes the conversion reproducible when the MNN runtime changes.
"""

from __future__ import annotations

import contextlib
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Documents = str | list[str]
Embeddings = list[list[float]]


class MnnEmbeddingFunction:
    """Generate sentence embeddings with an MNN-converted model."""

    MODEL_FOLDER_NAME = "mnn"
    MNN_MODEL_FILENAME = "model.mnn"
    ONNX_MODEL_FILENAME = "model.onnx"
    MAX_TOKENS = 256

    _MODEL_FILES = (
        "config.json",
        "model.onnx",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    )

    def __init__(
        self,
        model_name: str,
        hf_model_id: str,
        dimension: int,
        download_path: Path | None = None,
    ):
        """Initialize an MNN embedding function.

        Args:
            model_name: Name of the model used for cache directory naming.
            hf_model_id: Hugging Face model ID used for the first download.
            dimension: Output embedding dimension.
            download_path: Optional cache path override.
        """
        if not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not hf_model_id:
            raise ValueError("hf_model_id must be a non-empty string")
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.model_name = model_name
        self.hf_model_id = hf_model_id
        self._dimension = dimension
        self.download_path = (
            download_path
            if download_path is not None
            else Path.home() / ".cache" / "pyseekdb" / "mnn_models" / model_name
        )

        # These imports are intentionally lazy so importing pyseekdb does not
        # initialize the native MNN runtime until embeddings are requested.
        import MNN
        import tokenizers
        import tqdm

        self.MNN = MNN
        self.tokenizers = tokenizers
        self.tqdm = tqdm.tqdm

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function."""
        return self._dimension

    @property
    def _model_folder(self) -> Path:
        """Return the directory containing downloaded and converted files."""
        return self.download_path / self.MODEL_FOLDER_NAME

    @property
    def _mnn_model_path(self) -> Path:
        """Return the converted MNN model path."""
        return self._model_folder / self.MNN_MODEL_FILENAME

    @property
    def _onnx_model_path(self) -> Path:
        """Return the downloaded ONNX source model path."""
        return self._model_folder / self.ONNX_MODEL_FILENAME

    def _download(self, url: str, fname: Path, chunk_size: int = 8192) -> None:
        """Download a file from ``url`` to ``fname``."""
        logger.info("Downloading from %s", url)
        import httpx

        with httpx.Client(timeout=600.0, follow_redirects=True) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                fname.open("wb") as file,
                self.tqdm(
                    desc=fname.name,
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
        """Get the Hugging Face endpoint, honoring ``HF_ENDPOINT``."""
        return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

    def _download_from_huggingface(self) -> bool:
        """Download the ONNX model and tokenizer files from Hugging Face."""
        try:
            import httpx

            self._model_folder.mkdir(parents=True, exist_ok=True)
            hf_endpoint = self._get_hf_endpoint()
            files_to_download = {
                "onnx/model.onnx": self._model_folder / "model.onnx",
                "tokenizer.json": self._model_folder / "tokenizer.json",
                "config.json": self._model_folder / "config.json",
                "special_tokens_map.json": self._model_folder / "special_tokens_map.json",
                "tokenizer_config.json": self._model_folder / "tokenizer_config.json",
                "vocab.txt": self._model_folder / "vocab.txt",
            }

            logger.info("Downloading model from Hugging Face (endpoint: %s)", hf_endpoint)
            for hf_filename, local_path in files_to_download.items():
                if local_path.exists():
                    continue

                url = f"{hf_endpoint}/{self.hf_model_id}/resolve/main/{hf_filename}"
                try:
                    with contextlib.suppress(Exception):
                        head_resp = httpx.head(url, timeout=10.0, follow_redirects=True)
                        if head_resp.status_code == 404:
                            logger.warning("File %s was not found on Hugging Face", hf_filename)
                            return False

                    self._download(url, local_path)
                except httpx.HTTPStatusError as exc:
                    logger.warning("HTTP error downloading %s: %s", hf_filename, exc)
                    local_path.unlink(missing_ok=True)
                    return False
                except Exception as exc:
                    logger.warning("Failed to download %s: %s", hf_filename, exc)
                    local_path.unlink(missing_ok=True)
                    return False

            return all((self._model_folder / filename).exists() for filename in self._MODEL_FILES)
        except Exception:
            logger.exception("Error downloading the MNN embedding model")
            return False

    def _convert_onnx_to_mnn(self) -> None:
        """Convert the downloaded ONNX model into an MNN model once."""
        if self._mnn_model_path.exists():
            return
        if not self._onnx_model_path.exists():
            raise RuntimeError(f"ONNX model does not exist: {self._onnx_model_path}")

        try:
            from MNN.tools import mnnconvert
        except ImportError as exc:
            raise RuntimeError(
                "The installed MNN package does not include the ONNX converter. "
                "Please install an MNN wheel with converter support."
            ) from exc

        temporary_path = self._mnn_model_path.with_suffix(".mnn.tmp")
        temporary_path.unlink(missing_ok=True)
        logger.info("Converting %s to %s", self._onnx_model_path, self._mnn_model_path)
        args = [
            "mnnconvert",
            "-f",
            "ONNX",
            "--modelFile",
            str(self._onnx_model_path),
            "--MNNModel",
            str(temporary_path),
            "--bizCode",
            "MNN",
            "--transformerFuse",
            "1",
        ]
        try:
            result = mnnconvert.Tools.mnnconvert(args)
        except (Exception, SystemExit) as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to convert the ONNX model to MNN: {exc}") from exc

        if result not in (None, True, 0) or not temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError("MNN did not produce a converted model")
        temporary_path.replace(self._mnn_model_path)
        logger.info("MNN model conversion completed: %s", self._mnn_model_path)

    def _download_model_if_not_exists(self) -> None:
        """Download the source model and convert it to MNN on first use."""
        required_files = [self._model_folder / filename for filename in self._MODEL_FILES]
        if not all(path.exists() for path in required_files):
            self._model_folder.mkdir(parents=True, exist_ok=True)
            if not self._download_from_huggingface():
                raise RuntimeError(
                    f"Failed to download model from Hugging Face (endpoint: {self._get_hf_endpoint()}). "
                    "Please check your network connection or set HF_ENDPOINT to use a mirror site. "
                    f"Model ID: {self.hf_model_id}"
                )

        self._convert_onnx_to_mnn()

    def _make_input_tensor(self, data: npt.NDArray[np.int32]) -> Any:
        """Create an MNN host tensor for token IDs and masks."""
        return self.MNN.Tensor(
            data.shape,
            self.MNN.Halide_Type_Int,
            data,
            self.MNN.Tensor_DimensionType_Tensorflow,
        )

    def _run_model(
        self,
        input_ids: npt.NDArray[np.int32],
        attention_mask: npt.NDArray[np.int32],
        token_type_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:
        """Run a batch through the MNN model and return its first output."""
        input_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        inputs = self.interpreter.getSessionInputAll(self.session)
        for name, data in input_data.items():
            tensor = inputs.get(name)
            if tensor is None:
                raise RuntimeError(f"MNN model is missing expected input tensor: {name}")
            tensor.copyFrom(self._make_input_tensor(data))

        result = self.interpreter.runSession(self.session)
        if result != 0:
            raise RuntimeError(f"MNN inference failed with error code {result}")
        outputs = self.interpreter.getSessionOutputAll(self.session)
        if not outputs:
            raise RuntimeError("MNN model did not produce an output tensor")
        output = next(iter(outputs.values())).getNumpyData()
        return np.asarray(output, dtype=np.float32).copy()

    def _forward(self, documents: list[str], batch_size: int = 32) -> npt.NDArray[np.float32]:
        """Generate embeddings for a list of documents."""
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            encoded = [self.tokenizer.encode(document) for document in batch]

            for doc_tokens in encoded:
                if len(doc_tokens.ids) > self.max_tokens():
                    raise ValueError(
                        f"Document length {len(doc_tokens.ids)} is greater than the max tokens {self.max_tokens()}"
                    )

            input_ids = np.ascontiguousarray([item.ids for item in encoded], dtype=np.int32)
            attention_mask = np.ascontiguousarray([item.attention_mask for item in encoded], dtype=np.int32)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int32)
            last_hidden_state = self._run_model(input_ids, attention_mask, token_type_ids)

            attention_mask_float = attention_mask.astype(np.float32)
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask_float, -1), last_hidden_state.shape)
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )
            all_embeddings.append(embeddings.astype(np.float32))

        return np.concatenate(all_embeddings)

    @cached_property
    def tokenizer(self) -> Any:
        """Load the cached Hugging Face tokenizer JSON."""
        tokenizer = self.tokenizers.Tokenizer.from_file(str(self._model_folder / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_tokens())
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=self.max_tokens())  # noqa: S106
        return tokenizer

    @cached_property
    def model(self) -> Any:
        """Create the MNN interpreter and CPU session."""
        interpreter = self.MNN.Interpreter(str(self._mnn_model_path))
        session = interpreter.createSession({"backend": "CPU", "thread": 1})
        if session is None:
            raise RuntimeError(f"Failed to create an MNN session for {self._mnn_model_path}")
        return interpreter, session

    @property
    def interpreter(self) -> Any:
        """Return the cached MNN interpreter."""
        return self.model[0]

    @property
    def session(self) -> Any:
        """Return the cached MNN session."""
        return self.model[1]

    def max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model."""
        return self.MAX_TOKENS

    def __call__(self, documents: Documents) -> Embeddings:
        """Generate embeddings for one document or a list of documents."""
        if isinstance(documents, str):
            documents = [documents]
        if not documents:
            return []

        self._download_model_if_not_exists()
        embeddings = self._forward(documents)
        return [embedding.tolist() for embedding in embeddings]

    def __repr__(self) -> str:
        return f"MnnEmbeddingFunction(model_name='{self.model_name}')"
