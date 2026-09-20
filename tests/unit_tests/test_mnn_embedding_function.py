"""Unit tests for the MNN embedding function."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from pyseekdb.utils.embedding_functions.mnn_embedding_function import MnnEmbeddingFunction


def is_mnn_available() -> bool:
    """Check whether the native MNN package is installed."""
    return importlib.util.find_spec("MNN") is not None


pytestmark = pytest.mark.skipif(not is_mnn_available(), reason="MNN is not available")


def _make_embedding_function(tmp_path: Path, dimension: int = 3) -> MnnEmbeddingFunction:
    return MnnEmbeddingFunction(
        model_name="test-model",
        hf_model_id="org/test-model",
        dimension=dimension,
        download_path=tmp_path,
    )


def test_init_validates_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        MnnEmbeddingFunction("", "org/test", 3, tmp_path)
    with pytest.raises(ValueError, match="hf_model_id must be a non-empty string"):
        MnnEmbeddingFunction("test", "", 3, tmp_path)
    with pytest.raises(ValueError, match="dimension must be a positive integer"):
        MnnEmbeddingFunction("test", "org/test", 0, tmp_path)


def test_empty_input_does_not_download_or_initialize_model(tmp_path: Path) -> None:
    ef = _make_embedding_function(tmp_path)
    assert ef([]) == []
    assert not (tmp_path / "mnn").exists()


def test_first_use_downloads_then_converts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ef = _make_embedding_function(tmp_path)
    calls: list[str] = []

    def fake_download() -> bool:
        calls.append("download")
        ef._model_folder.mkdir(parents=True, exist_ok=True)
        for filename in ef._MODEL_FILES:
            (ef._model_folder / filename).touch()
        return True

    def fake_convert() -> None:
        if ef._mnn_model_path.exists():
            return
        calls.append("convert")
        ef._mnn_model_path.touch()

    monkeypatch.setattr(ef, "_download_from_huggingface", fake_download)
    monkeypatch.setattr(ef, "_convert_onnx_to_mnn", fake_convert)

    ef._download_model_if_not_exists()
    ef._download_model_if_not_exists()

    assert calls == ["download", "convert"]
    assert ef._mnn_model_path.exists()


def test_forward_uses_mnn_inputs_and_mean_pools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ef = _make_embedding_function(tmp_path)
    monkeypatch.setattr(ef, "_download_model_if_not_exists", lambda: None)

    class FakeEncoding:
        ids: ClassVar[list[int]] = [1, 2]
        attention_mask: ClassVar[list[int]] = [1, 1]

    class FakeTokenizer:
        def encode(self, _document: str) -> FakeEncoding:
            return FakeEncoding()

    class FakeInput:
        def __init__(self) -> None:
            self.data = None

        def copyFrom(self, tensor: object) -> bool:
            self.data = tensor.getNumpyData()  # type: ignore[attr-defined]
            return True

    class FakeOutput:
        def getNumpyData(self) -> np.ndarray:
            return np.ones((1, 2, 3), dtype=np.float32)

    class FakeInterpreter:
        def __init__(self) -> None:
            self.inputs = {name: FakeInput() for name in ("input_ids", "attention_mask", "token_type_ids")}

        def getSessionInputAll(self, _session: object) -> dict[str, FakeInput]:
            return self.inputs

        def runSession(self, _session: object) -> int:
            return 0

        def getSessionOutputAll(self, _session: object) -> dict[str, FakeOutput]:
            return {"output": FakeOutput()}

    ef.__dict__["tokenizer"] = FakeTokenizer()
    ef.__dict__["model"] = (FakeInterpreter(), SimpleNamespace())

    embeddings = ef("hello")

    assert embeddings == [[1.0, 1.0, 1.0]]
    assert ef.interpreter.inputs["input_ids"].data.dtype == np.int32
    assert ef.interpreter.inputs["attention_mask"].data.dtype == np.int32
