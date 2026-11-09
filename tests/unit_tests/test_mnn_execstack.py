"""Tests for clearing MNN's executable GNU stack flag."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import struct
from pathlib import Path

import pytest

from pyseekdb.utils.embedding_functions import mnn_embedding_function
from pyseekdb.utils.embedding_functions.mnn_embedding_function import (
    _PT_GNU_STACK,
    _clear_executable_gnu_stack,
    _gnu_stack_is_executable,
    _mnn_native_library_paths,
    _prepare_mnn_native_libraries,
)

_PT_LOAD = 1


def _build_elf(elf_class: int, *, stack_flags: int, load_flags: int = 5, little_endian: bool = True) -> bytes:
    """Build a tiny ELF whose program headers are enough to edit GNU_STACK."""
    endian = "<" if little_endian else ">"
    ei_data = 1 if little_endian else 2
    if elf_class == 2:
        phentsize = 56
        flags_offset = 4
        header = bytearray(64)
        header[0:4] = b"\x7fELF"
        header[4] = 2
        header[5] = ei_data
        header[6] = 1
        struct.pack_into(endian + "HHI", header, 16, 3, 62, 1)
        struct.pack_into(endian + "Q", header, 32, 64)
        struct.pack_into(endian + "IHHHHHH", header, 48, 0, 64, phentsize, 2, 0, 0, 0)
    elif elf_class == 1:
        phentsize = 32
        flags_offset = 24
        header = bytearray(52)
        header[0:4] = b"\x7fELF"
        header[4] = 1
        header[5] = ei_data
        header[6] = 1
        struct.pack_into(endian + "HHI", header, 16, 3, 3, 1)
        struct.pack_into(endian + "I", header, 28, 52)
        struct.pack_into(endian + "HHHHHH", header, 40, 52, phentsize, 2, 0, 0, 0)
    else:
        raise ValueError(elf_class)

    def program_header(p_type: int, p_flags: int) -> bytes:
        entry = bytearray(phentsize)
        struct.pack_into(endian + "I", entry, 0, p_type)
        struct.pack_into(endian + "I", entry, flags_offset, p_flags)
        return bytes(entry)

    return bytes(header) + program_header(_PT_LOAD, load_flags) + program_header(_PT_GNU_STACK, stack_flags)


def _flags_at(path: Path, elf_class: int, index: int, *, little_endian: bool = True) -> int:
    endian = "<" if little_endian else ">"
    data = path.read_bytes()
    if elf_class == 2:
        phoff, phentsize, flags_offset = 64, 56, 4
    else:
        phoff, phentsize, flags_offset = 52, 32, 24
    offset = phoff + index * phentsize + flags_offset
    return struct.unpack_from(endian + "I", data, offset)[0]


@pytest.mark.parametrize(
    ("elf_class", "little_endian"),
    [(2, True), (2, False), (1, True), (1, False)],
)
def test_clear_executable_gnu_stack(tmp_path: Path, elf_class: int, little_endian: bool) -> None:
    path = tmp_path / "library.so"
    path.write_bytes(_build_elf(elf_class, stack_flags=7, load_flags=5, little_endian=little_endian))

    assert _gnu_stack_is_executable(path)
    assert _clear_executable_gnu_stack(path)
    assert not _gnu_stack_is_executable(path)
    assert not _clear_executable_gnu_stack(path)
    assert _flags_at(path, elf_class, 0, little_endian=little_endian) == 5
    assert _flags_at(path, elf_class, 1, little_endian=little_endian) == 6


def test_non_elf_file_is_left_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"not an elf")

    assert not _gnu_stack_is_executable(path)
    assert not _clear_executable_gnu_stack(path)
    assert path.read_bytes() == b"not an elf"


def test_missing_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "missing.so"

    assert not _gnu_stack_is_executable(path)
    assert not _clear_executable_gnu_stack(path)


class _DistributionFile:
    def __init__(self, root: Path, relative: str):
        self._path = root / relative

    def locate(self) -> Path:
        return self._path


class _Distribution:
    def __init__(self, root: Path, relatives: list[str]):
        self._root = root
        self.files = [_DistributionFile(root, relative) for relative in relatives]

    def locate_file(self, relative: str) -> Path:
        return self._root / relative


def test_unrelated_tools_library_is_not_rewritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    owned = site / "_tools.cpython-311-x86_64-linux-gnu.so"
    unrelated = site / "_tools_other.cpython-311-x86_64-linux-gnu.so"
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside = outside_root / "_tools.cpython-311-x86_64-linux-gnu.so"
    owned.write_bytes(_build_elf(2, stack_flags=7))
    unrelated_bytes = _build_elf(2, stack_flags=7)
    outside_bytes = _build_elf(2, stack_flags=7)
    unrelated.write_bytes(unrelated_bytes)
    outside.write_bytes(outside_bytes)
    distribution = _Distribution(
        site,
        [
            "_tools.cpython-311-x86_64-linux-gnu.so",
            "../outside/_tools.cpython-311-x86_64-linux-gnu.so",
            "MNN/__init__.py",
        ],
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(mnn_embedding_function, "_mnn_native_libraries_prepared", False)

    paths = _mnn_native_library_paths()
    assert paths == [owned]

    _prepare_mnn_native_libraries()

    assert not _gnu_stack_is_executable(owned)
    assert unrelated.read_bytes() == unrelated_bytes
    assert outside.read_bytes() == outside_bytes


@pytest.mark.skipif(importlib.util.find_spec("MNN") is None, reason="MNN is not available")
def test_installed_mnn_libraries_load_without_executable_stack() -> None:
    mnn_embedding_function._mnn_native_libraries_prepared = False
    paths = _mnn_native_library_paths()
    names = {path.name for path in paths}

    assert any(name.startswith("_mnncengine") for name in names)
    assert any(name.startswith("_tools") for name in names)

    _prepare_mnn_native_libraries()

    for path in paths:
        assert not _gnu_stack_is_executable(path), path

    import MNN

    tensor = MNN.Tensor([1, 4], MNN.Halide_Type_Int, MNN.Tensor_DimensionType_Tensorflow)
    assert tensor is not None
