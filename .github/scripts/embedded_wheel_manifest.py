"""Bind a CI-local wheel to pinned sources and verify the installed payload."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_payload(wheel):
    with zipfile.ZipFile(wheel) as archive, archive.open("pylibseekdb/seekdb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(pin, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source"] != pin:
        raise ValueError("cached wheel source identity mismatch")
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 11):
        raise ValueError("wheel requires Linux x86_64 CPython 3.11")
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1 or wheels[0].name != manifest["wheel"]:
        raise ValueError("expected exactly the manifest wheel")
    if digest(wheels[0]) != manifest["wheel_sha256"]:
        raise ValueError("wheel checksum mismatch")
    if wheel_payload(wheels[0]) != manifest["binary_sha256"]:
        raise ValueError("wheel contains a different SeekDB binary")
    version = (directory / "seekdb-version.txt").read_text()
    if pin["seekdb_sha"] not in version or version != manifest["revision_output"]:
        raise ValueError("SeekDB full REVISION mismatch")
    return manifest


def main():
    mode, pin_path, directory_name, *rest = sys.argv[1:]
    pin = json.loads(Path(pin_path).read_text())
    directory = Path(directory_name)
    if mode == "create":
        (binary_name,) = rest
        (wheel,) = directory.glob("*.whl")
        manifest = {
            "source": pin,
            "wheel": wheel.name,
            "wheel_sha256": digest(wheel),
            "binary_sha256": digest(Path(binary_name)),
            "revision_output": (directory / "seekdb-version.txt").read_text(),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    elif mode not in {"verify", "installed"}:
        raise ValueError(f"unknown verification mode: {mode}")
    manifest = verify(pin, directory)
    if mode == "installed":
        binary = Path(importlib.metadata.distribution("pylibseekdb").locate_file("pylibseekdb/seekdb"))
        if digest(binary) != manifest["binary_sha256"]:
            raise ValueError("installed SeekDB differs from source-built wheel (dependency resync?)")
        # The executable is the verified local wheel payload, never shell input.
        version = subprocess.check_output(  # noqa: S603
            [str(binary), "-V"], text=True, stderr=subprocess.STDOUT
        )
        if pin["seekdb_sha"] not in version:
            raise ValueError("installed SeekDB does not report pinned full REVISION")
        print(f"[EMBEDDED_WHEEL] installed={binary}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
