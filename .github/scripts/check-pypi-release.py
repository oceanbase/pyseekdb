"""Validate pyseekdb distributions and decide whether the release needs an upload."""

import hashlib
import json
import os
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

PROJECT = "pyseekdb"


def distribution_files(version):
    wheels = list(Path("dist").glob("*.whl"))
    sources = list(Path("dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("Expected one wheel and one source distribution in dist/")
    wheel, source = wheels[0], sources[0]
    wheel_name, wheel_version, *_ = parse_wheel_filename(wheel.name)
    if (wheel_name, wheel_version) != (PROJECT, version) or parse_sdist_filename(source.name) != (PROJECT, version):
        raise SystemExit(f"Expected both distributions to be {PROJECT} {version}")
    return wheel, source


def published_digests(version):
    url = f"https://pypi.org/pypi/{PROJECT}/{quote(str(version), safe='')}/json"
    try:
        with urlopen(url, timeout=30) as response:
            return {item["filename"]: item["digests"]["sha256"] for item in json.load(response)["urls"]}
    except HTTPError as error:
        if error.code == 404:
            return {}
        raise


def needs_upload(files, version):
    existing = published_digests(version)
    if all(path.name in existing for path in files):
        print(f"PyPI already has both distributions for {PROJECT} {version}")
        return False

    for path in files:
        if path.name in existing:
            with path.open("rb") as package:
                digest = hashlib.file_digest(package, "sha256").hexdigest()
            if digest != existing[path.name]:
                raise SystemExit(f"PyPI already has different contents for {path.name}")
    return True


def main():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["name"] != PROJECT:
        raise SystemExit(f"Expected the project name {PROJECT}")
    version = Version(project["version"])
    tag = os.environ.get("RELEASE_TAG", "")
    publish = os.environ.get("PUBLISH_TO_PYPI") == "true"
    if publish and not tag:
        raise SystemExit("Publishing requires a release tag")
    if tag and Version(tag) != version:
        raise SystemExit(f"Release tag {tag} does not match pyproject.toml version {version}")

    files = distribution_files(version)
    upload = publish and needs_upload(files, version)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"upload={str(upload).lower()}\n")
    print(f"Validated {files[0].name} and {files[1].name}")


if __name__ == "__main__":
    main()
