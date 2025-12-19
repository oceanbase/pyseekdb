"""Sphinx configuration file for pyseekdb documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "pyseekdb"
copyright = "2025, OceanBase"
author = "OceanBase <open_oceanbase@oceanbase.com>"

try:
    import importlib.metadata
    release = importlib.metadata.version("pyseekdb")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.1.dev1"

version = release

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# Enable Markdown in docstrings
myst_enable_extensions = ["colon_fence"]
myst_all_links_external = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

autosummary_generate = True
autosummary_imported_members = True
autodoc_mock_imports = ["pylibseekdb", "pymysql", "onnxruntime", "tokenizers", "httpx", "tqdm", "tenacity", "numpy"]

# Source files
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
