"""Sphinx configuration file for pyseekdb documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "pyseekdb"
copyright = "2025, OceanBase"  # noqa: A001
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
    "sphinx_multiversion",
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
autodoc_mock_imports = [
    "pylibseekdb",
    "pymysql",
    "onnxruntime",
    "tokenizers",
    "httpx",
    "tqdm",
    "tenacity",
    "numpy",
]

# Source files
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Sphinx-multiversion configuration
# 配置要包含的分支和标签
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"  # 匹配 v1.0.0 格式的标签
smv_branch_whitelist = r"^(main|develop)$"  # 包含 main 和 develop 分支
smv_remote_whitelist = r"^origin$"  # 只使用 origin 远程仓库
smv_released_pattern = r"^refs/tags/.*$"  # 标记已发布的版本

# 自定义模板路径
templates_path = ["_templates"]

# 版本横幅配置
html_context = {
    "display_github": True,
    "github_user": "oceanbase",
    "github_repo": "pyseekdb",
    "github_version": "develop",
    "conf_py_path": "/docs/",
}
