# Contributing

Thanks for your interest in contributing to pyseekdb. This file covers development and testing
instructions.

## Development

- Use Python 3.11+.
- This repo uses `uv`. You can override the binary via `UV=...`.

This project uses [uv](https://docs.astral.sh/uv/) as the package manager with
[pdm-backend](https://pdm-backend.fming.dev/) as the build backend. All common development
tasks are unified through the `Makefile`.

### Prerequisites

Install uv:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/oceanbase/pyseekdb.git
cd pyseekdb

# Install dependencies (all groups)
make install
```

## Make Targets

```bash
make help                      # Show all available targets
make install                   # Install dependencies
make test                      # Run unit tests
make test-integration-embedded # Run embedded integration tests
make docs                      # Build documentation
make demo                      # Run RAG demo
make build                     # Build the package
make clean                     # Clean build artifacts
```

## Build Artifacts

After running `make build`, the distribution files will be in the `dist/` directory:
- `pyseekdb-<version>.tar.gz` - Source distribution
- `pyseekdb-<version>-py3-none-any.whl` - Wheel distribution

## Publishing to PyPI

Set `project.version` in `pyproject.toml` on the commit you want to release.
Run **Publish pyseekdb to PyPI** manually in Actions to build and check the
distributions. Set `package_version` to choose the version for that run, or
leave it blank to use `project.version`. Push a matching tag such as `v1.4.1`
or publish its GitHub Release to upload the wheel and source distribution to PyPI.

Configure [Trusted Publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
in the PyPI `pyseekdb` project with owner `oceanbase`, repository `pyseekdb`,
workflow `python-publish.yml`, and a blank environment field.

## Testing

```bash
# Run unit tests
make test

# Run embedded integration tests
make test-integration-embedded

# Run specific tests with uv run
uv run pytest tests/integration_tests/ -v -k "server"     # server mode (requires seekdb server)
uv run pytest tests/integration_tests/ -v -k "oceanbase"  # oceanbase mode (requires OceanBase)

# Run specific test file
uv run pytest tests/integration_tests/test_collection_query.py -v

# Run specific test function
uv run pytest tests/integration_tests/test_collection_query.py::TestCollectionQuery::test_collection_query -v
```
