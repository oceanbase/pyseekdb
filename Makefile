UV ?= uv

.PHONY: install
install: ## Install the virtual environment and pre-commit hooks
	@echo ">> Installing dependencies"
	@$(UV) sync --all-groups
	@$(UV) run prek install

.PHONY: check
check: ## Run code quality tools
	@echo ">> Checking lock file consistency with 'pyproject.toml'"
	@$(UV) lock --locked
	@echo ">> Running pre-commit hooks"
	@$(UV) run prek run -a

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks via prek
	@echo ">> Running pre-commit hooks"
	@$(UV) run prek run -a

.PHONY: demo
demo: ## Run RAG demo (Streamlit)
	@echo ">> Running RAG demo"
	@$(UV) run --project demo/rag streamlit run demo/rag/seekdb_app.py

.PHONY: test
test: ## Run unit tests
	@echo ">> Running unit tests"
	@$(UV) run pytest tests/unit_tests/ -v --log-cli-level=INFO

.PHONY: test-integration-embedded
test-integration-embedded: ## Run embedded integration tests
	@echo ">> Running embedded integration tests"
	@$(UV) run pytest tests/integration_tests/ -v --log-cli-level=INFO -k embedded

.PHONY: docs
docs: ## Build documentation
	@echo ">> Building documentation"
	@$(UV) run sphinx-build -b html docs docs/_build/html

.PHONY: build
build: ## Build package
	@echo ">> Building package"
	@$(UV) build

.PHONY: clean
clean: ## Clean build and docs artifacts
	@echo ">> Removing build artifacts"
	@$(UV) run python -c "import shutil; shutil.rmtree('dist', ignore_errors=True); shutil.rmtree('docs/_build', ignore_errors=True); shutil.rmtree('tests/seekdb.db', ignore_errors=True)"

.PHONY: help
help:
	@awk -F '## ' '/^[A-Za-z0-9_-]+:.*##/ { target = $$1; sub(/:.*/, "", target); printf "\033[36m%-26s\033[0m %s\n", target, $$2 }' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
