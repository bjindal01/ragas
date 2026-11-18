GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths
RAGAS_PATHS := src tests docs

# uv is the only tool needed before the first `make build`. If it is missing, the
# uv-based install targets install it with the official installer into UV_INSTALL_DIR,
# without editing shell profiles. Set UV_AUTO_INSTALL=0 to fail with instructions instead.
UV_INSTALL_DIR ?= $(HOME)/.local/bin
UV_INSTALLER_URL ?= https://astral.sh/uv/install.sh
UV_AUTO_INSTALL ?= 1
# Targets that work before uv is installed. Any other target fails fast with
# instructions instead of "uv: No such file or directory" (Error 127).
UV_FREE_GOALS := help all build ensure-uv install-minimal install setup-venv clean benchmarks-docker check-mermaid-deps
ifeq ($(shell command -v uv 2>/dev/null),)
# PATH is only touched when uv is not on it, and the directory is appended so the
# separator stays right on Windows, where PATH uses ";".
export PATH := $(PATH)$(if $(findstring ;,$(PATH)),;,:)$(UV_INSTALL_DIR)
ifeq ($(wildcard $(UV_INSTALL_DIR)/uv $(UV_INSTALL_DIR)/uv.exe),)
ifneq ($(filter-out $(UV_FREE_GOALS),$(MAKECMDGOALS)),)
$(error uv is required but was not found. Run 'make build' to install it and set up the project, or install it yourself: curl -LsSf $(UV_INSTALLER_URL) | sh)
endif
endif
endif

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

.PHONY: all build ensure-uv

build: ## One command from a fresh clone: install uv if needed, install-minimal, run-ci, package-smoke
	$(Q)$(MAKE) --no-print-directory install-minimal
	$(Q)$(MAKE) --no-print-directory run-ci
	$(Q)$(MAKE) --no-print-directory package-smoke
	@echo "Build complete: environment installed, lint, type check and unit tests passed, package builds and imports."

all: build ## Alias for build

ensure-uv: ## Check for uv; install it to UV_INSTALL_DIR if missing (UV_AUTO_INSTALL=0 only checks)
	$(Q)if command -v uv >/dev/null 2>&1; then \
		echo "Using $$(uv --version) at $$(command -v uv)"; \
	elif [ "$(UV_AUTO_INSTALL)" = "0" ]; then \
		echo "error: uv is required but was not found on PATH." >&2; \
		echo "Install it with: curl -LsSf $(UV_INSTALLER_URL) | sh" >&2; \
		echo "Other methods: https://docs.astral.sh/uv/getting-started/installation/" >&2; \
		exit 1; \
	else \
		echo "uv not found; installing it to $(UV_INSTALL_DIR) from $(UV_INSTALLER_URL)"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -LsSf "$(UV_INSTALLER_URL)" | env UV_INSTALL_DIR="$(UV_INSTALL_DIR)" UV_NO_MODIFY_PATH=1 sh || exit 1; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -qO- "$(UV_INSTALLER_URL)" | env UV_INSTALL_DIR="$(UV_INSTALL_DIR)" UV_NO_MODIFY_PATH=1 sh || exit 1; \
		else \
			echo "error: uv is missing and neither curl nor wget is available to install it." >&2; \
			echo "Install uv manually: https://docs.astral.sh/uv/getting-started/installation/" >&2; \
			exit 1; \
		fi; \
		command -v uv >/dev/null 2>&1 || { echo "error: uv was installed but is not runnable from $(UV_INSTALL_DIR)" >&2; exit 1; }; \
		echo "Installed $$(uv --version). Make finds it; add $(UV_INSTALL_DIR) to PATH to run uv directly."; \
	fi

setup-venv: ## Set up uv virtual environment
	@echo "Setting up uv virtual environment..."
	$(Q)VIRTUAL_ENV= uv venv
	@echo "Virtual environment created at .venv"
	@echo "To activate: source .venv/bin/activate"

install-minimal: ensure-uv ## Install minimal dev dependencies from uv.lock (fast setup, no torch)
	@echo "Installing minimal development dependencies (fast setup)..."
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found, creating one..."; \
		$(MAKE) setup-venv; \
	fi
	@echo "Installing core ragas + essential dev tools (locked versions from uv.lock)..."
	$(Q)uv sync --active --extra dev-minimal --no-default-groups
	@echo "Setting up pre-commit hooks..."
	$(Q)uv run --active --no-sync pre-commit install
	@echo "Minimal installation complete!"
	@echo "Note: For full features including ML packages, use 'make install'"

install: ensure-uv ## Install full dependencies with uv sync (backward compatible - modern approach)
	@echo "Installing full development dependencies with uv sync..."
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found, creating one..."; \
		$(MAKE) setup-venv; \
	fi
	@echo "Installing ragas with full dev environment..."
	$(Q)VIRTUAL_ENV= uv sync --group dev
	@echo "Setting up pre-commit hooks..."
	$(Q)uv run pre-commit install
	@echo "Full installation complete! (Modern uv sync approach)"

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help setup-venv install-minimal install format type lint check clean test test-e2e benchmarks benchmarks-docker run-ci run-ci-fast run-ci-format-check run-ci-type run-ci-tests package-smoke build-docs serve-docs
format: ## Format and lint all code
	@echo "Formatting and linting all code..."
	@echo "(ruff format) Formatting ragas..."
	$(Q)uv run --active ruff format src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	@echo "(ruff) Auto-fixing ragas (includes import sorting and unused imports)..."
	$(Q)uv run --active ruff check src tests docs --exclude src/ragas/_version.py --fix-only --config pyproject.toml
	@echo "(ruff) Final linting check for ragas..."
	$(Q)uv run --active ruff check src tests docs --exclude src/ragas/_version.py --config pyproject.toml

type: ## Type check all code
	@echo "Type checking all code..."
	@echo "(pyright) Typechecking ragas..."
	$(Q)PYRIGHT_PYTHON_FORCE_VERSION=latest uv run --active pyright -p pyproject.toml src

lint: ## Run every static check: ruff format/lint, pyright, and pre-commit file checks (what CI runs)
	@echo "Running all lint checks..."
	@echo "(ruff) Format check..."
	$(Q)uv run --active ruff format --check $(RAGAS_PATHS) --exclude src/ragas/_version.py --config pyproject.toml
	@echo "(ruff) Lint check..."
	$(Q)uv run --active ruff check $(RAGAS_PATHS) --exclude src/ragas/_version.py --config pyproject.toml
	$(Q)$(MAKE) type
	@echo "(pre-commit) File checks..."
	$(Q)SKIP=monorepo-ci uv run --active pre-commit run --all-files
	@echo "All lint checks passed!"

check: format type ## Quick health check (format + type, no tests)
	@echo "Code quality check complete!"

# =============================================================================
# BENCHMARKS
# =============================================================================
benchmarks: ## Run all benchmarks locally
	@echo "Running all benchmarks..."
	@echo "Running evaluation benchmarks..."
	$(Q)cd $(GIT_ROOT)/tests/benchmarks && uv run python benchmark_eval.py
	@echo "Running testset generation benchmarks..."
	$(Q)cd $(GIT_ROOT)/tests/benchmarks && uv run python benchmark_testsetgen.py

benchmarks-docker: ## Run benchmarks in docker
	@echo "Running benchmarks in docker..."
	$(Q)cd $(GIT_ROOT) || exit 1
	docker buildx build --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) -t ragas-benchmark -f $(GIT_ROOT)/tests/benchmarks/Dockerfile .
	docker inspect ragas-benchmark:latest | jq ".[0].Size" | numfmt --to=si

benchmarks-test: ## Run benchmarks for ragas unit tests
	@echo "Running ragas unit tests with timing benchmarks..."
	$(Q)uv run --active pytest --nbmake tests/unit --durations=0 -v $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

# =============================================================================
# CI/BUILD
# =============================================================================

run-ci: ## Run complete CI pipeline (mirrors GitHub CI exactly)
	@echo "Running complete CI pipeline..."
	$(Q)$(MAKE) lint
	@echo "Unit tests..."
	$(Q)__RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true uv run --active pytest --nbmake tests/unit --dist loadfile -n auto
	@echo "All CI checks passed!"

run-ci-format-check: ## Run format check in dry-run mode (like GitHub CI)
	@echo "Running format check (dry-run, like GitHub CI)..."
	@echo "Checking ragas formatting..."
	$(Q)uv run --active ruff format --check src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	$(Q)uv run --active ruff check src docs tests --exclude src/ragas/_version.py --config pyproject.toml

run-ci-type: ## Run type checking (matches GitHub CI)
	@echo "Running type checking (matches GitHub CI)..."
	$(Q)$(MAKE) type

run-ci-tests: ## Run all tests with CI options
	@echo "Running all tests with CI options..."
	$(Q)__RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true uv run --active pytest --nbmake tests/unit --dist loadfile -n auto

run-ci-fast: ## Fast CI check for quick local validation (2-3 minutes)
	@echo "Running fast CI check for quick feedback..."
	@echo "Format check..."
	$(Q)uv run --active ruff format --check src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	$(Q)uv run --active ruff check src docs tests --exclude src/ragas/_version.py --config pyproject.toml
	@echo "Core unit tests (no nbmake for speed)..."
	$(Q)uv run --active pytest tests/unit --dist loadfile -n auto -x
	@echo "Fast CI check completed!"

package-smoke: ## Build the package, install the wheel in a clean venv and import it (what a user gets)
	@echo "Building sdist and wheel, then installing the wheel with core dependencies only..."
	$(Q)SMOKE_DIR=$$(mktemp -d) && \
		uv build --out-dir "$$SMOKE_DIR/dist" && \
		uvx twine check "$$SMOKE_DIR"/dist/* && \
		uv venv --quiet "$$SMOKE_DIR/venv" && \
		uv pip install --quiet --python "$$SMOKE_DIR/venv/bin/python" "$$SMOKE_DIR"/dist/*.whl && \
		cd "$$SMOKE_DIR" && \
		RAGAS_DO_NOT_TRACK=true ./venv/bin/python -c "import ragas; print('ragas', ragas.__version__, 'imports')" && \
		RAGAS_DO_NOT_TRACK=true ./venv/bin/ragas --help > /dev/null && \
		uv pip check --python ./venv/bin/python
	@echo "Package smoke test passed!"

clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	$(Q)rm -rf site/ docs/site/ .mypy_cache .pytest_cache .ruff_cache
	$(Q)rm -rf dist/ build/ *.egg-info/ src/*.egg-info/
	$(Q)rm -rf .coverage htmlcov/ .tox/ .venv/
	$(Q)find . -name '*.log' -delete
	$(Q)find . -name '.DS_Store' -delete
	$(Q)find . -name 'temp*' -type d -exec rm -rf {} + 2>/dev/null || true
	$(Q)find . -name '.tmp*' -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all unit tests
	@echo "Running all unit tests..."
	$(Q)uv run --active pytest tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

test-all: ## Run all unit tests (including notebooks)
	@echo "Running all unit tests (including notebooks)..."
	$(Q)uv run --active pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

test-e2e: ## Run all end-to-end tests
	@echo "Running all end-to-end tests..."
	$(Q)uv run --active pytest --nbmake tests/e2e -s

# =============================================================================
# DOCUMENTATION
# =============================================================================

build-docs: ## Build all documentation
	@echo "Building all documentation..."
	@echo "Converting ipynb notebooks to md files..."
	$(Q)MKDOCS_CI=true uv run python $(GIT_ROOT)/docs/ipynb_to_md.py
	@echo "Building ragas documentation..."
	$(Q)MKDOCS_CI=false uv run --group docs mkdocs build


check-pdf-deps: ## Check if WeasyPrint is properly installed with all dependencies
	@echo "Checking if WeasyPrint is properly installed..."
	@uv run --group docs-pdf python -c "import weasyprint; weasyprint.HTML(string='<h1>Test</h1>').write_pdf(target=None)" 2>/dev/null && \
		echo "WeasyPrint is installed and all dependencies are available" || \
		(echo ""; \
		echo "WeasyPrint is not installed or has missing system dependencies"; \
		echo ""; \
		echo "Setup Instructions: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"; \
		echo "Troubleshooting: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting"; \
		echo ""; \
		exit 1)


check-mermaid-deps: ## Check Mermaid CLI is available and can render a diagram
	@command -v node >/dev/null || (echo "Node.js is required for Mermaid PDF rendering"; exit 1)
	@command -v mmdc >/dev/null || (echo "Missing 'mmdc' (Mermaid CLI). Mermaid diagrams in PDF depend on Mermaid CLI."; exit 1)
	@tmp_dir="$$(mktemp -d)"; \
	printf "graph TD\n  A-->B\n" > "$$tmp_dir/diag.mmd"; \
	mmdc -i "$$tmp_dir/diag.mmd" -o "$$tmp_dir/diag.svg" >/dev/null 2>&1 || \
		(echo "Mermaid CLI found, but rendering failed (mmdc couldn't produce SVG)."; rm -rf "$$tmp_dir"; exit 1); \
	test -s "$$tmp_dir/diag.svg" || \
		(echo "Mermaid CLI ran but produced an empty SVG."; rm -rf "$$tmp_dir"; exit 1); \
	rm -rf "$$tmp_dir"; \
	echo "Mermaid CLI is installed and can render diagrams"

	
build-docs-pdf: check-pdf-deps check-mermaid-deps ## Build documentation with PDF export (requires WeasyPrint)
	@echo "Building documentation with PDF export..."
	$(Q)MKDOCS_CI=false ENABLE_PDF_EXPORT=1 uv run --group docs --group docs-pdf mkdocs build -f mkdocs-pdf.yml
	@echo "PDF generated at: site/pdf/document.pdf"



serve-docs: ## Build and serve documentation locally
	$(Q)MKDOCS_CI=false uv run --group docs mkdocs serve --dirtyreload

