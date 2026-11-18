# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. This is the single source of truth for agent instructions; `CLAUDE.md` points here. Human-oriented detail lives in `CONTRIBUTING.md`.

## Project overview

Ragas is a Python library for evaluating LLM applications: metrics, test data generation, experiments and integrations with LLM frameworks. It is published to PyPI as `ragas` (version comes from git tags via setuptools_scm).

```
src/ragas/                 # the library (single package; there is no src/ragas/experimental)
  metrics/                 # legacy metrics (_*.py) + base classes
  metrics/collections/     # current metric API; one sub-package per metric
  llms/, embeddings/       # LLM and embedding wrappers and factories
  testset/                 # test data generation (graph, synthesizers, transforms)
  backends/                # dataset/experiment storage, registered via entry points
  integrations/            # LangChain, LlamaIndex, Langfuse, MLflow, AG-UI, Bedrock, ...
  prompt/, optimizers/     # prompt objects and prompt optimisation
  evaluation.py, cli.py    # evaluate()/aevaluate() and the `ragas` CLI
  _analytics.py            # anonymous usage telemetry
tests/unit/                # fast, mocked tests; run in CI
tests/e2e/                 # tests against real providers; not run in CI
tests/benchmarks/          # benchmark scripts (need API keys)
tests/conftest.py          # shared fixtures: fake_llm, fake_embedding, mock_llm, mock_embedding
examples/                  # separate workspace package `ragas-examples`
docs/                      # mkdocs site (docs.ragas.io)
```

`ragas.metrics` re-exports legacy metric classes with a deprecation warning; new metrics and fixes go in `ragas.metrics.collections`. The repo is a uv workspace with members `.` and `examples`.

## Prerequisites

- **uv** is required. Every Make target calls `uv`; without it `make` fails with `uv: No such file or directory`. Install it with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ```
- **Python >= 3.9.** `make install-minimal` creates `.venv` with the default `python3` on PATH. CI tests 3.9, 3.12 and 3.13 on Ubuntu, 3.11 on macOS and 3.10 on Windows.
- **Disk and network:** the dev environment is about 7 GB (torch and CUDA wheels) and downloads about 1.5 GB. `make type` downloads the latest pyright on each run.
- A full git clone (with tags) so setuptools_scm can compute the version.

## Commands

Run everything from the repo root. Use `uv run` for any Python tool (pytest, ruff, pyright); do not call `pip` or a global `python`.

| Task | Command | Notes |
|---|---|---|
| Install dev env | `make install-minimal` | ~2 min. Creates `.venv` and installs the pre-commit hook. Despite the "79 packages" message, it ends with the full `dev` group (~360 packages), because `uv run pre-commit install` syncs the default group. |
| Install (explicit full) | `make install` | `uv sync --group dev` plus the pre-commit hook. |
| Lint + format check | `make run-ci-format-check` | `ruff format --check` and `ruff check` on `src tests docs`. Read-only. |
| Auto-format and fix | `make format` | Rewrites files (ruff format, then `ruff check --fix-only`). |
| Type check | `make type` | pyright on `src`; ~70 s; must report 0 errors. |
| Unit tests | `RAGAS_DO_NOT_TRACK=true __RAGAS_DEBUG_TRACKING=true uv run pytest tests/unit -n auto --dist loadfile` | ~40 s; 963 passed, 89 skipped on main. |
| One file / one test | `RAGAS_DO_NOT_TRACK=true uv run pytest tests/unit/test_cache.py` or `... -k test_name` | `make test k=test_name` also works. |
| Full local CI | `make run-ci` | Format check, ruff, pyright, then unit tests with `--nbmake` (includes `tests/unit/test_executor_in_jupyter.ipynb`). ~2 min. Run before opening a PR. |
| Package build smoke | `make package-smoke` | Builds sdist and wheel, runs `twine check`, installs the wheel in a clean venv, imports it and runs `ragas --help`. |
| Build only | `uv build` | Writes `dist/`. |
| E2E tests | `OPENAI_API_KEY=... make test-e2e` | See "Tests and API keys". |
| Docs | `make build-docs` / `make serve-docs` | Use the `docs` dependency group. |

`make help` lists every target. `make check` runs `format` and `type`, so it also rewrites files. `make run-ci-tests` calls bare `pytest` and only works with `.venv` activated; use `make run-ci` instead.

## Tests and API keys

- Unit tests must not call real LLM or embedding providers. Use the fixtures in `tests/conftest.py` (`EchoLLM`/`fake_llm`, `EchoEmbedding`/`fake_embedding`, `mock_llm`, `mock_embedding`) or `unittest.mock`.
- Unit tests need the `dev` dependency group, not just `.[dev-minimal]`. With only `dev-minimal` installed, tests fail on missing `sacrebleu`, `rapidfuzz`, `GitPython` and `llama_index`.
- About 89 unit tests skip by design when optional packages (`dspy-ai`, `uvloop`, `ag-ui-protocol`) are not installed.
- Test environment variables, set by CI and `make run-ci`:
  - `RAGAS_DO_NOT_TRACK=true` disables the anonymous telemetry in `src/ragas/_analytics.py`. Set it for every local run.
  - `__RAGAS_DEBUG_TRACKING=true` logs telemetry payloads instead of sending them (developer debugging).
  - `RAGAS_CACHE_HOME` optionally overrides the cache directory.
- `tests/e2e` and `tests/benchmarks` need `OPENAI_API_KEY`. Tests that need a key skip when it is unset: on main, without a key, `tests/e2e` gives 39 passed and 48 skipped. CI never runs e2e tests, so do not depend on them for validation, and never commit keys (`.env` is gitignored).
- `tests/test_quoted_spans.py` and `tests/docs/` are outside `tests/unit`, so CI does not run them.
- New tests go in `tests/unit/` and follow the `test_*.py` / `test_*` naming. Async tests use `@pytest.mark.asyncio`.

## Python 3.9 compatibility

The package supports Python >= 3.9 (`requires-python = ">=3.9"`, ruff `target-version = "py39"`, pyright `pythonVersion = "3.9"`, CI runs 3.9).

- Do not write `X | Y` or `X | None` in annotations evaluated at runtime (pydantic models, dataclasses, `typing.get_type_hints`). Use `typing.Optional` / `typing.Union` (the codebase mostly uses `import typing as t`).
- `from __future__ import annotations` only helps for annotations nothing evaluates at runtime. It does not help pydantic fields.
- No `match` statements, parenthesised context managers or other 3.10+ syntax. Avoid stdlib APIs added after 3.9.
- Do not enable ruff pyupgrade autofixes (UP006, UP007, UP045) that rewrite to 3.10+ syntax.
- To check on 3.9 locally, use a separate environment so `.venv` stays untouched. The first run takes about 2 minutes to build the environment:
  ```bash
  RAGAS_DO_NOT_TRACK=true UV_PROJECT_ENVIRONMENT=/tmp/venv39 \
    uv run --python 3.9 --group dev pytest tests/unit -n auto --dist loadfile
  ```

## Code style and conventions

- ruff (rules `E`, `F`, `I`; line length 88, enforced by the formatter; double quotes) and pyright in `basic` mode. All config is in `pyproject.toml`. CI fails on any format, lint or type error.
- Imports are sorted by ruff (`known-first-party = ["ragas"]`). Keep imports of optional dependencies lazy (inside functions or `try/except ImportError`) so `import ragas` works with core dependencies only. `make package-smoke` checks this.
- Type hints on public APIs; Google-style docstrings.
- Use module-level loggers (`logger = logging.getLogger(__name__)`) for diagnostics in library code.
- Dependencies are declared only in `pyproject.toml`: core in `[project].dependencies`, optional features in `[project.optional-dependencies]`, dev tooling in `dev-minimal` and the `dev` dependency group. `uv.lock` is gitignored; do not commit it.
- Do not commit generated files: `src/ragas/_version.py`, `examples/ragas_examples/_version.py`, `dist/`, `site/`.
- Scratch files go in `_experiments/` (gitignored). Plans go in `plan/` (gitignored).
- Editor-specific rules in `.cursor/rules/` and `.claude/commands/` follow the same conventions.

## Commits and pull requests

- Conventional-commit style subjects, as in `git log`: `feat: ...`, `fix: ...`, `fix(llms): ...`, `docs: ...`, `test: ...`, `ci: ...`, `chore: ...`, `perf: ...`.
- Branch from `main`, keep each PR to one focused change, and include tests for behaviour changes.
- Run `make run-ci` before pushing. The pre-commit hook runs `make run-ci-format-check` on every commit; fix the findings (`make format`) rather than skipping the hook.
- Fill in `.github/pull_request_template.md`: the problem or issue link, the changes made, and how they were tested.
- CI (`.github/workflows/ci.yaml`) runs unit tests across the OS and Python matrix, the code-quality job (format, ruff, pyright) and `package_smoke`. Changes that affect users should also update `docs/`.

## Gotchas

- `make install-minimal` is not minimal (see above), and CI uses it too. Budget about 10 GB of free disk.
- `uv.lock` is not committed, so dependency versions float. A failure that appears without a code change is often an upstream release; the pins with comments in the `dev` group in `pyproject.toml` record past cases.
- `make type` needs network access because `PYRIGHT_PYTHON_FORCE_VERSION=latest` downloads pyright.
- `ragas.experimental` is a lazy alias for a separately installed `ragas_experimental` package. Experimental features such as `Dataset`, `experiment` and `ragas.backends` now live in the main package.
- The CONTRIBUTING note that "CI supports Python 3.9–3.12" is out of date. CI also tests 3.13 on Ubuntu. Its macOS ARM NumPy workaround for 3.13 may still apply.
