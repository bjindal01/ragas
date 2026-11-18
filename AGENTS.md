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
tests/unit/                # fast, mocked tests; run in CI with a coverage floor
tests/e2e/                 # tests against real providers; not run in CI
tests/benchmarks/          # benchmark scripts (need API keys)
tests/conftest.py          # shared fixtures: fake_llm, fake_embedding, mock_llm, mock_embedding
examples/                  # separate workspace package `ragas-examples`
docs/                      # mkdocs site (docs.ragas.io)
```

`ragas.metrics` re-exports legacy metric classes with a deprecation warning; new metrics and fixes go in `ragas.metrics.collections`. The repo is a uv workspace with members `.` and `examples`.

## Prerequisites

- `make`, `git`, and `curl` or `wget`. **uv** is required for everything else. `make build` and the install targets install it automatically when it is missing (see below). To install it yourself, run `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Python >= 3.9.** CI tests 3.9, 3.12 and 3.13 on Ubuntu, 3.11 on macOS and 3.10 on Windows (coverage is measured on 3.12). When `.venv` does not exist yet, uv picks the interpreter: an already installed uv-managed Python is preferred over `python3` on PATH (`uv python list --only-installed` shows them). Pin it with `UV_PYTHON`, for example `UV_PYTHON=3.12 make build`.
- **Network:** `make type` (and so `make lint` and `make build`) downloads the latest pyright on each run.
- A full git clone (with tags) so setuptools_scm can compute the version.

## Commands

Run everything from the repo root. Use `uv run` for any Python tool (pytest, ruff, pyright); do not call `pip` or a global `python`.

| Task | Command | Notes |
|---|---|---|
| **One command: fresh clone to verified build** | `make build` (alias `make all`) | Runs `install-minimal`, `run-ci` and `package-smoke`, and installs uv first if it is missing. |
| Install uv only | `make ensure-uv` | Installs uv into `~/.local/bin` without editing shell profiles. Set `UV_INSTALL_DIR=...` to change the location, or `UV_AUTO_INSTALL=0` to fail with instructions instead. |
| Minimal dev env | `make install-minimal` | `uv sync --extra dev-minimal` from `uv.lock`: ~190 packages, ~1 GB, no torch. Enough for lint, type check and the whole unit suite. Also installs the pre-commit hook. |
| Full dev env | `make install` | `uv sync --group dev`: ~360 packages, ~7 GB including torch and CUDA. Needed for e2e tests, benchmarks and the ML-heavy integrations. |
| All static checks | `make lint` | ruff format check, ruff lint, pyright, and the pre-commit file checks (large files, merge markers, JSON/TOML/YAML, debugger calls, private keys). This is what CI runs. |
| Auto-format and fix | `make format` | Rewrites files (ruff format, then `ruff check --fix-only`). |
| Type check only | `make type` | pyright on `src`; must report 0 errors. |
| Unit tests | `RAGAS_DO_NOT_TRACK=true __RAGAS_DEBUG_TRACKING=true uv run pytest tests/unit -n auto --dist loadfile` | ~20 s. On Python 3.12: 1353 passed, 49 skipped, 1 xfailed. |
| One file / one test | `RAGAS_DO_NOT_TRACK=true uv run pytest tests/unit/test_cache.py` or `... -k test_name` | `make test k=test_name` also works. |
| Unit tests + coverage | `make test-cov` | Fails below `fail_under = 70` (`[tool.coverage.report]` in `pyproject.toml`); currently ~71% on 3.12. Writes `coverage.xml`. |
| Full local CI | `make run-ci` | `make lint`, then unit tests with `--nbmake` and `--cov` (the coverage floor applies). Run before opening a PR. |
| Package build smoke | `make package-smoke` | Builds sdist and wheel, runs `twine check`, installs the wheel in a clean venv, imports it and runs `ragas --help`. |
| Build only | `uv build` | Writes `dist/`. |
| E2E tests | `make install`, then `OPENAI_API_KEY=... make test-e2e` | See "Tests and API keys". |
| Docs | `make build-docs` / `make serve-docs` | Use the `docs` dependency group. |

`make help` lists every target. `make check` runs `format` and `type`, so it also rewrites files. Without uv on PATH, uv-based targets other than `build`, `ensure-uv` and the install targets stop with instructions.

## Dependencies and `uv.lock`

- Dependencies are declared only in `pyproject.toml`:
  - core packages in `[project].dependencies`
  - optional features in `[project.optional-dependencies]`
  - the minimal dev tools, plus the light optional packages the unit tests import, in the `dev-minimal` extra
  - the full stack in the `dev` dependency group
- `[tool.uv] default-groups = []`: `uv sync` and `uv run` install no dependency group unless you pass `--group`, so `uv run` never pulls the 7 GB `dev` group into a minimal env.
- `uv.lock` is committed and CI runs `uv lock --check`. After any dependency change in `pyproject.toml`, run `uv lock` and commit `uv.lock` in the same change. A stale lockfile fails CI, and `uv run` rewrites it silently, so check `git status` before committing.
- To move locked versions forward, run `uv lock --upgrade` or `uv lock --upgrade-package <name>`.
- Keep imports of optional dependencies lazy (inside functions or `try/except ImportError`), so that `import ragas` works with core dependencies only. `make package-smoke` checks this.

## Tests and API keys

- Unit tests must not call real LLM or embedding providers. Use the fixtures in `tests/conftest.py` (`EchoLLM`/`fake_llm`, `EchoEmbedding`/`fake_embedding`, `mock_llm`, `mock_embedding`) or `unittest.mock`.
- Coverage is enforced: CI runs the 3.12 Ubuntu job with `--cov`, and `make run-ci` does too. New code needs tests that keep total coverage at or above 70%.
- About 49 unit tests skip on Python 3.12, by design:
  - `dspy-ai`, `uvloop` and `datacompy` are not in `dev-minimal`.
  - A few tests are marked skip in the code: CSV backend limits, a flaky analytics test, and a TODO.
  - The `langchain`/`langsmith` integration tests need langchain 0.x, which only resolves on Python 3.9.
- Test environment variables, set by CI and `make run-ci`:
  - `RAGAS_DO_NOT_TRACK=true` disables the anonymous telemetry in `src/ragas/_analytics.py`. Set it for every local run.
  - `__RAGAS_DEBUG_TRACKING=true` logs telemetry payloads instead of sending them (developer debugging).
  - `RAGAS_CACHE_HOME` optionally overrides the cache directory.
- `tests/e2e` and `tests/benchmarks` need `OPENAI_API_KEY` and the full `dev` env (`make install`). With only `install-minimal`, e2e tests fail on missing packages such as `rouge_score`.
- E2E tests that need a key skip when it is unset, except `tests/e2e/metrics_migration/metric_score_diff.ipynb`, which fails. Without a key, `make test-e2e` reports 39 passed, 48 skipped and 1 failed.
- CI never runs e2e tests, so do not depend on them for validation. Never commit keys (`.env` is gitignored).
- `tests/docs/` is outside `tests/unit`, so CI does not run it.
- New tests go in `tests/unit/` and follow the `test_*.py` / `test_*` naming. Async tests use `@pytest.mark.asyncio`.

## Python 3.9 compatibility

The package supports Python >= 3.9 (`requires-python = ">=3.9"`, ruff `target-version = "py39"`, pyright `pythonVersion = "3.9"`, CI runs 3.9).

- Do not write `X | Y` or `X | None` in annotations evaluated at runtime (pydantic models, dataclasses, `typing.get_type_hints`). Use `typing.Optional` / `typing.Union` (the codebase mostly uses `import typing as t`).
- `from __future__ import annotations` only helps for annotations nothing evaluates at runtime. It does not help pydantic fields.
- No `match` statements, parenthesised context managers or other 3.10+ syntax. Avoid stdlib APIs added after 3.9.
- ruff's pyupgrade rules (`UP`) are deliberately off. Do not enable them or apply their fixes.
- To check on 3.9 locally, use a separate environment so `.venv` stays untouched (~30 s once packages are cached):
  ```bash
  RAGAS_DO_NOT_TRACK=true UV_PROJECT_ENVIRONMENT=/tmp/venv39 \
    uv run --python 3.9 --extra dev-minimal pytest tests/unit -n auto --dist loadfile
  ```

## Code style and lint rules

All config is in `pyproject.toml`. CI fails on any `make lint` finding.

- **ruff:**
  - Rules: `E`, `F`, `I`, `B`, `C4`, `C90`, `SIM`, `PIE`, `RUF`, `T10` and `PGH004`, minus the ignores listed and explained in `[tool.ruff.lint]`.
  - Formatting: line length 88 (enforced by the formatter, E501 ignored), double quotes. Imports are sorted with `known-first-party = ["ragas"]`.
- **pyright:**
  - Runs in `basic` mode with `reportMissingImports = "error"`.
  - Mark each import of an optional dependency that may not be installed with `# pyright: ignore[reportMissingImports]`, so that `make lint` passes in the minimal env. Any other unresolved import is a real error.
- **Bugbear rules:**
  - Re-raise with context: `raise NewError(...) from e`.
  - No mutable default arguments (`def f(x=[])`). Use `None` and create the value inside the function.
- **Complexity:** max McCabe complexity is 15. The few existing functions over the limit carry `# noqa: C901`. Refactor instead of adding new ones.
- **Suppressions:** no blanket `# noqa`. Always name the rule (`# noqa: B008`).
- **Docs and logging:** type hints on public APIs and Google-style docstrings. For diagnostics in library code, use module-level loggers (`logger = logging.getLogger(__name__)`).
- **Generated files:** do not commit `src/ragas/_version.py`, `examples/ragas_examples/_version.py`, `dist/`, `site/` or `coverage.xml`.
- **Scratch files and plans:** scratch files go in `_experiments/` and plans go in `plan/`. Both are gitignored.
- **Editor rules:** the rules in `.cursor/rules/` and `.claude/commands/` follow the same conventions.

## Commits and pull requests

- Conventional-commit style subjects, as in `git log`: `feat: ...`, `fix: ...`, `fix(llms): ...`, `docs: ...`, `test: ...`, `ci: ...`, `chore: ...`, `perf: ...`.
- Branch from `main`, keep each PR to one focused change, and include tests for behaviour changes.
- Run `make run-ci` before pushing. The pre-commit hook (installed by the install targets) runs the file checks and `make run-ci-format-check` on every commit. Fix the findings (`make format`) rather than skipping the hook.
- Fill in `.github/pull_request_template.md`: the problem or issue link, the changes made, and how they were tested.
- CI (`.github/workflows/ci.yaml`) runs:
  - unit tests across the OS and Python matrix, with coverage on the 3.12 Ubuntu job;
  - the code-quality job (`uv lock --check`, then `make lint`);
  - `package_smoke`.
- Changes that affect users should also update `docs/`.

## Gotchas

- Dependency changes need `uv lock` (see above). A CI failure on `uv lock --check` means `uv.lock` was not updated.
- Code that imports an optional package passes `make lint` in the full env and fails it in the minimal env unless the import carries `# pyright: ignore[reportMissingImports]`. CI lints in the minimal env.
- `make type` needs network access because `PYRIGHT_PYTHON_FORCE_VERSION=latest` downloads pyright. New pyright releases can surface new errors without a code change.
- `ragas.experimental` is a lazy alias for a separately installed `ragas_experimental` package. Experimental features such as `Dataset`, `experiment` and `ragas.backends` now live in the main package.
