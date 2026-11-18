# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

All agent instructions live in [AGENTS.md](./AGENTS.md). It covers:

- project layout and prerequisites (uv, Python >= 3.9);
- the verified build, lint, type-check and test commands;
- test environment variables and API-key requirements;
- dependency and `uv.lock` rules;
- Python 3.9 compatibility rules, code style and PR conventions.

Read it before making changes. Keep this file short and put new guidance in AGENTS.md, so the two files cannot drift apart.

Essentials:

- `make build` takes a fresh clone to a verified build. It installs uv if missing, then runs `install-minimal`, `run-ci` (lint, type check, unit tests with coverage) and `package-smoke`.
- Always use `uv run` for Python tools (pytest, ruff, pyright).
- Set `RAGAS_DO_NOT_TRACK=true` when running code or tests.
- Run `make run-ci` before finishing. After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`.
- Keep code Python 3.9 compatible. Do not use `X | None` in annotations that are evaluated at runtime.
- Put scratch files in `_experiments/` and saved plans in `plan/`. Both are gitignored.
