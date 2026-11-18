# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

All agent instructions live in [AGENTS.md](./AGENTS.md): project layout, prerequisites (uv, Python >= 3.9), the verified setup, lint, type-check, test and build commands, test environment variables, API-key requirements, Python 3.9 compatibility rules, code style and PR conventions. Read it before making changes. Keep this file short and put new guidance in AGENTS.md so the two files cannot drift apart.

Essentials:

- Install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then run `make install-minimal`.
- Always use `uv run` for Python tools (pytest, ruff, pyright).
- Set `RAGAS_DO_NOT_TRACK=true` when running code or tests.
- Run `make run-ci` (format check, ruff, pyright, unit tests) before finishing.
- Keep code Python 3.9 compatible. Do not use `X | None` in annotations that are evaluated at runtime.
- Put scratch files in `_experiments/` and saved plans in `plan/`. Both are gitignored.
