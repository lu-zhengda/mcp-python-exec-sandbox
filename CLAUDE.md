# CLAUDE.md

## Project

MCP server for sandboxed Python execution. Scripts run in ephemeral, isolated environments with inline dependencies (PEP 723) via `uv`. Zero host pollution.

## Stack

- Python 3.13+, no runtime deps beyond `fastmcp` and `tomli-w`
- `uv` for script execution, dependency resolution, and Python version management
- `hatchling` build backend, `src/` layout

## Structure

```
src/mcp_python_exec_sandbox/   # Package source
  server.py               # FastMCP server + tool definitions
  executor.py             # uv subprocess orchestration
  script.py               # PEP 723 metadata parsing/merging
  sandbox.py              # Sandbox ABC + factory
  sandbox_linux.py        # bubblewrap sandbox (Linux)
  sandbox_docker.py       # Docker sandbox (macOS/any)
  config.py, cache.py, output.py, errors.py
tests/                    # Unit + integration tests (mocked or local uv)
e2e_tests/                # End-to-end tests (require uv + network)
profiles/                 # Dockerfile, warmup packages
```

## Commands

```bash
uv sync --dev             # Install deps
uv run pytest tests/ -v   # Unit + integration tests
uv run pytest e2e_tests/ -v  # E2E tests (slow, needs network)
```

## Rules

- Run `uv run pytest tests/ -v` before committing. All tests must pass.
- Keep dependencies minimal. Do not add runtime deps without strong justification.
- Lint with `uv run ruff check .` and format with `uv run ruff format --check .` before committing. Fix issues with `--fix` / `ruff format .`.
- Tool docstrings in `server.py` are user-facing — they become the MCP tool descriptions that agents see. Write them for an LLM audience: include examples, avoid unexplained jargon, link PEPs.
- Always pin versions in examples (e.g. `"pandas>=2.2"` not `"pandas"`).
- Sandbox backends must degrade gracefully: if the tool (bwrap, docker) is missing, fall back to `NoopSandbox` with a warning. Native sandbox is Linux-only (bwrap); macOS defaults to Docker.

## Contribution format

- One logical change per commit. Descriptive commit message (imperative mood).
- Keep PRs focused. Separate refactors from feature work.
- Add tests for new functionality: unit tests in `tests/`, E2E in `e2e_tests/` if it needs real execution.
- Update tool docstrings in `server.py` if changing tool behavior.
- Update `README.md` if changing user-facing config or adding features.
