# mcp-python-exec-sandbox

Sandboxed Python execution for AI agents. Scripts run in ephemeral, isolated environments with inline dependencies ([PEP 723](https://peps.python.org/pep-0723/)) -- **zero host pollution, zero leftover venvs, zero package conflicts**.

## Why?

Every coding agent can already run Python on your host. The problem is what happens next: packages accumulate, venvs sprawl, and a rogue `pip install` breaks your system. **mcp-python-exec-sandbox** eliminates this:

- Scripts execute in a sandbox (bubblewrap on Linux, sandbox-exec on macOS, Docker everywhere)
- Dependencies are declared inline and resolved ephemerally via `uv`
- Nothing touches your host's Python, site-packages, or virtualenvs
- Each execution is isolated and disposable

## Features

- **Sandboxed execution** -- platform-specific isolation prevents host filesystem access
- **PEP 723 inline metadata** -- declare dependencies directly in scripts with `# /// script` blocks
- **Multi-version Python** -- run scripts on Python 3.13, 3.14, or 3.15 (uv downloads the right version automatically)
- **Ephemeral environments** -- dependencies are resolved per-execution, never persisted
- **Package caching** -- uv's global cache makes repeat installs near-instant
- **Timeout enforcement** -- configurable per-execution timeouts
- **Output truncation** -- prevents runaway output from overwhelming the agent

## Prerequisites

All setups require:

- **Python 3.13+** -- to run the MCP server process
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** -- manages script execution, dependency resolution, and Python version downloads. Also provides `uvx` for running the server without installing it globally.

Additional requirements depend on your chosen sandbox backend:

| Setup | Additional requirements | Install |
|-------|------------------------|---------|
| **Native sandbox (Linux)** | [bubblewrap](https://github.com/containers/bubblewrap) | `sudo apt install bubblewrap` |
| **Native sandbox (macOS)** | None -- `sandbox-exec` is built into macOS | -- |
| **Docker sandbox** | [Docker Engine](https://docs.docker.com/engine/install/) | See Docker docs |
| **No sandbox** | None | -- |

> **Host Python vs. execution Python:** These are independent. Python 3.13+ is needed to run the server process itself. The `--python-version` flag controls which Python version your *scripts* execute on -- uv downloads the target version automatically. You do not need to install Python 3.14 or 3.15 on your host to run scripts on those versions.

## Quick start

### Claude Code (native sandbox -- recommended)

```bash
claude mcp add python-sandbox -- uvx mcp-python-exec-sandbox --sandbox-backend native
```

### Claude Code (Docker sandbox)

```bash
docker build -t mcp-python-exec-sandbox profiles/
claude mcp add python-sandbox -- uvx mcp-python-exec-sandbox --sandbox-backend docker
```

> The Docker image build requires the repo source. Clone it first: `git clone https://github.com/lu-zhengda/mcp-python-exec-sandbox.git`

### Claude Code (no sandbox)

```bash
claude mcp add python-sandbox -- uvx mcp-python-exec-sandbox --sandbox-backend none
```

### Cursor

Add to `.cursor/mcp.json` (project-level) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "python-sandbox": {
      "command": "uvx",
      "args": ["mcp-python-exec-sandbox", "--sandbox-backend", "native"]
    }
  }
}
```

### OpenAI Codex CLI

```bash
codex mcp add python-sandbox -- uvx mcp-python-exec-sandbox --sandbox-backend native
```

Or add to `.codex/config.toml`:

```toml
[mcp_servers.python-sandbox]
command = "uvx"
args = ["mcp-python-exec-sandbox", "--sandbox-backend", "native"]
```

### Other MCP clients

Any client that supports the MCP stdio transport can use this server:

```json
{
  "mcpServers": {
    "python-sandbox": {
      "command": "uvx",
      "args": ["mcp-python-exec-sandbox", "--sandbox-backend", "native"]
    }
  }
}
```

## Multi-version Python

Use `--python-version` to target a specific Python version. uv downloads it automatically -- no manual install needed.

```bash
# Python 3.13 (default)
uvx mcp-python-exec-sandbox --python-version 3.13

# Python 3.14
uvx mcp-python-exec-sandbox --python-version 3.14

# Python 3.15
uvx mcp-python-exec-sandbox --python-version 3.15
```

This works across all sandbox backends. The Docker sandbox uses uv inside the container to manage Python versions, so the same `--python-version` flag applies.

## Tools

### `execute_python`

Execute a Python script with automatic dependency management.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `script` | str | required | Python source code, may include PEP 723 inline metadata |
| `dependencies` | list[str] | `[]` | Extra PEP 508 dependency specifiers to merge |
| `timeout_seconds` | int | 30 | Maximum execution time (1--300) |

```python
# Simple script
execute_python(script="print('hello world')")

# Script with dependencies
execute_python(
    script="import requests; print(requests.get('https://httpbin.org/get').status_code)",
    dependencies=["requests"]
)

# Script with inline PEP 723 metadata
execute_python(script="""
# /// script
# dependencies = ["pandas", "matplotlib"]
# ///

import pandas as pd
print(pd.DataFrame({'a': [1,2,3]}).describe())
""")
```

### `check_environment`

Returns information about the execution environment: Python version, uv version, platform, sandbox status, and configuration.

### `validate_script`

Validates a script's PEP 723 metadata and dependencies without executing it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `script` | str | required | Python source code to validate |
| `dependencies` | list[str] | `[]` | Extra dependency specifiers to validate |

## Sandbox backends

| Backend | Platform | Tool | Notes |
|---------|----------|------|-------|
| `native` | Linux | bubblewrap | Namespace isolation, network allowed |
| `native` | macOS | sandbox-exec | Seatbelt profiles, network allowed |
| `docker` | Any | Docker | Container isolation, resource limits |
| `none` | Any | -- | No sandboxing (not recommended) |

If the requested sandbox tool is unavailable, the server falls back to `none` with a warning.

### Docker sandbox setup

```bash
docker build -t mcp-python-exec-sandbox profiles/
```

## CLI options

```
mcp-python-exec-sandbox [OPTIONS]

Options:
  --python-version TEXT     Python version for execution (default: 3.13)
  --sandbox-backend TEXT    native | docker | none (default: native)
  --max-timeout INT         Maximum allowed timeout in seconds (default: 300)
  --default-timeout INT     Default timeout in seconds (default: 30)
  --max-output-bytes INT    Maximum output size in bytes (default: 102400)
  --no-warm-cache           Skip cache warming on startup
  --uv-path TEXT            Path to uv binary (default: uv)
```

## Development

### Setup

```bash
git clone https://github.com/lu-zhengda/mcp-python-exec-sandbox.git
cd mcp-python-exec-sandbox
uv sync --dev
```

### Project structure

```
src/mcp_python_exec_sandbox/   # Package source
  server.py               # FastMCP server + tool definitions
  executor.py             # uv subprocess orchestration
  script.py               # PEP 723 metadata parsing/merging
  sandbox.py              # Sandbox ABC + factory
  sandbox_{linux,macos,docker}.py
  config.py, cache.py, output.py, errors.py
tests/                    # Unit + integration tests (mocked or local uv)
e2e_tests/                # End-to-end tests (require uv + network)
profiles/                 # Dockerfile, macOS seatbelt profile, warmup packages
.devcontainer/            # Devcontainer for Linux sandbox testing from macOS
```

### Running tests

**Unit and integration tests** -- fast, run everywhere:

```bash
uv run pytest tests/ -v
```

**E2E tests** -- require `uv` and network access. These exercise real script execution, package installation, MCP protocol flow, and sandbox enforcement:

```bash
uv run pytest e2e_tests/ -v
```

### Docker sandbox tests

The Docker E2E tests (`e2e_tests/test_docker_sandbox.py`) verify execution, dependency installation, read-only filesystem enforcement, host isolation, and timeout handling through the Docker backend.

Prerequisites:

1. Docker must be installed and running
2. Build the sandbox image:

```bash
docker build -t mcp-python-exec-sandbox profiles/
```

Then run:

```bash
uv run pytest e2e_tests/test_docker_sandbox.py -v
```

These tests are automatically skipped if Docker is unavailable or the image hasn't been built.

### Linux sandbox tests (devcontainer)

The Linux sandbox tests (`e2e_tests/test_sandbox_enforcement.py::test_linux_sandbox_blocks_etc_shadow`) use bubblewrap (`bwrap`) for namespace isolation. They are skipped on macOS because `bwrap` is Linux-only.

To run them from macOS, use the included devcontainer which provides Ubuntu 24.04 with `bwrap` pre-installed:

**VS Code:**

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
2. Open the project and select **Reopen in Container**
3. In the integrated terminal:

```bash
uv run pytest e2e_tests/test_sandbox_enforcement.py -v
```

**CLI:**

```bash
# Install the devcontainer CLI (once)
npm install -g @devcontainers/cli

# Build and start the container
devcontainer up --workspace-folder .

# Run the Linux sandbox tests inside the container
devcontainer exec --workspace-folder . uv run pytest e2e_tests/test_sandbox_enforcement.py -v
```

### Test matrix

| Test suite | Command | Requirements |
|------------|---------|-------------|
| Unit tests | `uv run pytest tests/ -v` | `uv` |
| Integration tests | `uv run pytest tests/test_integration.py -v` | `uv` |
| E2E (general) | `uv run pytest e2e_tests/ -v` | `uv`, network |
| E2E (Docker sandbox) | `uv run pytest e2e_tests/test_docker_sandbox.py -v` | `uv`, Docker, sandbox image |
| E2E (Linux/bwrap sandbox) | `uv run pytest e2e_tests/test_sandbox_enforcement.py -v` | `uv`, Linux with `bwrap` (or devcontainer) |

### Contributing

- One logical change per commit. Descriptive commit message (imperative mood).
- Run `uv run pytest tests/ -v` before committing -- all tests must pass.
- Add tests for new functionality: unit tests in `tests/`, E2E in `e2e_tests/` if it needs real execution.
- Keep dependencies minimal. Do not add runtime deps without strong justification.
- Tool docstrings in `server.py` are user-facing MCP tool descriptions. Write them for an LLM audience.
- Sandbox backends must degrade gracefully: if the tool is missing, fall back to `NoopSandbox` with a warning.

## License

MIT
