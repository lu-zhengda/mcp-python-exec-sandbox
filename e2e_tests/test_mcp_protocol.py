"""E2E tests for the full MCP protocol flow — require uv + network."""

import json
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)

_SERVER_CMD = [
    sys.executable, "-m", "mcp_python_sandbox",
    "--sandbox-backend", "none",
    "--no-warm-cache",
]


class MCPClient:
    """Minimal MCP JSON-RPC client over stdio for testing."""

    def __init__(self):
        self.proc = subprocess.Popen(
            _SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def send(self, method: str, params: dict | None = None, *, notify: bool = False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self._next_id()
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        return self._recv()

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, "Server closed stdout unexpectedly"
        return json.loads(line)

    def initialize(self):
        result = self.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        })
        self.send("notifications/initialized", {}, notify=True)
        return result

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.send("tools/call", {"name": name, "arguments": arguments})

    def list_tools(self) -> dict:
        return self.send("tools/list", {})

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


@pytest.fixture
def mcp():
    client = MCPClient()
    client.initialize()
    yield client
    client.close()


class TestMCPProtocol:
    def test_initialize(self):
        """Test MCP initialize handshake."""
        client = MCPClient()
        result = client.initialize()
        assert "result" in result
        info = result["result"]
        assert info["protocolVersion"] == "2024-11-05"
        assert "tools" in info["capabilities"]
        client.close()

    def test_list_tools(self, mcp):
        """Test tools/list returns all three tools."""
        result = mcp.list_tools()
        tools = result["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"execute_python", "check_environment", "validate_script"}

    def test_tool_schemas(self, mcp):
        """Test that tool schemas have proper input definitions."""
        result = mcp.list_tools()
        tools = {t["name"]: t for t in result["result"]["tools"]}

        # execute_python should have script, dependencies, timeout_seconds
        ep = tools["execute_python"]["inputSchema"]
        assert "script" in ep["properties"]
        assert "dependencies" in ep["properties"]
        assert "timeout_seconds" in ep["properties"]

        # validate_script should have script and dependencies
        vs = tools["validate_script"]["inputSchema"]
        assert "script" in vs["properties"]
        assert "dependencies" in vs["properties"]

    def test_check_environment(self, mcp):
        """Test check_environment returns env info."""
        result = mcp.call_tool("check_environment", {})
        text = result["result"]["content"][0]["text"]
        assert "Python version:" in text
        assert "uv:" in text
        assert "Sandbox backend: none" in text

    def test_execute_simple_script(self, mcp):
        """Test executing a simple print script."""
        result = mcp.call_tool("execute_python", {
            "script": "print('hello from mcp')",
        })
        text = result["result"]["content"][0]["text"]
        assert "hello from mcp" in text
        assert "exit_code: 0" in text

    def test_execute_with_deps(self, mcp):
        """Test executing a script with dependencies."""
        result = mcp.call_tool("execute_python", {
            "script": "import pydantic; print(f'v{pydantic.__version__}')",
            "dependencies": ["pydantic>=2.0"],
            "timeout_seconds": 120,
        })
        text = result["result"]["content"][0]["text"]
        assert "exit_code: 0" in text
        assert "v2." in text

    def test_execute_pandas(self, mcp):
        """Test executing a pandas script via MCP."""
        result = mcp.call_tool("execute_python", {
            "script": (
                "import pandas as pd; "
                "df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}); "
                "print(df.sum().to_dict())"
            ),
            "dependencies": ["pandas"],
            "timeout_seconds": 120,
        })
        text = result["result"]["content"][0]["text"]
        assert "exit_code: 0" in text
        assert "'a': 6" in text
        assert "'b': 15" in text

    def test_execute_numpy_scipy(self, mcp):
        """Test numpy + scipy through MCP."""
        result = mcp.call_tool("execute_python", {
            "script": (
                "import numpy as np; from scipy import stats; "
                "np.random.seed(42); "
                "r, p = stats.pearsonr(np.random.randn(100), np.random.randn(100)); "
                "print(f'r={r:.4f}')"
            ),
            "dependencies": ["numpy", "scipy"],
            "timeout_seconds": 120,
        })
        text = result["result"]["content"][0]["text"]
        assert "exit_code: 0" in text
        assert "r=" in text

    def test_execute_with_pep723_inline(self, mcp):
        """Test a script with inline PEP 723 metadata block."""
        script = (
            '# /// script\n'
            '# dependencies = ["rich"]\n'
            '# ///\n'
            '\n'
            'from rich.text import Text\n'
            'print(Text("hello").plain)\n'
        )
        result = mcp.call_tool("execute_python", {"script": script, "timeout_seconds": 120})
        text = result["result"]["content"][0]["text"]
        assert "exit_code: 0" in text
        assert "hello" in text

    def test_execute_timeout(self, mcp):
        """Test that timeout is enforced."""
        result = mcp.call_tool("execute_python", {
            "script": "import time; time.sleep(60)",
            "timeout_seconds": 2,
        })
        text = result["result"]["content"][0]["text"]
        assert "timed_out: true" in text

    def test_execute_nonzero_exit(self, mcp):
        """Test script that exits with non-zero code."""
        result = mcp.call_tool("execute_python", {
            "script": "import sys; print('bye'); sys.exit(1)",
        })
        text = result["result"]["content"][0]["text"]
        assert "exit_code: 1" in text
        assert "bye" in text

    def test_validate_script_valid(self, mcp):
        """Test validate_script with valid deps."""
        result = mcp.call_tool("validate_script", {
            "script": "import pandas",
            "dependencies": ["pandas>=2.0", "numpy"],
        })
        text = result["result"]["content"][0]["text"]
        assert "VALID" in text
        assert "pandas>=2.0" in text
        assert "numpy" in text

    def test_validate_script_no_deps(self, mcp):
        """Test validate_script with a bare script."""
        result = mcp.call_tool("validate_script", {
            "script": "print('hello')",
        })
        text = result["result"]["content"][0]["text"]
        assert "VALID" in text
        assert "dependencies: none" in text

    def test_validate_script_inline_metadata(self, mcp):
        """Test validate_script with inline PEP 723 metadata."""
        script = (
            '# /// script\n'
            '# dependencies = ["requests"]\n'
            '# requires-python = ">=3.11"\n'
            '# ///\n'
            '\n'
            'import requests\n'
        )
        result = mcp.call_tool("validate_script", {"script": script})
        text = result["result"]["content"][0]["text"]
        assert "VALID" in text
        assert "requests" in text
        assert ">=3.11" in text
