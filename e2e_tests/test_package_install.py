"""E2E tests for package installation via PEP 723."""

import shutil
import tempfile
from pathlib import Path

import pytest

from mcp_python_exec_sandbox.executor import execute
from mcp_python_exec_sandbox.script import build_script

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)


@pytest.mark.asyncio
async def test_install_and_use_requests():
    """Test installing requests via PEP 723 metadata."""
    script = build_script(
        """\
import requests
resp = requests.get("https://httpbin.org/get")
print(resp.status_code)
""",
        extra_dependencies=["requests"],
        python_version="3.13",
    )

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=120,
            sandbox=None,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0
    assert "200" in result.stdout


@pytest.mark.asyncio
async def test_inline_metadata_in_script():
    """Test that scripts with inline PEP 723 metadata work."""
    script = '''\
# /// script
# dependencies = ["rich"]
# requires-python = ">=3.11"
# ///

from rich.text import Text
t = Text("hello")
print(t.plain)
'''

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=120,
            sandbox=None,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0
    assert "hello" in result.stdout
