"""E2E tests — require uv + network."""

import asyncio
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
async def test_network_access():
    """Test that scripts can make network requests."""
    script = build_script(
        """\
import urllib.request
resp = urllib.request.urlopen("https://httpbin.org/get")
print(resp.status)
""",
        python_version="3.13",
    )

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=30,
            sandbox=None,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0
    assert "200" in result.stdout


@pytest.mark.asyncio
async def test_file_write_in_tempdir():
    """Test that scripts can write files within their temp directory."""
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        output_file = Path(tmpdir) / "output.txt"
        script_path.write_text(f"""\
with open("{output_file}", "w") as f:
    f.write("hello from script")
print("written")
""")

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=30,
            sandbox=None,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0
    assert "written" in result.stdout
