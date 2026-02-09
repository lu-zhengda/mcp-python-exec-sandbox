"""E2E tests for sandbox enforcement."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from mcp_python_exec_sandbox.executor import execute
from mcp_python_exec_sandbox.sandbox import get_sandbox

pytestmark = [
    pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv not installed",
    ),
]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
@pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap not found",
)
@pytest.mark.asyncio
async def test_linux_sandbox_blocks_etc_shadow():
    """Test that Linux sandbox blocks reading /etc/shadow."""
    sandbox = get_sandbox("native")

    script = """\
try:
    with open("/etc/shadow") as f:
        print(f.read())
    print("ACCESS_GRANTED")
except (PermissionError, FileNotFoundError, OSError) as e:
    print(f"ACCESS_DENIED: {e}")
"""

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=30,
            sandbox=sandbox,
            max_output_bytes=102400,
        )

    assert "ACCESS_GRANTED" not in result.stdout
