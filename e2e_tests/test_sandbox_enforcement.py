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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="sandbox-exec not found",
)
@pytest.mark.asyncio
async def test_macos_sandbox_blocks_home_read():
    """Test that macOS sandbox blocks reading files outside allowed dirs."""
    sandbox = get_sandbox("native")

    script = """\
import os
try:
    with open(os.path.expanduser("~/.ssh/id_rsa")) as f:
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

    # Should not have been able to read the file
    assert "ACCESS_GRANTED" not in result.stdout


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
