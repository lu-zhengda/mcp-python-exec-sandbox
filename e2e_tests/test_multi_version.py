"""E2E tests for multi-version Python execution."""

import shutil
import tempfile
from pathlib import Path

import pytest

from mcp_python_exec_sandbox.executor import execute

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)


@pytest.mark.parametrize("python_version", ["3.13", "3.14"])
@pytest.mark.asyncio
async def test_version_matches(python_version: str):
    """Test that the script runs on the requested Python version."""
    script = """\
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
"""

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version=python_version,
            timeout=60,
            sandbox=None,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() == python_version


@pytest.mark.parametrize("python_version", ["3.13", "3.14"])
@pytest.mark.asyncio
async def test_dependency_install_across_versions(python_version: str):
    """Test that PEP 723 dependencies work on each Python version."""
    script = """\
# /// script
# dependencies = ["pydantic>=2.0"]
# ///

import pydantic
print(f"pydantic {pydantic.__version__}")
"""

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version=python_version,
            timeout=120,
            sandbox=None,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert "pydantic 2." in result.stdout
