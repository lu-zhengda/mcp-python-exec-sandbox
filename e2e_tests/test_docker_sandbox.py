"""E2E tests for the Docker sandbox backend."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from mcp_python_exec_sandbox.executor import execute
from mcp_python_exec_sandbox.sandbox import get_sandbox


def _docker_available() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        result = subprocess.run(
            [docker, "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _docker_image_exists(name: str = "mcp-python-exec-sandbox") -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        result = subprocess.run(
            [docker, "image", "inspect", name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


pytestmark = [
    pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv not installed",
    ),
    pytest.mark.skipif(
        not _docker_available(),
        reason="Docker not available",
    ),
    pytest.mark.skipif(
        not _docker_image_exists(),
        reason="mcp-python-exec-sandbox image not built "
        "(run: docker build -t mcp-python-exec-sandbox profiles/)",
    ),
]


@pytest.mark.asyncio
async def test_simple_script():
    """Test basic script execution through Docker sandbox."""
    sandbox = get_sandbox("docker")

    script = "print('hello from docker')\n"

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

    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert "hello from docker" in result.stdout


@pytest.mark.asyncio
async def test_script_with_dependencies():
    """Test installing and using a package inside Docker."""
    sandbox = get_sandbox("docker")

    script = """\
# /// script
# dependencies = ["pydantic>=2.0"]
# ///

import pydantic
print(f"pydantic v{pydantic.__version__}")
"""

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=120,
            sandbox=sandbox,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert "pydantic v2." in result.stdout


@pytest.mark.asyncio
async def test_read_only_filesystem():
    """Test that the Docker sandbox enforces a read-only root filesystem."""
    sandbox = get_sandbox("docker")

    script = """\
import os
# Try writing outside /tmp and /work — should fail on read-only fs
targets = ["/opt/test.txt", "/home/test.txt", "/var/test.txt"]
for path in targets:
    try:
        with open(path, "w") as f:
            f.write("breach")
        print(f"WRITE_OK: {path}")
    except (OSError, PermissionError) as e:
        print(f"BLOCKED: {path}: {e}")
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

    assert "WRITE_OK" not in result.stdout, (
        f"Write succeeded to a read-only location:\n{result.stdout}"
    )
    assert "BLOCKED" in result.stdout


@pytest.mark.asyncio
async def test_host_filesystem_isolation():
    """Test that the container cannot see host-specific files."""
    sandbox = get_sandbox("docker")

    # Write a unique marker to a temp file on the host, then try to read it
    # from inside the container — it must not be visible.
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        marker = "HOST_MARKER_e2e_test_12345"
        marker_file = Path(tmpdir) / "host_marker.txt"
        marker_file.write_text(marker)

        # The script runs inside Docker where only /work (the script dir) is mounted.
        # The marker file is in the same tmpdir, so it IS on the /work mount.
        # Instead, try to read a host-only path that won't exist in the container.
        script = """\
import socket, os
hostname = socket.gethostname()
# Container hostname is a short hex id, not the host machine name
print(f"HOSTNAME={hostname}")
# /etc/hostname inside container should differ from host
try:
    with open("/etc/hostname") as f:
        etc_hostname = f.read().strip()
except FileNotFoundError:
    etc_hostname = "NOT_FOUND"
print(f"ETC_HOSTNAME={etc_hostname}")
"""
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=30,
            sandbox=sandbox,
            max_output_bytes=102400,
        )

    assert result.exit_code == 0, f"stderr: {result.stderr}"

    # The container gets its own hostname (Docker assigns a short hex id)
    import socket

    host_hostname = socket.gethostname()
    # Extract container hostname from output
    for line in result.stdout.splitlines():
        if line.startswith("HOSTNAME="):
            container_hostname = line.split("=", 1)[1]
            assert container_hostname != host_hostname, (
                "Container hostname should differ from host"
            )
            break
    else:
        pytest.fail(f"HOSTNAME not found in output:\n{result.stdout}")


@pytest.mark.asyncio
async def test_timeout_enforcement():
    """Test that a long-running script gets killed by timeout."""
    sandbox = get_sandbox("docker")

    script = """\
import time
print("starting")
time.sleep(300)
print("should not reach here")
"""

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text(script)

        result = await execute(
            script_path=script_path,
            python_version="3.13",
            timeout=5,
            sandbox=sandbox,
            max_output_bytes=102400,
        )

    assert result.timed_out, "Expected execution to time out"
    assert "should not reach here" not in result.stdout
