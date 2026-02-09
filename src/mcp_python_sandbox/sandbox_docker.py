"""Docker-based sandbox implementation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from mcp_python_sandbox.sandbox import Sandbox

_IMAGE_NAME = "mcp-python-sandbox"


class DockerSandbox(Sandbox):
    """Cross-platform sandbox using Docker containers."""

    def __init__(self) -> None:
        self._docker_path = shutil.which("docker")

    def is_available(self) -> bool:
        if self._docker_path is None:
            return False
        try:
            result = subprocess.run(
                [self._docker_path, "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def wrap(self, cmd: list[str], script_path: Path) -> list[str]:
        script_dir = str(script_path.parent)
        docker = self._docker_path or "docker"

        # Rewrite host script path to container path (/work/<filename>)
        container_script = f"/work/{script_path.name}"
        cmd = [container_script if arg == str(script_path) else arg for arg in cmd]

        return [
            docker, "run", "--rm",
            "--network=bridge",
            "--read-only",
            "--memory=512m",
            "--cpus=1",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=100m",
            "-v", "mcp-python-sandbox-uv-cache:/cache",
            "-v", f"{script_dir}:/work",
            "-e", "UV_CACHE_DIR=/cache",
            "-e", "UV_PYTHON_INSTALL_DIR=/cache/python",
            "-w", "/work",
            _IMAGE_NAME,
            *cmd,
        ]

    def describe(self) -> str:
        if self._docker_path and self.is_available():
            return f"docker ({self._docker_path})"
        return "docker (not available)"
