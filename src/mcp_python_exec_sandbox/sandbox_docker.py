"""Docker-based sandbox implementation."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path

from mcp_python_exec_sandbox.sandbox import Sandbox

_IMAGE_NAME = "ghcr.io/lu-zhengda/mcp-python-exec-sandbox"

logger = logging.getLogger(__name__)


class DockerSandbox(Sandbox):
    """Cross-platform sandbox using Docker containers."""

    def __init__(self) -> None:
        self._docker_path = shutil.which("docker")
        self._pull_thread: threading.Thread | None = None
        self._pull_success: bool | None = None

    def _image_exists(self) -> bool:
        """Check if the Docker image is available locally."""
        docker = self._docker_path or "docker"
        try:
            result = subprocess.run(
                [docker, "image", "inspect", _IMAGE_NAME],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _pull_image(self) -> bool:
        """Pull the Docker image from GHCR. Returns True on success."""
        docker = self._docker_path or "docker"
        logger.info("Pulling Docker image %s ...", _IMAGE_NAME)
        try:
            result = subprocess.run(
                [docker, "pull", _IMAGE_NAME],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("Successfully pulled %s", _IMAGE_NAME)
                return True
            logger.warning(
                "Failed to pull %s: %s",
                _IMAGE_NAME,
                result.stderr.decode(errors="replace").strip(),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Timed out pulling %s", _IMAGE_NAME)
            return False
        except OSError as exc:
            logger.warning("Error pulling %s: %s", _IMAGE_NAME, exc)
            return False

    def _pull_image_background(self) -> None:
        """Pull the image and store the result."""
        self._pull_success = self._pull_image()

    def _ensure_image(self) -> bool:
        """Wait for a background pull to finish, or check that the image exists."""
        if self._pull_thread is not None:
            self._pull_thread.join(timeout=120)
            if self._pull_thread.is_alive():
                logger.warning("Background pull still running, proceeding without image")
                return False
            self._pull_thread = None
            return bool(self._pull_success)
        return self._image_exists()

    def is_available(self) -> bool:
        if self._docker_path is None:
            return False
        try:
            result = subprocess.run(
                [self._docker_path, "info"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                return False
        except (subprocess.TimeoutExpired, OSError):
            return False
        if not self._image_exists():
            self._pull_thread = threading.Thread(
                target=self._pull_image_background, daemon=True
            )
            self._pull_thread.start()
        return True

    def wrap(self, cmd: list[str], script_path: Path) -> list[str]:
        # Block until background pull (if any) finishes
        self._ensure_image()

        script_dir = str(script_path.parent)
        docker = self._docker_path or "docker"

        # Rewrite host script path to container path (/work/<filename>)
        container_script = f"/work/{script_path.name}"
        cmd = [container_script if arg == str(script_path) else arg for arg in cmd]

        return [
            docker,
            "run",
            "--rm",
            "--network=bridge",
            "--read-only",
            "--memory=512m",
            "--cpus=1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=100m",
            "-v",
            "mcp-python-exec-sandbox-uv-cache:/cache",
            "-v",
            f"{script_dir}:/work",
            "-e",
            "UV_CACHE_DIR=/cache",
            "-e",
            "UV_PYTHON_INSTALL_DIR=/cache/python",
            "-w",
            "/work",
            _IMAGE_NAME,
            *cmd,
        ]

    def describe(self) -> str:
        if self._docker_path and self.is_available():
            return f"docker ({self._docker_path})"
        return "docker (not available)"
