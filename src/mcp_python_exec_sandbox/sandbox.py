"""Sandbox abstract base class and factory."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path


class Sandbox(ABC):
    """Abstract base class for execution sandboxes."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this sandbox backend is available on the current system."""
        ...

    @abstractmethod
    def wrap(self, cmd: list[str], script_path: Path) -> list[str]:
        """Wrap a command with sandbox isolation.

        Args:
            cmd: The original command to execute.
            script_path: Path to the script being executed.

        Returns:
            The wrapped command list.
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of the sandbox configuration."""
        ...


class NoopSandbox(Sandbox):
    """No-op sandbox that passes commands through unchanged."""

    def is_available(self) -> bool:
        return True

    def wrap(self, cmd: list[str], script_path: Path) -> list[str]:
        return cmd

    def describe(self) -> str:
        return "none (no sandboxing)"


def _get_docker_sandbox() -> Sandbox:
    """Create a Docker sandbox, falling back to NoopSandbox if unavailable."""
    from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

    sb = DockerSandbox()
    if not sb.is_available():
        import logging

        logging.warning("Docker not available, falling back to no sandbox")
        return NoopSandbox()
    return sb


def get_sandbox(backend: str) -> Sandbox:
    """Create a sandbox instance for the given backend.

    Falls back to NoopSandbox with a warning if the requested backend
    is not available.
    """
    if backend == "none":
        return NoopSandbox()

    if backend == "native":
        if sys.platform == "linux":
            from mcp_python_exec_sandbox.sandbox_linux import BubblewrapSandbox

            sb = BubblewrapSandbox()
            if not sb.is_available():
                import logging

                logging.warning("bwrap not found, falling back to no sandbox")
                return NoopSandbox()
            return sb

        # Native sandbox is only supported on Linux; use Docker on other platforms.
        import logging

        logging.info(
            "Native sandbox is Linux-only; using Docker sandbox on %s.",
            sys.platform,
        )
        return _get_docker_sandbox()

    if backend == "docker":
        return _get_docker_sandbox()

    raise ValueError(f"Unknown sandbox backend: {backend!r}")
