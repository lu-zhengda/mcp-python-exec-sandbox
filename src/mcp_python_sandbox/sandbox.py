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


def get_sandbox(backend: str) -> Sandbox:
    """Create a sandbox instance for the given backend.

    Falls back to NoopSandbox with a warning if the requested backend
    is not available.
    """
    if backend == "none":
        return NoopSandbox()

    if backend == "native":
        if sys.platform == "linux":
            from mcp_python_sandbox.sandbox_linux import BubblewrapSandbox

            sb = BubblewrapSandbox()
        elif sys.platform == "darwin":
            from mcp_python_sandbox.sandbox_macos import SandboxExecSandbox

            sb = SandboxExecSandbox()
        else:
            import logging

            logging.warning(
                "Native sandbox not supported on %s, falling back to no sandbox",
                sys.platform,
            )
            return NoopSandbox()

        if not sb.is_available():
            import logging

            logging.warning(
                "Native sandbox tool not found, falling back to no sandbox"
            )
            return NoopSandbox()
        return sb

    if backend == "docker":
        from mcp_python_sandbox.sandbox_docker import DockerSandbox

        sb = DockerSandbox()
        if not sb.is_available():
            import logging

            logging.warning(
                "Docker not available, falling back to no sandbox"
            )
            return NoopSandbox()
        return sb

    raise ValueError(f"Unknown sandbox backend: {backend!r}")
