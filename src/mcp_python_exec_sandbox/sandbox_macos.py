"""macOS sandbox-exec (Seatbelt) sandbox implementation."""

from __future__ import annotations

import shutil
from pathlib import Path

from mcp_python_exec_sandbox.sandbox import Sandbox

_PROFILE_DIR = Path(__file__).parent.parent.parent / "profiles"


class SandboxExecSandbox(Sandbox):
    """macOS sandbox using sandbox-exec with Seatbelt profiles."""

    def __init__(self) -> None:
        self._sandbox_exec_path = shutil.which("sandbox-exec")
        self._uv_cache_dir = Path.home() / "Library" / "Caches" / "uv"
        self._profile_path = _PROFILE_DIR / "sandbox_macos.sb"

    def is_available(self) -> bool:
        return self._sandbox_exec_path is not None

    def _build_profile(self, script_dir: str) -> str:
        """Build a Seatbelt profile string with the given script directory."""
        cache_dir = str(self._uv_cache_dir)

        # If the profile file exists, read and substitute parameters
        if self._profile_path.exists():
            template = self._profile_path.read_text()
            return template.replace("{{SCRIPT_DIR}}", script_dir).replace(
                "{{CACHE_DIR}}", cache_dir
            )

        # Fallback inline profile
        return f"""\
(version 1)
(deny default)

;; Allow reading system libraries and frameworks
(allow file-read*
    (subpath "/usr/lib")
    (subpath "/usr/local")
    (subpath "/System")
    (subpath "/Library/Frameworks")
    (subpath "/opt/homebrew")
    (subpath "/private/tmp")
    (subpath "/private/var/folders")
    (subpath "/dev")
    (subpath "{cache_dir}")
    (subpath "{script_dir}"))

;; Allow writing to cache and script dir
(allow file-write*
    (subpath "{cache_dir}")
    (subpath "{script_dir}")
    (subpath "/private/tmp")
    (subpath "/private/var/folders"))

;; Allow process operations
(allow process-exec*)
(allow process-fork)

;; Allow network access
(allow network*)

;; Allow sysctl reads (needed by Python)
(allow sysctl-read)

;; Allow mach lookups (needed for DNS, etc.)
(allow mach-lookup)

;; Allow ipc operations
(allow ipc-posix-shm-read-data)
(allow ipc-posix-shm-write-data)
"""

    def wrap(self, cmd: list[str], script_path: Path) -> list[str]:
        script_dir = str(script_path.parent)
        profile = self._build_profile(script_dir)

        return [
            self._sandbox_exec_path or "sandbox-exec",
            "-p",
            profile,
            *cmd,
        ]

    def describe(self) -> str:
        if self._sandbox_exec_path:
            return f"sandbox-exec ({self._sandbox_exec_path})"
        return "sandbox-exec (not found)"
