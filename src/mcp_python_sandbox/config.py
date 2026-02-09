"""Server configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Configuration for the MCP Python executor server."""

    python_version: str = "3.13"
    sandbox_backend: str = "native"  # "native" | "docker" | "none"
    max_timeout: int = 300
    default_timeout: int = 30
    max_output_bytes: int = 102_400  # 100KB
    warm_cache: bool = True
    uv_path: str = "uv"

    def __post_init__(self) -> None:
        valid_backends = ("native", "docker", "none")
        if self.sandbox_backend not in valid_backends:
            raise ValueError(
                f"Invalid sandbox_backend {self.sandbox_backend!r}, "
                f"must be one of {valid_backends}"
            )
        if self.max_timeout < 1:
            raise ValueError("max_timeout must be >= 1")
        if self.default_timeout < 1 or self.default_timeout > self.max_timeout:
            raise ValueError(
                f"default_timeout must be between 1 and {self.max_timeout}"
            )
        if self.max_output_bytes < 1024:
            raise ValueError("max_output_bytes must be >= 1024")
