"""Tests for server configuration."""

import pytest

from mcp_python_exec_sandbox.__main__ import parse_args
from mcp_python_exec_sandbox.config import ServerConfig


class TestServerConfig:
    def test_defaults(self):
        config = ServerConfig()
        assert config.python_version == "3.13"
        assert config.sandbox_backend == "native"
        assert config.max_timeout == 300
        assert config.default_timeout == 30
        assert config.max_output_bytes == 102_400
        assert config.warm_cache is True
        assert config.uv_path == "uv"

    def test_invalid_sandbox_backend(self):
        with pytest.raises(ValueError, match="Invalid sandbox_backend"):
            ServerConfig(sandbox_backend="invalid")

    def test_invalid_max_timeout(self):
        with pytest.raises(ValueError, match="max_timeout must be >= 1"):
            ServerConfig(max_timeout=0)

    def test_invalid_default_timeout(self):
        with pytest.raises(ValueError, match="default_timeout must be between"):
            ServerConfig(default_timeout=0)

    def test_default_timeout_exceeds_max(self):
        with pytest.raises(ValueError, match="default_timeout must be between"):
            ServerConfig(max_timeout=10, default_timeout=20)

    def test_invalid_max_output_bytes(self):
        with pytest.raises(ValueError, match="max_output_bytes must be >= 1024"):
            ServerConfig(max_output_bytes=100)

    def test_custom_values(self):
        config = ServerConfig(
            python_version="3.14",
            sandbox_backend="docker",
            max_timeout=600,
            default_timeout=60,
            max_output_bytes=200_000,
            warm_cache=False,
        )
        assert config.python_version == "3.14"
        assert config.sandbox_backend == "docker"
        assert config.max_timeout == 600
        assert config.default_timeout == 60
        assert config.max_output_bytes == 200_000
        assert config.warm_cache is False


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.python_version == "3.13"
        assert args.sandbox_backend == "native"
        assert args.max_timeout == 300
        assert args.default_timeout == 30
        assert args.max_output_bytes == 102_400
        assert args.no_warm_cache is False

    def test_custom_args(self):
        args = parse_args(
            [
                "--python-version",
                "3.14",
                "--sandbox-backend",
                "docker",
                "--max-timeout",
                "600",
                "--default-timeout",
                "60",
                "--max-output-bytes",
                "200000",
                "--no-warm-cache",
            ]
        )
        assert args.python_version == "3.14"
        assert args.sandbox_backend == "docker"
        assert args.max_timeout == 600
        assert args.default_timeout == 60
        assert args.max_output_bytes == 200_000
        assert args.no_warm_cache is True
