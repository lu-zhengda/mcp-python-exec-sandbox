"""Tests for MCP server tool definitions."""

from unittest.mock import patch, MagicMock

import pytest

from mcp_python_sandbox.config import ServerConfig
from mcp_python_sandbox.server import create_server


@pytest.fixture
def server():
    config = ServerConfig(sandbox_backend="none", warm_cache=False)
    return create_server(config)


class TestCreateServer:
    def test_creates_server(self, server):
        assert server is not None

    def test_uv_not_found(self):
        config = ServerConfig(sandbox_backend="none", warm_cache=False, uv_path="/nonexistent/uv")
        with pytest.raises(RuntimeError, match="uv not found"):
            create_server(config)
