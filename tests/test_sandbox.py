"""Tests for sandbox layer."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_python_exec_sandbox.sandbox import NoopSandbox, get_sandbox


class TestNoopSandbox:
    def test_is_available(self):
        sb = NoopSandbox()
        assert sb.is_available() is True

    def test_wrap_passthrough(self):
        sb = NoopSandbox()
        cmd = ["uv", "run", "--script", "test.py"]
        assert sb.wrap(cmd, Path("/tmp/test.py")) == cmd

    def test_describe(self):
        sb = NoopSandbox()
        assert "none" in sb.describe()


class TestGetSandbox:
    def test_none_backend(self):
        sb = get_sandbox("none")
        assert isinstance(sb, NoopSandbox)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            get_sandbox("invalid")

    @patch("sys.platform", "linux")
    def test_native_linux_fallback_when_bwrap_missing(self):
        with patch("shutil.which", return_value=None):
            sb = get_sandbox("native")
            assert isinstance(sb, NoopSandbox)

    @patch("sys.platform", "darwin")
    def test_native_macos_uses_docker(self):
        """On macOS, 'native' redirects to Docker sandbox."""
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        with patch("shutil.which", return_value="/usr/local/bin/docker"):
            sb = get_sandbox("native")
            assert isinstance(sb, DockerSandbox)

    @patch("sys.platform", "darwin")
    def test_native_macos_falls_back_when_docker_missing(self):
        """On macOS, 'native' falls back to NoopSandbox if Docker is unavailable."""
        with patch("shutil.which", return_value=None):
            sb = get_sandbox("native")
            assert isinstance(sb, NoopSandbox)

    def test_docker_fallback_when_unavailable(self):
        with patch("shutil.which", return_value=None):
            sb = get_sandbox("docker")
            assert isinstance(sb, NoopSandbox)


class TestBubblewrapSandbox:
    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_wrap_command(self):
        from mcp_python_exec_sandbox.sandbox_linux import BubblewrapSandbox

        with patch("shutil.which", return_value="/usr/bin/bwrap"):
            sb = BubblewrapSandbox()
            cmd = ["uv", "run", "--script", "--python", "3.13", "/tmp/test.py"]
            wrapped = sb.wrap(cmd, Path("/tmp/test.py"))
            assert wrapped[0] == "/usr/bin/bwrap"
            assert "--unshare-all" in wrapped
            assert "--share-net" in wrapped
            assert "--die-with-parent" in wrapped
            # Original command should be at the end
            assert wrapped[-6:] == cmd


class TestDockerSandbox:
    def test_wrap_command(self):
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        with patch("shutil.which", return_value="/usr/local/bin/docker"):
            sb = DockerSandbox()
            cmd = ["uv", "run", "--script", "--python", "3.13", "/tmp/test.py"]
            wrapped = sb.wrap(cmd, Path("/tmp/test.py"))
            assert wrapped[0] == "/usr/local/bin/docker"
            assert "run" in wrapped
            assert "--rm" in wrapped
            assert "--read-only" in wrapped
            assert "--memory=512m" in wrapped

    def test_is_available_no_docker(self):
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        with patch("shutil.which", return_value=None):
            sb = DockerSandbox()
            assert sb.is_available() is False
