"""Tests for sandbox layer."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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
    def test_native_macos_fallback_when_sandbox_exec_missing(self):
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


class TestSandboxExecSandbox:
    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_wrap_command(self):
        from mcp_python_exec_sandbox.sandbox_macos import SandboxExecSandbox

        with patch("shutil.which", return_value="/usr/bin/sandbox-exec"):
            sb = SandboxExecSandbox()
            cmd = ["uv", "run", "--script", "--python", "3.13", "/tmp/test.py"]
            wrapped = sb.wrap(cmd, Path("/tmp/test.py"))
            assert wrapped[0] == "/usr/bin/sandbox-exec"
            assert "-p" in wrapped
            # Original command should be at the end
            assert wrapped[-6:] == cmd

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_describe(self):
        from mcp_python_exec_sandbox.sandbox_macos import SandboxExecSandbox

        with patch("shutil.which", return_value="/usr/bin/sandbox-exec"):
            sb = SandboxExecSandbox()
            assert "sandbox-exec" in sb.describe()


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
