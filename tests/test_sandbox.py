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

        mock_result = type("Result", (), {"returncode": 0})()
        with (
            patch("shutil.which", return_value="/usr/local/bin/docker"),
            patch("subprocess.run", return_value=mock_result),
            patch.object(DockerSandbox, "_image_exists", return_value=True),
        ):
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
            assert "ghcr.io/lu-zhengda/mcp-python-exec-sandbox" in wrapped

    def test_is_available_no_docker(self):
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        with patch("shutil.which", return_value=None):
            sb = DockerSandbox()
            assert sb.is_available() is False

    def test_is_available_starts_background_pull_when_image_missing(self):
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        mock_result = type("Result", (), {"returncode": 0})()
        with (
            patch("shutil.which", return_value="/usr/local/bin/docker"),
            patch("subprocess.run", return_value=mock_result),
        ):
            sb = DockerSandbox()
        with (
            patch.object(sb, "_image_exists", return_value=False),
            patch.object(sb, "_pull_image_background") as mock_bg,
            patch("subprocess.run", return_value=mock_result),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value.start = mock_bg
            assert sb.is_available() is True
            # Pull is started in background, not blocking
            mock_thread.assert_called_once()

    def test_is_available_skips_pull_when_image_exists(self):
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        mock_result = type("Result", (), {"returncode": 0})()
        with (
            patch("shutil.which", return_value="/usr/local/bin/docker"),
            patch("subprocess.run", return_value=mock_result),
        ):
            sb = DockerSandbox()
        with (
            patch.object(sb, "_image_exists", return_value=True) as mock_exists,
            patch.object(sb, "_pull_image") as mock_pull,
            patch("subprocess.run", return_value=mock_result),
        ):
            assert sb.is_available() is True
            mock_exists.assert_called_once()
            mock_pull.assert_not_called()

    def test_wrap_waits_for_background_pull(self):
        from mcp_python_exec_sandbox.sandbox_docker import DockerSandbox

        with patch("shutil.which", return_value="/usr/local/bin/docker"):
            sb = DockerSandbox()
        with patch.object(sb, "_ensure_image") as mock_ensure:
            cmd = ["uv", "run", "--script", "/tmp/test.py"]
            sb.wrap(cmd, Path("/tmp/test.py"))
            mock_ensure.assert_called_once()
