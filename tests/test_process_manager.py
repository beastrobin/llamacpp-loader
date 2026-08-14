"""Tests for process_manager.manager module."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from llamacpp_loader.process_manager.manager import ServerConfig, ProcessManager


class TestServerConfig:

    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.model_path == ""
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.ctx_size == 4096

    def test_custom_values(self):
        cfg = ServerConfig(
            model_path="/models/model.gguf",
            port=9000, gpu_layers=50, n_threads=8)
        assert cfg.port == 9000
        assert cfg.gpu_layers == 50


class TestProcessManagerInitialState:

    def test_initial_state_is_idle(self):
        mgr = ProcessManager(log_callback=None)
        assert mgr.state.value == "idle"

    def test_get_pid_returns_none_when_idle(self):
        mgr = ProcessManager(log_callback=None)
        assert mgr.get_pid() is None

    def test_is_running_false_when_idle(self):
        mgr = ProcessManager(log_callback=None)
        assert not mgr.is_running()


class TestProcessManagerBuildCommand:

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_includes_model_path(self, mock_popen):
        cfg = ServerConfig(
            model_path="/models/model.gguf", port=9001)
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(cfg)
        assert "--model" in cmd
        assert "/models/model.gguf" in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_includes_all_params(self, mock_popen):
        cfg = ServerConfig(
            model_path="/models/model.gguf", port=9001, ctx_size=8192, gpu_layers=33)
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(cfg)
        assert "--port" in cmd
        assert "8192" in cmd  # ctx_size as -c value


class TestProcessManagerLifecycle:

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_start_returns_true(self, mock_popen):
        mgr = ProcessManager(log_callback=None)
        result = mgr.start(ServerConfig())
        assert result is True

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_stop_graceful(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited normally
        cfg = ServerConfig()
        mgr = ProcessManager(log_callback=None)
        mgr._process = mock_proc  # type: ignore[assignment]
        result = mgr.stop()
        assert result is True

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_restart_stops_then_starts(self, mock_popen):
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001)
        mgr = ProcessManager(log_callback=None)
        result = mgr.restart(cfg)
        assert result is True


class TestProcessWatcher:

    @patch("llamacpp_loader.process_manager.manager.ProcessManager")
    def test_watcher_starts(self, mock_mgr):
        from llamacpp_loader.process_manager.manager import ProcessWatcher
        watcher = ProcessWatcher(mock_mgr, poll_interval=0.1)
        assert not watcher._stop_event.is_set()

    @patch("llamacpp_loader.process_manager.manager.ProcessManager")
    def test_watcher_stop(self, mock_mgr):
        from llamacpp_loader.process_manager.manager import ProcessWatcher
        watcher = ProcessWatcher(mock_mgr)
        watcher.stop_watching()
        assert watcher._stop_event.is_set()


class TestProcessManagerWithProfile:

    """Test that ProcessManager correctly extracts config from ModelProfile."""

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_start_from_profile(self, mock_popen, sample_server_config):
        mgr = ProcessManager(log_callback=None)
        result = mgr.start(sample_server_config)
        assert result is True
        # Verify config was properly converted to command args
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "--model" in cmd


class TestProcessStateTransitions:

    def test_start_rejects_invalid_config_type(self):
        mgr = ProcessManager(log_callback=None)
        result = mgr.start("not a config")  # type: ignore[arg-type]
        assert result is False


class TestBrowserOpen:

    @patch("llamacpp_loader.process_manager.manager.webbrowser.open")
    def test_opens_browser_on_start(self, mock_open):
        mgr = ProcessManager(log_callback=None)
        mgr._open_browser(8080)
        mock_open.assert_called_once_with("http://localhost:8080")
