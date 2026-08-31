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

    def test_cpu_moe_defaults_off(self):
        cfg = ServerConfig()
        assert cfg.cpu_moe is False


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

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_cpu_moe_emits_flag_when_enabled(self, mock_popen):
        cfg = ServerConfig(model_path="/models/moe.gguf", port=9001, cpu_moe=True)
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(cfg)
        assert "--cpu-moe" in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_cpu_moe_omitted_when_disabled(self, mock_popen):
        cfg = ServerConfig(model_path="/models/moe.gguf", port=9001)
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(cfg)
        assert "--cpu-moe" not in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_n_cpu_moe_emits_flag_when_set(self, mock_popen):
        cfg = ServerConfig(model_path="/models/moe.gguf", port=9001, n_cpu_moe=18)
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(cfg)
        assert "--n-cpu-moe" in cmd
        assert "18" in cmd
        assert "--cpu-moe" not in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_cpu_moe_takes_precedence_over_n_cpu_moe(self, mock_popen):
        cfg = ServerConfig(
            model_path="/models/moe.gguf", port=9001, cpu_moe=True, n_cpu_moe=18)
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(cfg)
        assert "--cpu-moe" in cmd
        assert "--n-cpu-moe" not in cmd


class TestNgramSpeculativeDecoding:
    """N-gram spec decoding is free (no draft model) and stackable."""

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_disabled_by_default(self, mock_popen):
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001)
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        assert "--spec-type" not in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_emits_ngram_type_when_enabled(self, mock_popen):
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001,
                           ngram_enabled=True, ngram_type="ngram-simple")
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "ngram-simple"

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_stacks_with_native_mtp(self, mock_popen):
        """The headline case: draft-mtp + ngram in one comma-joined flag."""
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001,
                           mtp_enabled=True, mtp_native=True, mtp_n_max=2,
                           ngram_enabled=True, ngram_type="ngram-simple")
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        spec = cmd[cmd.index("--spec-type") + 1]
        assert spec == "draft-mtp,ngram-simple"
        # Native MTP needs no draft file, but still honours n-max.
        assert "--spec-draft-model" not in cmd
        assert "--spec-draft-n-max" in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_stacks_with_external_draft(self, mock_popen, tmp_path):
        draft = tmp_path / "mtp-model.gguf"
        draft.write_bytes(b"")
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001,
                           mtp_enabled=True, mtp_model=str(draft), mtp_n_max=4,
                           ngram_enabled=True, ngram_type="ngram-mod")
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp,ngram-mod"
        assert "--spec-draft-model" in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_stacks_with_dflash(self, mock_popen, tmp_path):
        draft = tmp_path / "dflash.gguf"
        draft.write_bytes(b"")
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001,
                           dflash_enabled=True, dflash_model=str(draft),
                           ngram_enabled=True, ngram_type="ngram-map-k")
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        assert cmd[cmd.index("--spec-type") + 1] == "draft-dflash,ngram-map-k"

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_ngram_survives_missing_draft_file(self, mock_popen):
        """A broken draft path must not take the free n-gram speed-up down."""
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001,
                           mtp_enabled=True, mtp_model="/nope/gone.gguf",
                           ngram_enabled=True, ngram_type="ngram-simple")
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        assert cmd[cmd.index("--spec-type") + 1] == "ngram-simple"

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_unknown_ngram_type_is_ignored(self, mock_popen):
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001,
                           ngram_enabled=True, ngram_type="ngram-bogus")
        cmd = ProcessManager(log_callback=None)._build_command(cfg)
        assert "--spec-type" not in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_passes_through_from_profile(self, mock_popen, sample_profile):
        sample_profile.ngram_enabled = True
        sample_profile.ngram_type = "ngram-simple"
        mgr = ProcessManager(log_callback=None)
        assert mgr.start(sample_profile) is True
        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("--spec-type") + 1] == "ngram-simple"


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

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_cpu_moe_passes_through_from_profile(self, mock_popen, sample_profile):
        sample_profile.cpu_moe = True
        mgr = ProcessManager(log_callback=None)
        result = mgr.start(sample_profile)
        assert result is True
        cmd = mock_popen.call_args[0][0]
        assert "--cpu-moe" in cmd

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_n_cpu_moe_passes_through_from_profile(self, mock_popen, sample_profile):
        sample_profile.cpu_moe = False
        sample_profile.n_cpu_moe = 18
        mgr = ProcessManager(log_callback=None)
        result = mgr.start(sample_profile)
        assert result is True
        cmd = mock_popen.call_args[0][0]
        assert "--n-cpu-moe" in cmd
        assert "18" in cmd


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
