"""Tests for process_manager.manager module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from llamacpp_loader.process_manager.manager import (
    ServerConfig,
    ProcessManager,
)


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

    def test_configured_folder_wins_over_model_directory_binary(self, tmp_path):
        install_dir = tmp_path / "llama.cpp"
        install_dir.mkdir()
        trusted_server = install_dir / "llama-server.exe"
        trusted_server.write_bytes(b"")
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "llama-server.exe").write_bytes(b"")

        store = MagicMock()
        store.get_ui_state.return_value = SimpleNamespace(
            llama_server_path=str(install_dir)
        )
        command = ProcessManager(config_store=store)._build_command(
            ServerConfig(model_path=str(model_dir / "model.gguf"))
        )

        assert command[0] == str(trusted_server)

    def test_invalid_configured_folder_does_not_fall_back_to_path(self, tmp_path):
        store = MagicMock()
        store.get_ui_state.return_value = SimpleNamespace(
            llama_server_path=str(tmp_path / "missing")
        )
        with patch("shutil.which", return_value="attacker-server") as which:
            with pytest.raises(FileNotFoundError, match="does not contain"):
                ProcessManager(config_store=store)._build_command(ServerConfig())
        which.assert_not_called()

    def test_config_store_requires_explicit_server_folder(self):
        store = MagicMock()
        store.get_ui_state.return_value = SimpleNamespace(llama_server_path="")
        with pytest.raises(ValueError, match="Choose the llama.cpp folder"):
            ProcessManager(config_store=store)._build_command(ServerConfig())

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
    def test_rejects_non_loopback_bind_addresses(self, host):
        with pytest.raises(ValueError, match="Refusing"):
            ProcessManager(log_callback=None)._build_command(ServerConfig(host=host))

    def test_normalizes_localhost_to_loopback(self):
        command = ProcessManager(log_callback=None)._build_command(
            ServerConfig(host="localhost")
        )
        assert command[command.index("--host") + 1] == "127.0.0.1"

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_start_rejects_non_loopback_before_popen(self, mock_popen):
        result = ProcessManager(log_callback=None).start(ServerConfig(host="0.0.0.0"))
        assert result is False
        mock_popen.assert_not_called()

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


# ==================================================== reasoning-off guard + n_predict


class TestReasoningOffGuard:
    """--reasoning off must never be emitted on buggy/unknown llama.cpp builds.

    b10588-era builds crash the chat endpoint when the server default is off
    (verified 2026-09-05), so the loader refuses unless the operator opts in
    via env or the build is known fixed (REASONING_OFF_MIN_BUILD).
    """

    def test_reasoning_on_is_always_emitted(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning=True))
        i = cmd.index("--reasoning")
        assert cmd[i + 1] == "on"

    def test_reasoning_off_omitted_without_env_or_fixed_build(self):
        """Default path: no flag at all + a guidance warning (not a crash)."""
        from llamacpp_loader.process_manager import manager
        lines = []
        mgr = ProcessManager(log_callback=lines.append)
        cmd = mgr._build_command(ServerConfig(model_path="/m.gguf", port=9001))
        assert "--reasoning" not in cmd
        assert any("enable_thinking=false" in ln for ln in lines)
        # Clean up the once-per-instance warning latch for other tests.
        mgr._reasoning_off_warned = True

    def test_reasoning_off_emitted_when_env_override_set(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_REASONING_OFF", "1")
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001))
        assert cmd[cmd.index("--reasoning") + 1] == "off"

    def test_reasoning_off_allowed_gates_on_fixed_build(self, monkeypatch):
        from llamacpp_loader.process_manager import manager
        monkeypatch.setattr(manager, "REASONING_OFF_MIN_BUILD", 20000)
        monkeypatch.setattr(
            manager, "_probe_server_build", staticmethod(lambda exe: 25000))
        assert manager.reasoning_off_allowed("/x/llama-server.exe") is True
        # Buggy build (b10588) still refuses even with a threshold set.
        monkeypatch.setattr(
            manager, "_probe_server_build", staticmethod(lambda exe: 10588))
        assert manager.reasoning_off_allowed("/x/llama-server.exe") is False

    def test_reasoning_off_refused_on_unknown_build(self, monkeypatch):
        from llamacpp_loader.process_manager import manager
        monkeypatch.setattr(manager, "REASONING_OFF_MIN_BUILD", 20000)
        monkeypatch.setattr(
            manager, "_probe_server_build", staticmethod(lambda exe: None))
        assert manager.reasoning_off_allowed("/x/llama-server.exe") is False


class TestNPredictEmission:

    def test_n_predict_emitted_when_set(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, n_predict=12000))
        assert cmd[cmd.index("--n-predict") + 1] == "12000"

    def test_n_predict_omitted_when_zero(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001))
        assert "--n-predict" not in cmd
