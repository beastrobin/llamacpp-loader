"""Tests for process_manager.manager module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from llamacpp_loader.process_manager.manager import (
    ProcessState,
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


class TestParallelSlots:
    """-np must be sent even for a single slot.

    llama.cpp's own default is "-1 = auto", which current builds resolve to
    several slots.  With a unified KV cache the total -c is then shared out, so
    an auto slot count silently serves a fraction of the configured window
    (observed: -c 131072 became n_ctx_slot 32768 at n_slots = 4) and every
    extra slot holds its own per-sequence state (~400 MB on a hybrid model).
    """

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_single_slot_is_explicit(self, mock_popen):
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001, n_parallel=1)

        cmd = ProcessManager(log_callback=None)._build_command(cfg)

        assert cmd[cmd.index("-np") + 1] == "1"

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_configured_slot_count_is_forwarded(self, mock_popen):
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001, n_parallel=4)

        cmd = ProcessManager(log_callback=None)._build_command(cfg)

        assert cmd[cmd.index("-np") + 1] == "4"

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_a_negative_value_still_defers_to_llama_cpp(self, mock_popen):
        """-1 keeps meaning "pick for me"; only that stays unset."""
        cfg = ServerConfig(model_path="/models/m.gguf", port=9001, n_parallel=-1)

        cmd = ProcessManager(log_callback=None)._build_command(cfg)

        assert "-np" not in cmd


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
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on"))
        i = cmd.index("--reasoning")
        assert cmd[i + 1] == "on"

    def test_reasoning_auto_emits_no_flag(self):
        """auto is llama.cpp's own default, so the argument stays off."""
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001))
        assert "--reasoning" not in cmd

    def test_legacy_bool_reasoning_still_maps(self):
        """Configs written before the tri-state change stored a bool."""
        on = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning=True))
        assert on[on.index("--reasoning") + 1] == "on"
        auto = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning=False))
        assert "--reasoning" not in auto

    def test_reasoning_off_omitted_without_env_or_fixed_build(self):
        """Buggy build: no flag at all + a guidance warning (not a crash)."""
        from llamacpp_loader.process_manager import manager
        lines = []
        mgr = ProcessManager(log_callback=lines.append)
        cmd = mgr._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="off"))
        assert "--reasoning" not in cmd
        assert any("enable_thinking=false" in ln for ln in lines)
        # Clean up the once-per-instance warning latch for other tests.
        mgr._reasoning_off_warned = True

    def test_reasoning_off_emitted_when_env_override_set(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_REASONING_OFF", "1")
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="off"))
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


class TestUbatchBackendSamplingAndMinP:
    """The Hermes prefill recipe: -ub 2048, --backend-sampling, --min-p.

    All three are opt-in: llama.cpp exits on unknown flags, so defaults must
    emit nothing at all.
    """

    def test_defaults_emit_nothing(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001))
        assert "-ub" not in cmd
        assert "--backend-sampling" not in cmd
        assert "--spec-draft-backend-sampling" not in cmd
        assert "--min-p" not in cmd

    def test_ubatch_emitted_when_set(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001,
                         n_batch=4096, n_ubatch=2048))
        assert cmd[cmd.index("-ub") + 1] == "2048"

    def test_ubatch_ignored_when_exceeding_batch(self):
        lines = []
        cmd = ProcessManager(log_callback=lines.append)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001,
                         n_batch=512, n_ubatch=2048))
        assert "-ub" not in cmd
        assert any("ubatch" in ln for ln in lines)

    def test_backend_sampling_emits_both_flags(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, backend_sampling=True))
        assert "--backend-sampling" in cmd
        assert "--spec-draft-backend-sampling" in cmd

    def test_metrics_emits_flag_when_enabled(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, metrics=True))
        assert "--metrics" in cmd

    def test_metrics_omitted_by_default(self):
        """llama.cpp exits on arguments it does not recognise, so the flag is
        emitted only when the profile actually asks for it."""
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001))
        assert "--metrics" not in cmd

    def test_metrics_never_touches_slots_flags(self):
        """llama.cpp exposes /slots by default; --slots would be noise and
        --no-slots would break external monitoring."""
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, metrics=True))
        assert "--slots" not in cmd
        assert "--no-slots" not in cmd

    def test_min_p_explicit_zero_is_emitted(self):
        """0.0 disables min-p (Hermes/Qwen recipe); it must NOT be treated
        as an unset sentinel — that is what None is for."""
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, min_p=0.0))
        assert cmd[cmd.index("--min-p") + 1] == "0.0"

    def test_min_p_value_emitted(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, min_p=0.1))
        assert cmd[cmd.index("--min-p") + 1] == "0.1"

    def test_min_p_none_omitted(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, min_p=None))
        assert "--min-p" not in cmd

    def test_profile_mapping_carries_new_fields(self):
        from llamacpp_loader.config.store import InferenceParams, ModelProfile
        profile = ModelProfile(profile_name="m", gguf_file="m.gguf",
                               inference=InferenceParams(n_batch=4096,
                                                         n_ubatch=2048))
        profile.server.backend_sampling = True
        profile.server.metrics = True
        profile.sampling.min_p = 0.0
        mgr = ProcessManager(log_callback=None)
        cmd = mgr._build_command(mgr._profile_to_server_config(profile))
        assert cmd[cmd.index("-ub") + 1] == "2048"
        assert "--backend-sampling" in cmd
        assert "--metrics" in cmd
        assert cmd[cmd.index("--min-p") + 1] == "0.0"


class TestReasoningBudget:
    """--reasoning-budget caps the thinking trace on its own.

    Every flag here is opt-in: llama.cpp exits on arguments it does not know,
    so an unset box must produce no argument at all.
    """

    def test_all_flags_omitted_when_unset(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on"))
        assert "--reasoning-budget" not in cmd
        assert "--reasoning-budget-message" not in cmd
        assert "--reasoning-effort" not in cmd

    def test_budget_emitted_when_set(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_budget=4096))
        assert cmd[cmd.index("--reasoning-budget") + 1] == "4096"

    def test_zero_budget_is_emitted(self):
        """0 means "end thinking immediately" — a real value, not "unset"."""
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_budget=0))
        assert cmd[cmd.index("--reasoning-budget") + 1] == "0"

    def test_negative_one_budget_is_emitted(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_budget=-1))
        assert cmd[cmd.index("--reasoning-budget") + 1] == "-1"

    def test_message_requires_a_budget(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_budget_message="wrap up"))
        assert "--reasoning-budget-message" not in cmd

    def test_message_emitted_with_budget(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_budget=2048,
                         reasoning_budget_message="wrap up"))
        assert cmd[cmd.index("--reasoning-budget-message") + 1] == "wrap up"

    def test_effort_default_emits_nothing(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_effort="default"))
        assert "--reasoning-effort" not in cmd

    def test_effort_emitted_when_set(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_effort="high"))
        assert cmd[cmd.index("--reasoning-effort") + 1] == "high"

    def test_invalid_effort_falls_back_to_default(self):
        cmd = ProcessManager(log_callback=None)._build_command(
            ServerConfig(model_path="/m.gguf", port=9001, reasoning="on",
                         reasoning_effort="banana"))
        assert "--reasoning-effort" not in cmd


# ============================================ crash-loop budget + launch diagnosis


class TestAutoRestartBudget:
    """A model the binary cannot load must stop respawning.

    Regression (the endless-loop report): ``start()`` cleared the
    consecutive-restart counter on *every* launch, including the relaunch
    issued by the crash handler.  ``MAX_AUTO_RESTARTS`` was therefore
    unreachable and llama-server respawned once a second forever whenever the
    model file itself was the problem -- e.g. a GGUF whose tensors use a ggml
    type the running build does not implement, which llama-server rejects
    ~0.3 s after launch.
    """

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_explicit_start_resets_the_budget(self, mock_popen):
        mgr = ProcessManager(log_callback=None)
        mgr._restart_attempts = 2
        assert mgr.start(ServerConfig(model_path="/m.gguf", port=9001)) is True
        assert mgr._restart_attempts == 0

    @patch("llamacpp_loader.process_manager.manager.subprocess.Popen")
    def test_crash_relaunch_keeps_the_budget(self, mock_popen):
        mgr = ProcessManager(log_callback=None)
        mgr._restart_attempts = 2
        assert mgr.start(
            ServerConfig(model_path="/m.gguf", port=9001),
            reset_restart_budget=False) is True
        assert mgr._restart_attempts == 2

    def test_crash_handler_relaunches_without_resetting(self):
        mgr = ProcessManager(log_callback=None)
        mgr._state = ProcessState.RUNNING
        mgr._last_config = ServerConfig(model_path="/m.gguf", port=9001)
        mgr._process = MagicMock()
        with patch.object(ProcessManager, "start", return_value=True) as start:
            mgr._handle_crash(1)
        assert start.call_args.kwargs.get("reset_restart_budget") is False

    def test_loop_gives_up_after_max_auto_restarts(self):
        """Four consecutive crashes -> three respawns, then ERROR for good."""
        lines: list[str] = []
        mgr = ProcessManager(log_callback=lines.append)
        mgr._last_config = ServerConfig(model_path="/m.gguf", port=9001)

        def fake_start(config, *, reset_restart_budget=True):
            # Mimic a real launch so the next crash looks like a live server.
            mgr._state = ProcessState.RUNNING
            return True

        with patch.object(ProcessManager, "start", side_effect=fake_start):
            for _ in range(4):
                mgr._state = ProcessState.RUNNING
                mgr._process = MagicMock()
                mgr._handle_crash(1)

        assert mgr._restart_attempts == ProcessManager.MAX_AUTO_RESTARTS
        assert sum("Auto-restarting" in ln for ln in lines) == 3
        assert any("Giving up" in ln for ln in lines)
        assert mgr.state is ProcessState.ERROR

    def test_stop_during_crash_handling_wins(self):
        """A stop() landing mid-handling must never be resurrected.

        ``_handle_crash`` runs on the watcher thread, so a user pressing Stop
        at that exact moment is a real race.  Any state other than ERROR means
        someone else won and the relaunch is skipped.
        """
        lines: list[str] = []
        mgr = ProcessManager(log_callback=lines.append)
        mgr._state = ProcessState.RUNNING
        mgr._last_config = ServerConfig(model_path="/m.gguf", port=9001)
        mgr._process = MagicMock()
        # A no-op _set_state leaves the observable state at RUNNING, standing
        # in for "stop() got there first".
        with patch.object(ProcessManager, "_set_state", return_value=None):
            with patch.object(ProcessManager, "start", return_value=True) as start:
                mgr._handle_crash(1)
        start.assert_not_called()
        assert any("Auto-restart skipped" in ln for ln in lines)


class TestLastErrorLine:
    """The status bar must be able to explain *why* a launch failed.

    Without this the user only saw a generic smoke-test failure ("connection
    refused") while the real cause -- e.g. ``tensor 'output.weight' has invalid
    ggml type 142. should be in [0, 43)`` -- stayed buried in the console.
    """

    #: The exact line stock llama.cpp printed for the Bonsai 2 ternary GGUFs.
    BONSAI_REJECTION = (
        "0.34.120.950 E gguf_init_from_reader: tensor 'output.weight' has "
        "invalid ggml type 142. should be in [0, 43)")

    def test_empty_when_nothing_captured(self):
        assert ProcessManager(log_callback=None).last_error_line() == ""

    def test_ignores_ordinary_progress_lines(self):
        mgr = ProcessManager(log_callback=None)
        mgr._forward_log("llama_model_loader: loaded meta data")
        mgr._forward_log("main: server is listening on 127.0.0.1:8080")
        assert mgr.last_error_line() == ""

    def test_returns_most_recent_error(self):
        mgr = ProcessManager(log_callback=None)
        mgr._forward_log("error: first failure")
        mgr._forward_log("error: second failure")
        assert mgr.last_error_line() == "error: second failure"

    def test_skips_trailing_noise_after_the_error(self):
        mgr = ProcessManager(log_callback=None)
        mgr._forward_log(self.BONSAI_REJECTION)
        mgr._forward_log("")
        mgr._forward_log("   ")
        mgr._forward_log("srv  update_slots: all slots are idle")
        assert mgr.last_error_line() == self.BONSAI_REJECTION

    def test_recognises_bare_llama_cpp_log_level(self):
        """llama.cpp brackets its level with spaces: " E module: message"."""
        mgr = ProcessManager(log_callback=None)
        mgr._forward_log("load_tensors: offloading 64 repeating layers to GPU")
        mgr._forward_log("0.91.004.112 E llama_init_from_gpt_params: error "
                         "loading model")
        assert "error loading model" in mgr.last_error_line()

    def test_captures_failed_wording(self):
        mgr = ProcessManager(log_callback=None)
        mgr._forward_log("Smoke test failed: connection refused")
        assert mgr.last_error_line() == "Smoke test failed: connection refused"

    def test_recent_output_is_bounded(self):
        """A server spewing 160 KB of errors must not grow the buffer forever."""
        mgr = ProcessManager(log_callback=None)
        for i in range(500):
            mgr._forward_log(f"error line {i}")
        assert len(mgr._recent_output) == mgr._recent_output.maxlen
        assert mgr.last_error_line() == "error line 499"

    def test_log_callback_failure_does_not_break_capture(self):
        """Diagnostics must survive a broken GUI callback."""
        def boom(_line):
            raise RuntimeError("widget destroyed")

        mgr = ProcessManager(log_callback=boom)
        mgr._forward_log("error: still recorded")
        assert mgr.last_error_line() == "error: still recorded"
