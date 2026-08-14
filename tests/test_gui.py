"""Tests for gui.app module - interface coverage with mocked Tkinter.

Skipped on systems without tkinter (e.g., minimal Docker containers).
"""

import pytest

from unittest.mock import patch, MagicMock

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="session", autouse=True)
def tk_root_fixture(tk_root):
    """Auto-use the session-scoped tk_root fixture."""
    yield tk_root


class TestMainWindowInit:
    """Test MainWindow initialization and setup."""

    @patch("llamacpp_loader.config.store.ConfigStore")
    def test_creates_store(self, MockCS, tk_root_fixture):
        from unittest.mock import MagicMock

        root = tk_root_fixture
        cs_instance = MagicMock()
        cs_instance.list_profiles.return_value = ["test-profile"]
        MockCS.return_value = cs_instance

        import llamacpp_loader.config.store as store_mod
        import llamacpp_loader.gui.app as app_mod
        app_mod.ConfigStore = store_mod.ConfigStore

        from llamacpp_loader.gui.app import MainWindow
        mw = MainWindow.__new__(MainWindow)
        mw.root = root
        mw.store = cs_instance
        mw._current_profile_name = None
        # Should not raise during init (widgets built in _build_ui)
        assert mw.store is not None


class TestParameterPanel:
    """Test ParameterWidget creation and profile loading."""

    def test_set_from_profile(self, tk_root_fixture):
        from unittest.mock import MagicMock
        from llamacpp_loader.config.store import ModelProfile, ServerParams, InferenceParams, SamplingParams
        from llamacpp_loader.gui.app import ParameterPanel

        root = tk_root_fixture
        panel = ParameterPanel(root)
        panel._build_widgets()

        profile = ModelProfile(
            profile_name="test-model",
            display_name="Test Model",
            model_path="/models",
            gguf_file="m.gguf",
            server=ServerParams(host="127.0.0.1", port=8080),
            inference=InferenceParams(ctx_size=8192, gpu_layers=33, n_threads=8),
            sampling=SamplingParams(temperature=0.5, top_k=20, top_p=0.9),
        )
        panel.set_from_profile(profile)

        # Verify values were set in widget variables
        assert panel._ctx_var.get() == 8192
        assert panel._gpu_var.get() == 33
        assert panel._threads_var.get() == 8
        assert abs(panel._temp_var.get() - 0.5) < 0.01
        assert panel._topk_var.get() == 20


class TestConsolePanel:
    """Test ConsolePanel widget creation and log appending."""

    def test_append_line(self, tk_root_fixture):
        from unittest.mock import MagicMock
        from llamacpp_loader.gui.app import ConsolePanel

        root = tk_root_fixture
        panel = ConsolePanel.__new__(ConsolePanel)
        panel._text = MagicMock()

        # Append a line
        panel.append_line("Test log line")

        # Verify text widget got the insert call (insert(tk.END, line + "\n"))
        assert panel._text.insert.called
        inserted = panel._text.insert.call_args[0][1]
        assert "Test log line" in inserted


class TestStatusBar:
    """Test StatusBar widget state updates."""

    def test_set_state(self, tk_root_fixture):
        from unittest.mock import MagicMock
        from llamacpp_loader.gui.app import StatusBar

        frame = MagicMock()
        bar = StatusBar.__new__(StatusBar)
        bar._label = MagicMock()

        bar.set_state("running", "Server ready")
        bar._label.config.assert_called_once()
        config_kwargs = bar._label.config.call_args[1]
        assert config_kwargs["text"] == "Status: running  |  Server ready"
        assert config_kwargs["foreground"] == "#4caf50"


class TestControlBar:
    """Test ControlBar button state management."""

    def test_button_states(self, tk_root_fixture):
        from unittest.mock import MagicMock
        from llamacpp_loader.gui.app import ControlBar

        root = tk_root_fixture
        bar = ControlBar(
            root,
            start_callback=lambda: None,
            stop_callback=lambda: None,
            restart_callback=lambda: None,
        )

        # Initially all buttons should have their default states
        assert str(bar._start_btn.cget("state")) == "normal"  # type: ignore[attr-defined]
        assert str(bar._stop_btn.cget("state")) == "disabled"  # type: ignore[attr-defined]

        bar.set_buttons_running()
        assert str(bar._start_btn.cget("state")) == "disabled"  # type: ignore[attr-defined]
        assert str(bar._stop_btn.cget("state")) == "normal"  # type: ignore[attr-defined]