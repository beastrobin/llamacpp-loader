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
    """Test toolbar button state management on MainWindow."""

    def test_button_states(self, tk_root_fixture, tmp_path):
        from unittest.mock import MagicMock, patch
        from llamacpp_loader.config.store import ConfigStore
        from llamacpp_loader.gui.app import MainWindow

        # Isolate the store to a temp file so the test never touches the real
        # %APPDATA%\llamacpp-loader\settings.json on the developer's machine.
        isolated = ConfigStore(path=tmp_path / "settings.json")
        with patch("llamacpp_loader.config.store.ConfigStore", return_value=isolated):
            mw = MainWindow(tk_root_fixture)

        # Initially: Start enabled, Stop/Restart disabled
        assert str(mw._toolbar_start_btn.cget("state")) == "normal"  # type: ignore[attr-defined]
        assert str(mw._toolbar_stop_btn.cget("state")) == "disabled"  # type: ignore[attr-defined]
        assert str(mw._toolbar_restart_btn.cget("state")) == "disabled"  # type: ignore[attr-defined]

        mw._set_toolbar_running(True)
        assert str(mw._toolbar_start_btn.cget("state")) == "disabled"  # type: ignore[attr-defined]
        assert str(mw._toolbar_stop_btn.cget("state")) == "normal"  # type: ignore[attr-defined]
        assert str(mw._toolbar_restart_btn.cget("state")) == "normal"  # type: ignore[attr-defined]


class TestOnClosing:
    """Regression: closing the window must stop the managed llama-server.

    Previously the production entry point (main.py) registered a
    WM_DELETE_WINDOW handler that only saved window state and never stopped the
    server, leaving it running (and holding VRAM/RAM) after the GUI closed.
    The handler now lives in MainWindow._on_closing and must be registered and
    must call proc_mgr.stop().
    """

    def test_close_handler_registered(self, tk_root_fixture, tmp_path):
        from unittest.mock import MagicMock, patch
        from llamacpp_loader.config.store import ConfigStore
        from llamacpp_loader.gui.app import MainWindow

        isolated = ConfigStore(path=tmp_path / "settings.json")
        with patch("llamacpp_loader.config.store.ConfigStore", return_value=isolated):
            # Capture protocol() registrations made during construction.
            with patch.object(tk_root_fixture, "protocol", MagicMock()) as mock_proto:
                MainWindow(tk_root_fixture)

        registered = any(
            c.args and c.args[0] == "WM_DELETE_WINDOW"
            for c in mock_proto.call_args_list
        )
        assert registered, "WM_DELETE_WINDOW handler not registered on root"

    def test_close_stops_server_and_quits(self, tk_root_fixture, tmp_path):
        from unittest.mock import MagicMock, patch
        from llamacpp_loader.config.store import ConfigStore
        from llamacpp_loader.gui.app import MainWindow

        isolated = ConfigStore(path=tmp_path / "settings.json")
        with patch("llamacpp_loader.config.store.ConfigStore", return_value=isolated):
            mw = MainWindow(tk_root_fixture)

        # Stub destroy on the shared root so this test does not tear down the
        # session-scoped fixture used by other tests.
        real_destroy = tk_root_fixture.destroy
        tk_root_fixture.destroy = MagicMock()
        try:
            mw.proc_mgr.stop = MagicMock()  # observe the stop call
            mw._on_closing()  # invoke the close handler directly

            mw.proc_mgr.stop.assert_called_once()
            tk_root_fixture.destroy.assert_called_once()
        finally:
            tk_root_fixture.destroy = real_destroy
