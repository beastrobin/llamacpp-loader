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
        # While running, Start becomes a safe shortcut to the existing Web UI.
        assert str(mw._toolbar_start_btn.cget("state")) == "normal"  # type: ignore[attr-defined]
        assert mw._toolbar_start_btn.cget("text") == "Open Web"  # type: ignore[attr-defined]
        assert str(mw._toolbar_stop_btn.cget("state")) == "normal"  # type: ignore[attr-defined]
        assert str(mw._toolbar_restart_btn.cget("state")) == "normal"  # type: ignore[attr-defined]


class TestMainLayout:
    """Regression coverage for the model table/detail split pane."""

    def test_model_panes_are_mapped_and_added_model_is_visible(
            self, tk_root_fixture, tmp_path):
        from unittest.mock import MagicMock, patch
        from llamacpp_loader.config.store import ConfigStore, ModelProfile
        from llamacpp_loader.gui.app import MainWindow

        model_file = tmp_path / "visible-model.gguf"
        model_file.write_bytes(b"GGUF")
        isolated = ConfigStore(path=tmp_path / "settings.json")
        isolated.add(ModelProfile(
            profile_name="visible-model",
            display_name="Visible Model",
            model_path=str(tmp_path),
            gguf_file=model_file.name,
        ))

        with patch("llamacpp_loader.config.store.ConfigStore", return_value=isolated):
            mw = MainWindow(tk_root_fixture)

        tk_root_fixture.update_idletasks()
        panes = {str(pane) for pane in mw._upper_pane.panes()}

        assert mw._table_frame.winfo_parent() == str(mw._upper_pane)
        assert mw._detail_frame.winfo_parent() == str(mw._upper_pane)
        assert str(mw._table_frame) in panes
        assert str(mw._detail_frame) in panes
        assert mw._detail_panel.winfo_manager() == "pack"
        assert mw._detail_panel.pack_info()["fill"] == "both"
        assert mw._detail_panel._title.cget("text") == "Visible Model"
        # The upper splitter intentionally uses the same themed ttk pane as
        # Server/Test so its sash has the same visual and hit target.
        assert mw._upper_pane.winfo_class() == "TPanedwindow"
        assert mw._workspace_separator.winfo_manager() == "pack"
        assert mw._tree.exists("visible-model")
        # The custom canvas header is the only header; an empty ``show`` value
        # prevents ttk from reserving a second blank native heading row.
        assert mw._tree.cget("show") == ""

        mw._running_profile_name = "visible-model"
        mw.proc_mgr.stop = MagicMock()
        mw._on_stop()
        assert mw._tree.set("visible-model", "status") == "Ready"

        # The custom header must be real canvas content and remain aligned
        # with the Treeview when the shared horizontal scrollbar moves.
        assert mw._header_labels["model"].winfo_parent() == str(mw._header_inner)
        mw._tree.column("model", width=1200)
        mw._relayout_header()
        mw._scroll_table_x("moveto", "1.0")
        tk_root_fixture.update_idletasks()
        assert mw._tree.xview()[0] > 0
        assert abs(mw._tree.xview()[0] - mw._header_canvas.xview()[0]) < 0.02


class TestModelDetailPanel:
    """Compact parameter editor conversions and live budget display."""

    def test_context_k_and_kv_display_map_to_runtime_values(
            self, tk_root_fixture, tmp_path):
        from llamacpp_loader.config.store import InferenceParams, ModelProfile
        from llamacpp_loader.gui.detail_panel import ModelDetailPanel

        model = tmp_path / "demo-q4.gguf"
        model.write_bytes(b"GGUF")
        changes = []
        panel = ModelDetailPanel(
            tk_root_fixture,
            on_change=lambda name, update: changes.append((name, update)),
            on_preset=lambda *_: None,
            on_action=lambda *_: None)
        panel.set_profile(ModelProfile(
            profile_name="demo", display_name="Demo",
            model_path=str(tmp_path), gguf_file=model.name,
            kv_cache="q8_0",
            inference=InferenceParams(ctx_size=32768, n_predict=12000)))

        assert panel._vars["ctx"].get() == "32"
        assert panel._vars["kv"].get() == "Q8"
        assert int(panel._inputs["gpu"].cget("width")) == 9
        assert int(panel._inputs["gpu"].grid_info()["pady"]) == 2
        assert panel._vscroll.winfo_manager() == "grid"
        assert panel._hscroll.winfo_manager() == "grid"
        assert panel._canvas.cget("xscrollcommand")
        assert panel._canvas.cget("yscrollcommand")
        assert panel._canvas.bind("<MouseWheel>")
        assert "Estimated VRAM" in panel._vram.cget("text")
        assert "weights" in panel._vram.cget("text")
        assert "KV" in panel._vram.cget("text")
        assert panel._pages["Quick"].grid_columnconfigure(0)["weight"] == 0
        assert panel._pages["Generation"].grid_columnconfigure(0)["weight"] == 0
        # CPU-MoE sits immediately below GPU layers and shares its columns.
        assert {child.grid_info()["row"] for child in panel._quick_columns[0].grid_slaves()} == {0, 1, 2, 3, 4}
        assert {child.grid_info()["row"] for child in panel._quick_columns[1].grid_slaves()} == {0, 1, 2, 3}
        # Advanced is a four-row form, not four explanatory Configure buttons.
        assert set(panel._capability_status) == {
            "vision_status", "mtp_status", "dflash_status", "ngram_status"}
        assert panel._vars["mtp_n_max"].get() == "7"
        assert panel._inputs["mtp_n_max"].cget("from") == 1.0
        assert panel._inputs["mtp_n_max"].cget("to") == 16.0
        # Max tokens row: maps to inference.n_predict and shows 0 = auto.
        assert panel._vars["max_tokens"].get() == "12000"
        assert panel._vars["cpu_moe_mode"].get() == "GPU all"
        assert panel._inputs["cpu_moe_layers"].master is panel._quick_columns[0]
        assert panel._inputs["cpu_moe_mode"].grid_info()["column"] == panel._inputs["gpu"].grid_info()["column"]
        assert panel._inputs["cpu_moe_layers"].grid_info()["column"] == 3

        panel._vars["ctx"].set("64")
        panel._commit("ctx")
        panel._vars["kv"].set("Q4")
        panel._commit("kv")

        assert ("demo", {"inference.ctx_size": 65536}) in changes
        assert ("demo", {"kv_cache": "q4_0"}) in changes

        panel._vars["mtp_enabled"].set("On")
        panel._commit_capability("mtp")
        panel._vars["mtp_n_max"].set("4")
        panel._commit_mtp_n_max()
        panel._vars["cpu_moe_mode"].set("First N")
        panel._vars["cpu_moe_layers"].set("12")
        panel._commit_cpu_moe()
        assert ("demo", {"mtp_enabled": True}) in changes
        assert ("demo", {"mtp_n_max": 4}) in changes
        assert ("demo", {"cpu_moe": False, "n_cpu_moe": 12}) in changes


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
