"""Main Tkinter window and all GUI widgets.

Layout::
    +---------------------------------------------------------------+
    |  llamacpp-loader                                              |
    +---------------------------+-----------------------------------+
    | [Model Selector ▼]        | [Console Output (read-only)]      |
    |                           |                                   |
    | -- Inference Params --    | > Server log lines stream here   |
    | ctx_size:  [4096     ]    |                                   |
    | gpu_layers:[35       ]    +-----------------------------------+
    | n_threads: [8        ]    |                                     |
    |                           | -- Sampling Params --               |
    | -- Sampling Params --     | temperature:  [0.7      ]         |
    | temp:      [0.7      ]    | top_k:       [40        ]         |
    | top_p:     [0.95     ]    | top_p (server):[0.95     ]        |
    |                           +-------------------------------------+
    | [Browse Model] [Start] [Stop] [Restart] Status: idle           |
    +---------------------------------------------------------------+

Control flow::
    1. User clicks "Browse" -> file dialog selects GGUF model directory
    2. Profile list auto-refreshes from ConfigStore
    3. User selects a profile, params populate the fields
    4. User clicks "Start":
       a. Build ModelProfile from widget state (or use selected profile)
       b. ProcessManager.start(profile) -> launches llama-server subprocess
       c. SmokeTestRunner.wait_until_ready() polls health endpoint
       d. On PASS: webbrowser.open(http://localhost:<port>)
       e. ConsolePanel starts streaming server output
    5. User clicks "Stop" or process exits
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- MainWindow


class MainWindow:
    """Single-window main application class."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("llamacpp-loader")

        # Config store (thread-safe)
        from llamacpp_loader.config.store import ConfigStore
        self.store = ConfigStore()

        # Process manager
        from llamacpp_loader.process_manager.manager import ProcessManager, ServerConfig
        self.proc_mgr = ProcessManager(log_callback=self._on_server_log)

        # State
        self._current_profile_name: Optional[str] = None  # type: ignore[assignment]
        self._active_port: int = 8080

        self._build_ui()
        self._restore_window_state()
        self._refresh_profile_list()

    def _on_server_log(self, line: str) -> None:
        """Callback invoked by ProcessManager when a new log line arrives.

        Thread-safe: uses root.after() to schedule on the main thread.
        """
        self.root.after(0, lambda: self._console_panel.append_line(line))

    def _build_ui(self) -> None:
        """Build all widgets and lay them out."""
        # Main frame with padding
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Top row: model selector + buttons
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        self._build_toolbar(toolbar)

        # Main content area (left panel + right console)
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Parameter panels
        left_pane = ttk.PanedWindow(content, orient=tk.VERTICAL)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self._param_panel = ParameterPanel(left_pane, store=self.store)
        left_pane.add(self._param_panel, weight=1)

        # Right: Console panel
        self._console_panel = ConsolePanel(content)
        self._console_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Bottom: status bar + controls
        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.X, pady=(6, 0))
        self._status_bar = StatusBar(bottom)
        self._control_bar = ControlBar(
            bottom,
            start_callback=self._on_start,
            stop_callback=self._on_stop,
            restart_callback=self._on_restart,
            profile_refresh_callback=self._refresh_profile_list,
        )

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        """Build the top toolbar with model selector and browse button."""
        label = ttk.Label(parent, text="Model Profile:")
        label.pack(side=tk.LEFT)

        # Profile name dropdown (populated from ConfigStore)
        self._profile_var = tk.StringVar()
        self._profile_combo = ttk.Combobox(
            parent, textvariable=self._profile_var, state="readonly", width=30)
        self._profile_combo.pack(side=tk.LEFT, padx=(4, 4))
        self._profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)

        # Browse button
        browse_btn = ttk.Button(parent, text="Browse...", command=self._browse_model)
        browse_btn.pack(side=tk.RIGHT)

    def _refresh_profile_list(self) -> None:
        """Reload the profile list from ConfigStore and update combo box."""
        profiles = self.store.list_profiles()
        if not profiles:
            # Create a default empty profile for first use
            profile = self.store.create_default_profile(
                display_name="New Model",
                model_path="",
                gguf_file="",
            )
            self.store.add(profile)
            profiles = [profile.profile_name]

        self._profile_combo["values"] = profiles
        if not self._current_profile_name or self._current_profile_name not in profiles:
            self._profile_var.set(profiles[0])
            self._on_profile_selected(None)

    def _browse_model(self) -> None:
        """Open a directory selection dialog for the model path."""
        from tkinter import filedialog as fd
        dir_path = fd.askdirectory(
            title="Select Model Directory",
            initialdir=self.store.get_ui_state().last_browse_dir,
        )
        if not dir_path:
            return  # User cancelled

        self.store.set_ui_state(last_browse_dir=dir_path)

        # Look for GGUF files in selected directory
        gguf_files = list(Path(dir_path).glob("*.gguf"))
        if not gguf_files:
            messagebox.showwarning(
                "No GGUF Files",
                f"No .gguf files found in {dir_path}\n"
                "Please select a directory containing a GGUF model file.",
            )
            return

        # Auto-create or update profile with selected model
        gguf = gguf_files[0]  # Pick first for simplicity; user can adjust
        profile_name = gguf.stem.replace(" ", "-").lower()

        existing_profile = self.store.load(profile_name)
        if existing_profile:
            # Update existing profile's path
            self.store.update(profile_name, {
                "model_path": dir_path,
                "gguf_file": gguf.name,
            })
        else:
            new_profile = ModelProfile(
                display_name=gguf.stem,
                model_path=dir_path,
                gguf_file=gguf.name,
            )
            self.store.add(new_profile)

        # Refresh UI
        self._refresh_profile_list()
        self._load_profile_to_ui(profile_name)

    def _on_profile_selected(self, event=None) -> None:
        """Handler for profile selection change."""
        name = self._profile_var.get().strip()
        if not name:
            return
        self._current_profile_name = name
        self._load_profile_to_ui(name)

    def _load_profile_to_ui(self, name: str) -> None:
        """Load a profile\'s parameters into the UI widgets."""
        profile = self.store.load(name)
        if not profile:
            logger.warning("Profile %s not found", name)
            return

        # Populate parameter panel fields
        self._param_panel.set_from_profile(profile)

    def _restore_window_state(self) -> None:
        """Restore saved window dimensions and position."""
        ui = self.store.get_ui_state()
        w, h = ui.window_width, ui.window_height
        self.root.geometry(f"{w}x{h}")

    def save_window_state(self) -> None:
        """Save current window state to ConfigStore before exit."""
        try:
            geom = self.root.geometry()  # "960x680+100+200"
            size_part = geom.split("+")[0]
            w, h = map(int, size_part.split("x"))
            self.store.set_ui_state(window_width=w, window_height=h)
        except (ValueError, IndexError):
            pass  # Ignore parse errors on exit

    def _on_start(self) -> None:
        """Start the server with the currently selected profile."""
        name = self._profile_var.get().strip()
        if not name:
            messagebox.showwarning("No Profile", "Please select or create a model profile first.")
            return

        profile = self.store.load(name)
        if not profile:
            messagebox.showerror("Error", f"Profile '{name}' not found in config store.")
            return

        if not profile.gguf_file and not profile.model_path:
            messagebox.showwarning(
                "No Model Selected",
                "Please browse for a GGUF model file first.",
            )
            return

        # Start process manager
        success = self.proc_mgr.start(profile)
        if not success:
            self._status_bar.set_state("error", "Failed to start server")
            messagebox.showerror(
                "Start Failed",
                "Could not launch llama-server. Check logs for details.",
            )
            return

        # Update UI state
        self._active_port = profile.server.port
        self._status_bar.set_state("running", f"Running on port {profile.server.port}")
        self._control_bar.set_buttons_running()

        # Run smoke test in background thread, then open browser on success
        threading.Thread(
            target=self._smoke_test_and_open_browser,
            args=(profile,),
            daemon=True,
        ).start()

    def _smoke_test_and_open_browser(self, profile: ModelProfile) -> None:
        """Run smoke test and open browser if server is healthy.

        Runs in a background thread to avoid blocking the GUI.
        """
        from llamacpp_loader.smoke_test.runner import SmokeTestRunner, SmokeResult

        try:
            runner = SmokeTestRunner(
                host=profile.server.host,
                port=profile.server.port,
                timeout=15.0,
            )
            result = runner.wait_until_ready(poll_interval=1.0)

            if result.status == SmokeResult.PASS:
                # Success — open browser on the main thread
                self.root.after(
                    0,
                    lambda: self._open_browser(profile.server.port),
                )
                self._status_bar.set_state("running", f"Server ready (latency: {result.latency_ms}ms)")
            else:
                # Failed — update UI on main thread
                detail = result.detail or "Unknown error"
                self.root.after(
                    0,
                    lambda d=detail: self._status_bar.set_state("error", f"Smoke test failed: {d}"),
                )

        except Exception as exc:
            logger.exception("Smoke test exception")
            self.root.after(
                0,
                lambda e=str(exc): self._status_bar.set_state("error", f"Error: {e}"),
            )

    def _open_browser(self, port: int) -> None:
        """Open the default browser to the server\'s web UI."""
        import webbrowser
        url = f"http://localhost:{port}"
        logger.info("Opening browser: %s", url)
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Failed to open browser: %s", exc)

    def _on_stop(self) -> None:
        """Stop the running server."""
        self.proc_mgr.stop()
        self._status_bar.set_state("idle", "Server stopped")
        self._control_bar.set_buttons_idle()

    def _on_restart(self) -> None:
        """Restart the server with current profile settings."""
        name = self._profile_var.get().strip()
        if not name:
            messagebox.showwarning("No Profile", "Please select a model profile first.")
            return

        profile = self.store.load(name)
        if not profile:
            return

        success = self.proc_mgr.restart(profile)
        if success:
            self._status_bar.set_state("running", f"Restarting on port {profile.server.port}")


# --------------------------------------------------------------------------- ParameterPanel


class ParameterPanel(ttk.Frame):
    """Left-side parameter input panel."""

    def __init__(self, parent: ttk.Frame, store=None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._store = store  # Optional[ConfigStore] for defaults
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create all input widgets in this panel."""
        # Inference section
        inf_frame = ttk.LabelFrame(self, text="Inference", padding=4)
        inf_frame.pack(fill=tk.X, pady=(0, 4))

        # ctx_size
        ttk.Label(inf_frame, text="Context Size:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4), pady=2)
        self._ctx_var = tk.IntVar(value=4096)
        ctx_spin = ttk.Spinbox(
            inf_frame, from_=64, to=131072, increment=512,
            textvariable=self._ctx_var, width=10, command=self._on_param_change)
        ctx_spin.grid(row=0, column=1, sticky=tk.W, pady=2)

        # gpu_layers
        ttk.Label(inf_frame, text="GPU Layers:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=2)
        self._gpu_var = tk.IntVar(value=-1)
        gpu_spin = ttk.Spinbox(
            inf_frame, from_=-1, to=999, textvariable=self._gpu_var, width=10,
            command=self._on_param_change)
        gpu_spin.grid(row=1, column=1, sticky=tk.W, pady=2)

        # n_threads
        ttk.Label(inf_frame, text="Threads:").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=2)
        self._threads_var = tk.IntVar(value=4)
        threads_spin = ttk.Spinbox(
            inf_frame, from_=1, to=128, textvariable=self._threads_var, width=10,
            command=self._on_param_change)
        threads_spin.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Sampling section
        sam_frame = ttk.LabelFrame(self, text="Sampling", padding=4)
        sam_frame.pack(fill=tk.X, pady=(4, 0))

        # temperature
        ttk.Label(sam_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4), pady=2)
        self._temp_var = tk.DoubleVar(value=0.7)
        temp_spin = ttk.Spinbox(
            sam_frame, from_=0.0, to=2.0, increment=0.05,
            textvariable=self._temp_var, width=10, command=self._on_param_change)
        temp_spin.grid(row=0, column=1, sticky=tk.W, pady=2)

        # top_k
        ttk.Label(sam_frame, text="Top-K:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=2)
        self._topk_var = tk.IntVar(value=40)
        topk_spin = ttk.Spinbox(
            sam_frame, from_=1, to=256, textvariable=self._topk_var, width=10,
            command=self._on_param_change)
        topk_spin.grid(row=1, column=1, sticky=tk.W, pady=2)

        # top_p (server param)
        ttk.Label(sam_frame, text="Top-P:").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=2)
        self._topp_var = tk.DoubleVar(value=0.95)
        topp_spin = ttk.Spinbox(
            sam_frame, from_=0.0, to=1.0, increment=0.01,
            textvariable=self._topp_var, width=10, command=self._on_param_change)
        topp_spin.grid(row=2, column=1, sticky=tk.W, pady=2)

    def _on_param_change(self) -> None:
        """Auto-save current parameter values to ConfigStore."""
        if not self._store or not self._current_profile_name:
            return  # type: ignore[attr-defined]

        try:
            self._store.update(
                str(self._current_profile_name),  # type: ignore[arg-type]
                {
                    "inference.ctx_size": int(self._ctx_var.get()),
                    "inference.gpu_layers": int(self._gpu_var.get()),
                    "inference.n_threads": max(1, int(self._threads_var.get())),
                    "sampling.temperature": round(float(self._temp_var.get()), 2),
                    "sampling.top_k": max(1, int(self._topk_var.get())),
                    "sampling.top_p": round(float(self._topp_var.get()), 2),
                },
            )
        except Exception as exc:
            logger.warning("Failed to auto-save parameters: %s", exc)

    def set_from_profile(self, profile) -> None:
        """Load a ModelProfile\'s values into the UI widgets.

        Args:
            profile: A ModelProfile from ConfigStore.load()
        """
        self._current_profile_name = profile.profile_name  # type: ignore[attr-defined]
        self._ctx_var.set(profile.inference.ctx_size)
        self._gpu_var.set(profile.inference.gpu_layers)
        self._threads_var.set(max(1, profile.inference.n_threads))
        self._temp_var.set(profile.sampling.temperature)
        self._topk_var.set(profile.sampling.top_k)
        self._topp_var.set(profile.sampling.top_p)

    def get_profile(self, display_name: str, model_path: str, gguf_file: str):
        """Build a ModelProfile from current widget values.

        Returns a fully-populated ModelProfile ready for ProcessManager.start().
        """
        from llamacpp_loader.config.store import (
            ModelProfile, ServerParams, InferenceParams, SamplingParams)
        return ModelProfile(
            profile_name=self._current_profile_name or display_name.lower().replace(" ", "-"),  # type: ignore[attr-defined]
            display_name=display_name,
            model_path=model_path,
            gguf_file=gguf_file,
            server=ServerParams(host="127.0.0.1", port=8080),
            inference=InferenceParams(
                ctx_size=int(self._ctx_var.get()),
                gpu_layers=int(self._gpu_var.get()),
                n_threads=max(1, int(self._threads_var.get())),
            ),
            sampling=SamplingParams(
                temperature=float(self._temp_var.get()),
                top_k=max(1, int(self._topk_var.get())),
                top_p=float(self._topp_var.get()),
            ),
        )


# --------------------------------------------------------------------------- ConsolePanel


class ConsolePanel(ttk.Frame):
    """Right-side live output panel (read-only text area)."""

    def __init__(self, parent: ttk.Frame, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create the read-only console output widget."""
        frame = ttk.LabelFrame(self, text="Server Output", padding=4)
        frame.pack(fill=tk.BOTH, expand=True)

        # Text widget with vertical scrollbar
        self._text = tk.Text(
            frame,
            wrap=tk.WORD,
            state=tk.DISABLED,  # read-only
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Consolas" if __import__("platform").system().lower() == "windows" else ("Monaco" if __import__("platform").system().lower() == "darwin" else "Ubuntu Mono")),  # noqa: E501
            padx=6,
            pady=4,
        )
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._text.yview)
        self._text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(fill=tk.BOTH, expand=True)

    def append_line(self, line: str) -> None:
        """Append a single log line to the console output.

        Thread-safe via root.after(). Appends at end of text widget and
        auto-scrolls to show new content.
        """
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, line + "\n")
        # Auto-scroll to bottom
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        """Clear all console output."""
        self._text.configure(state=tk.NORMAL)
        self._text.delete(1.0, tk.END)
        self._text.configure(state=tk.DISABLED)


# --------------------------------------------------------------------------- StatusBar


class StatusBar(ttk.Frame):
    """Bottom status bar showing process state and server info."""

    def __init__(self, parent: ttk.Frame, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._build_widgets()

    def _build_widgets(self) -> None:
        self._label = ttk.Label(
            self, text="Status: idle", relief=tk.SUNKEN, anchor=tk.W)
        self.pack(fill=tk.X)

    def set_state(self, state: str, message: str = "") -> None:
        """Update the status bar with a new state and optional detail message.

        Args:
            state: One of "idle", "running", "error" — used for color coding.
            message: Human-readable detail shown after the state label.
        """
        colors = {"idle": "#888888", "running": "#4caf50", "error": "#f44336"}
        self._label.config(text=f"Status: {state}  |  {message}", foreground=colors.get(state, "#888888"))


# --------------------------------------------------------------------------- ControlBar

class ControlBar(ttk.Frame):
    """Button bar with Start / Stop / Restart and profile management."""

    def __init__(self, parent: ttk.Frame, **kwargs) -> None:
        self._start_callback = kwargs.pop("start_callback", lambda: None)  # type: ignore[arg-type]
        self._stop_callback = kwargs.pop("stop_callback", lambda: None)  # type: ignore[arg-type]
        self._restart_callback = kwargs.pop("restart_callback", lambda: None)  # type: ignore[arg-type]
        self._profile_refresh_callback = kwargs.pop("profile_refresh_callback", lambda: None)  # type: ignore[arg-type]
        super().__init__(parent, **kwargs)
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create control buttons and status display."""
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.LEFT)

        # Start button
        self._start_btn = ttk.Button(btn_frame, text="Start", command=self._on_start)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 4))

        # Stop button (disabled by default)
        self._stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=(4, 4))

        # Restart button (disabled by default)
        self._restart_btn = ttk.Button(
            btn_frame, text="Restart", command=self._on_restart, state=tk.DISABLED)
        self._restart_btn.pack(side=tk.LEFT, padx=4)

        # Status label on the right side of toolbar
        self._status_label = ttk.Label(self, text="  |  Server: idle")
        self._status_label.pack(side=tk.RIGHT)

    def set_buttons_running(self) -> None:
        """Update button states when server is running."""
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._restart_btn.config(state=tk.NORMAL)

    def set_buttons_idle(self) -> None:
        """Update button states when server is idle/stopped."""
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._restart_btn.config(state=tk.DISABLED)

    def _on_start(self) -> None:
        """Callback for Start button — delegates to MainWindow."""
        try:
            self._start_callback()
        except Exception as exc:
            logger.error("Start callback error: %s", exc)

    def _on_stop(self) -> None:
        """Callback for Stop button."""
        try:
            self._stop_callback()
        except Exception as exc:
            logger.error("Stop callback error: %s", exc)

    def _on_restart(self) -> None:
        """Callback for Restart button."""
        try:
            self._restart_callback()
        except Exception as exc:
            logger.error("Restart callback error: %s", exc)


# --------------------------------------------------------------------------- Entry point (used by main.py and tests)

def create_app(root=None):
    """Factory function to create a MainWindow instance.

    Used both by main.py (with real tk.Tk()) and by test_gui for unit testing.
    """
    if root is None:
        root = tk.Tk()
        # Clean exit on window close
        def on_closing():
            mw.root.after(0, lambda: logger.info("App closing"))
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_closing)
    return MainWindow(root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app(tk.Tk())
    tk.mainloop()

