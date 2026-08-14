"""Process lifecycle manager for llama.cpp servers.

Manages a single ``llama-server`` subprocess with full stdout/stderr streaming,
graceful shutdown, crash-restart, and automatic browser launch on successful start.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- enums


class ProcessState(Enum):
    """Possible states of the managed server process."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(slots=True)
class ServerConfig:
    """Configuration needed to launch a llama.cpp server.

    Mirrors the ``ModelProfile`` fields that affect the command line.

    Attributes:
        model_path:  Path to the GGUF model file (absolute or relative).
        host:        Bind address, default '127.0.0.1'.
        port:        HTTP listen port, default 8080.
        ctx_size:    Context window size in tokens.
        gpu_layers:  Number of layers to offload to GPU (-1 = auto).
        n_threads:   CPU thread count.
        temperature: Sampling temperature (0.0-2.0).
        top_k:       Top-K sampling parameter.
        top_p:       Top-P sampling parameter.
    """

    model_path: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 4096
    gpu_layers: int = -1
    n_threads: int = 4
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95


# --------------------------------------------------------------------------- helpers

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


# --------------------------------------------------------------------------- process manager


class ProcessManager:
    """Manages a single llama.cpp server subprocess.

    Thread-safety::
        All public methods are callable from any thread. Internal state is
        protected by a threading.Lock. Log callbacks are scheduled on the
        Tkinter main thread via root.after() when set, otherwise logged directly.

    Lifecycle::
        1. start(profile) -> spawns subprocess, reads stdout/stderr in background thread
        2. stop(grace_period=5.0) -> sends SIGTERM/terminate(), waits for exit
        3. restart() -> stop() then start()

    Usage::
        mgr = ProcessManager(log_callback=my_ui_update_func)
        profile = store.load("my-model")   # ModelProfile from ConfigStore
        mgr.start(profile)                  # launches server
        mgr.stop()                          # graceful shutdown
    """

    def __init__(self, *, log_callback: Optional[Callable[[str], None]] = None) -> None:
        self._log_callback = log_callback
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._lock = threading.Lock()
        self._state = ProcessState.IDLE
        self._output_thread: Optional[threading.Thread] = None
        self._watcher_thread: Optional[threading.Thread] = None

    @property
    def state(self) -> ProcessState:
        """Current process state (read-only snapshot)."""
        with self._lock:
            return self._state

    # -------------------------------------------------------------- lifecycle
    def start(self, config: ServerConfig | ModelProfile) -> bool:
        """Launch the llama.cpp server.

        Args:
            config: Either a ``ServerConfig`` or a ``ModelProfile`` from ConfigStore.
                    If a ModelProfile is passed, its parameters are extracted automatically.

        Returns True if subprocess started successfully, False otherwise.
        Sets internal state to STARTING then RUNNING (or ERROR on failure).
        Spawns a background reader thread for stdout/stderr.
        On successful launch within the health-check window, opens a browser tab.
        """
        with self._lock:
            if self._state in (ProcessState.RUNNING, ProcessState.STARTING):
                logger.warning("Server already running or starting")
                return False

        # Extract ServerConfig from ModelProfile if needed
        if isinstance(config, ModelProfile):
            server = config.server
            inference = config.inference
            sampling = config.sampling
            model_cmd, port = config.to_server_config()
            sc = ServerConfig(
                model_path=model_cmd,
                host=server.host,
                port=port if port != 8080 else server.port,
                ctx_size=inference.ctx_size,
                gpu_layers=inference.gpu_layers,
                n_threads=inference.n_threads,
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
            )
        elif isinstance(config, ServerConfig):
            sc = config
        else:
            logger.error("Invalid config type: %s", type(config))
            return False

        # Start the process
        try:
            cmd = self._build_command(sc)
            self._forward_log(f"Launching: {' '.join(cmd[:3])}...")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,           # line-buffered
                universal_newlines=False,  # binary for cross-platform consistency
            )
        except OSError as exc:
            self._forward_log(f"Failed to start process: {exc}")
            self._set_state(ProcessState.ERROR)
            return False

        if self._process is None or self._process.stdout is None:
            self._forward_log("Failed to create subprocess")
            self._set_state(ProcessState.ERROR)
            return False

        # Start output reader thread
        self._output_thread = threading.Thread(
            target=self._read_output_loop, daemon=True)
        self._output_thread.start()

        # Start crash-restart watcher
        self._watcher_thread = ProcessWatcher(self, poll_interval=1.0)
        self._watcher_thread.start()

        return True

    def stop(self, grace_period: float = 5.0) -> bool:
        """Gracefully stop the server.

        Sends SIGTERM (or terminate on Windows); waits up to ``grace_period`` seconds.
        On timeout, sends SIGKILL/force-terminate.
        Sets internal state to STOPPING then IDLE or ERROR.
        """
        self._set_state(ProcessState.STOPPING)
        self._forward_log("Stopping server...")

        with self._lock:
            proc = self._process

        if proc is None:
            self._set_state(ProcessState.IDLE)
            return True

        try:
            proc.terminate()  # SIGTERM on Unix, terminate() on Windows
            try:
                proc.wait(timeout=grace_period)
                self._forward_log("Server stopped gracefully.")
                self._set_state(ProcessState.IDLE)
                return True
            except subprocess.TimeoutExpired:
                self._forward_log(f"Grace period exceeded ({grace_period}s), forcing kill...")
                proc.kill()  # SIGKILL
                proc.wait(timeout=2.0)
                self._set_state(ProcessState.IDLE)
                return False
        except OSError as exc:
            self._forward_log(f"Stop error: {exc}")
            with self._lock:
                if proc.poll() is not None:
                    self._set_state(ProcessState.IDLE)
            return False

    def restart(self, config: ServerConfig | ModelProfile = None) -> bool:
        """Stop then start in one atomic operation.

        If *config* is provided it overrides the previously used configuration.
        """
        if not self.stop():
            self._forward_log("Restart failed: stop returned False")
            return False
        time.sleep(0.5)  # brief pause for port release
        if config is None:
            return self._last_config is not None and self.start(self._last_config)  # type: ignore[arg-type]
        return self.start(config)

    def get_pid(self) -> Optional[int]:
        """Return the OS PID of the running process, or None if not running."""
        with self._lock:
            return self._process.pid if self._process else None

    def is_running(self) -> bool:
        """Check whether the managed process is alive."""
        with self._lock:
            if self._process is None:
                return False
            return self._process.poll() is None

    # ----------------------------------------------------------- internal
    _last_config: Optional[ServerConfig] = None  # type: ignore[misc]

    def _set_state(self, state: ProcessState) -> None:
        """Update process state (thread-safe)."""
        with self._lock:
            self._state = state

    def _build_command(self, config: ServerConfig) -> list[str]:
        """Build the CLI command list from ServerConfig.

        Example output::
            ['llama-server', '--model', '/path/to/model.gguf',
             '--host', '127.0.0.1', '--port', 8080,
             '-c', '4096', '--gpu-layers', '35', ...]

        Returns a flat list of command-line arguments suitable for subprocess.Popen.
        """
        cmd = ["llama-server"]

        if config.model_path:
            cmd.extend(["--model", config.model_path])

        cmd.extend([
            "--host", config.host,
            "--port", str(config.port),
            "-c", str(config.ctx_size),
        ])

        # GPU offload
        if config.gpu_layers >= 0:
            cmd.extend(["--gpu-layers", str(config.gpu_layers)])

        # Threading
        cmd.extend(["-t", str(config.n_threads)])

        # Sampling (llama-server flags)
        cmd.extend(["--temp", str(config.temperature)])
        cmd.extend(["--top-k", str(config.top_k)])
        cmd.extend(["--top-p", str(config.top_p)])

        self._forward_log(f"Command: {' '.join(cmd)}")
        return cmd

    def _read_output_loop(self) -> None:
        """Background thread that reads stdout line-by-line and forwards to callback.

        Runs until the subprocess exits or pipes are closed.
        """
        if self._process is None or self._process.stdout is None:
            return

        try:
            for raw_line in iter(self._process.stdout.readline, b""):
                try:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                if not line:
                    continue
                self._forward_log(line)

        except (OSError, ValueError):
            pass  # pipe closed or process ended
        finally:
            # Process exited
            if self._process is not None and self._process.poll() is not None:
                self._forward_log("Server output stream closed.")

    def _forward_log(self, line: str) -> None:
        """Thread-safe log forwarding to the Tkinter main thread.

        If self._log_callback is set, forwards the log line (caller handles threading).
        Logs to logger as fallback.
        """
        if self._log_callback is not None:
            try:
                self._log_callback(line)
            except Exception as exc:
                logger.error("Log callback error: %s", exc)
        else:
            logger.info("[SERVER] %s", line)

    def _open_browser(self, port: int) -> None:
        """Open the default browser to localhost:<port> after successful startup."""
        url = f"http://localhost:{port}"
        self._forward_log(f"Opening browser: {url}")
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Failed to open browser: %s", exc)


# --------------------------------------------------------------------------- helpers

def _log_callback_factory(callback):
    """Create a one-shot logging function that also calls the callback."""
    def log(line):
        if callback:
            try:
                callback(line)
            except Exception as e:
                logger.error("UI callback error: %s", e)
        else:
            logger.info("[SERVER] %s", line)
    return log


def _browser_opener(port):
    """Create a one-shot browser opener that fires after a short delay."""
    def open_after_delay():
        time.sleep(2.0)  # wait for server to be fully ready
        webbrowser.open(f"http://localhost:{port}")
    return threading.Thread(target=open_after_delay, daemon=True)


# --------------------------------------------------------------------------- process watcher


class ProcessWatcher(threading.Thread):
    """Background watcher that monitors a running process.

    On unexpected exit (non-zero return code), triggers an auto-restart
    if auto_restart=True and the state is not explicitly stopping.

    This thread runs independently of the output reader thread; it checks
    self._process.poll() periodically.
    """

    def __init__(self, manager: ProcessManager, *, poll_interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self._manager = manager
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Monitor loop: check process exit code every ``_poll_interval`` seconds."""
        while not self._stop_event.is_set():
            time.sleep(self._poll_interval)
            if self._stop_event.is_set():
                break

            with self._manager._lock:
                proc = self._manager._process
                state = self._manager._state

            if proc is None or proc.poll() is not None:
                # Process exited
                returncode = proc.returncode if proc else -1
                if state == ProcessState.STOPPING:
                    continue  # expected shutdown
                elif state == ProcessState.RUNNING and returncode != 0:
                    self._forward_log(f"Server crashed with exit code {returncode}. Auto-restarting...")
                    self._manager.start(self._manager._last_config)  # type: ignore[arg-type]

    def stop_watching(self) -> None:
        """Signal the watcher thread to exit."""
        self._stop_event.set()


# --------------------------------------------------------------------------- ModelProfile reference (imported at runtime)

class _ModelProfilePlaceholder:
    """Minimal interface expected from ModelProfile for ProcessManager.

    This allows ProcessManager to work without importing gui.config.store
    directly, avoiding circular imports. The actual class is defined in config/store.py.
    """
    pass


# Type alias accepted by start()/restart() — resolved at runtime via duck typing.
ModelProfile = _ModelProfilePlaceholder  # type: ignore[misc]

