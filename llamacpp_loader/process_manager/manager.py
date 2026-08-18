"""Process lifecycle manager for llama.cpp servers.

Manages a single ``llama-server`` subprocess with full stdout/stderr streaming,
graceful shutdown, crash-restart, and automatic browser launch on successful start.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import signal
import subprocess
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
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
    repeat_penalty: float = 1.1
    kv_cache: str = "f16"          # f16 / q8_0 / q4_0
    reasoning: bool = False        # --reasoning on/off
    mmproj: str = ""               # vision projector GGUF path (--mmproj)
    mtp_enabled: bool = False      # use an MTP draft model (speculative decoding)
    mtp_model: str = ""            # draft GGUF path (--spec-draft-model)
    mtp_native: bool = False       # native MTP head inside the model GGUF (no external draft)
    mtp_n_max: int = 3             # max draft tokens (--spec-draft-n-max)
    flash_attn: str = "auto"      # Flash Attention: "auto" | "on" | "off"


# --------------------------------------------------------------------------- helpers

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _resolve_llama_server_executable(model_path: str = "", config_root: str = "") -> str:
    """Locate llama-server.exe / llama-server.

    Resolution order:
      1. If *config_root* (user-configured llama.cpp install dir) is given and
         contains llama-server.exe, use that.
      2. Walk up from *model_path* looking for a directory that contains
         llama-server.exe.
      3. Walk up from *config_root* the same way.
      4. Fall back to PATH lookup.
      5. Final fallback: just the bare name.
    """
    import shutil
    exe_win = "llama-server.exe"
    exe_nix = "llama-server"

    def try_dir(d: str) -> str | None:
        if not d:
            return None
        p = Path(d)
        for _ in range(8):  # walk up to 8 levels
            for n in (exe_win, exe_nix):
                cand = p / n
                if cand.is_file():
                    return str(cand)
            if p.parent == p:
                break
            p = p.parent
        return None

    # 1. config_root
    for n in (try_dir(config_root), try_dir(model_path)):
        if n:
            return n
    # 2. PATH
    found = shutil.which(exe_nix) or shutil.which(exe_win)
    if found:
        return found
    # 3. Bare name
    return exe_win if os.name == "nt" else exe_nix


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

    def __init__(self, *, log_callback: Optional[Callable[[str], None]] = None,
                 config_store=None) -> None:
        self._log_callback = log_callback
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._lock = threading.Lock()
        self._state = ProcessState.IDLE
        self._output_thread: Optional[threading.Thread] = None
        self._watcher_thread: Optional[threading.Thread] = None
        # Cached llama-server path; auto-detected on first start, persisted later
        self._config_store = config_store
        self._cached_server_path: Optional[str] = None

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
        MP = _get_model_profile_class()
        if isinstance(config, MP):
            server = config.server
            inference = config.inference
            sampling = config.sampling
            model_cmd, port = config.to_server_config()
            # Pick the first extra file that looks like a vision projector
            mmproj = ""
            for f in config.extra_files:
                if "mmproj" in f.lower() or "clip" in f.lower():
                    mmproj = os.path.join(config.model_path, f) if config.model_path else f
                    break
            # MTP draft model (separate field so it is not confused with mmproj)
            mtp_model = ""
            if config.mtp_enabled and config.mtp_model:
                mtp_model = (os.path.join(config.model_path, config.mtp_model)
                             if config.model_path else config.mtp_model)
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
                repeat_penalty=sampling.repeat_penalty,
                kv_cache=config.kv_cache,
                reasoning=config.reasoning,
                mmproj=mmproj,
                mtp_enabled=config.mtp_enabled,
                mtp_model=mtp_model,
                mtp_native=getattr(config, "mtp_native", False),
                mtp_n_max=getattr(config, "mtp_n_max", 3),
                flash_attn=getattr(config.server, "flash_attn", "auto"),
            )
        elif isinstance(config, ServerConfig):
            sc = config
        else:
            logger.error("Invalid config type: %s", type(config))
            return False

        # Start the process
        try:
            cmd = self._build_command(sc)
            # Log the full command line so the operator can verify flags like
            # -c (ctx size) actually made it into the launch, instead of only
            # the first three tokens.
            self._forward_log("Launching: " + " ".join(cmd))

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,           # line-buffered
                universal_newlines=False,  # binary for cross-platform consistency
                # CREATE_NO_WINDOW (0x08000000) hides the black console window
                # that would otherwise pop up on Windows when launching the
                # llama-server subprocess (e.g. triggered by Smoke Test).
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
        except OSError as exc:
            import traceback
            self._forward_log(f"Failed to start process: {exc}")
            self._forward_log(f"Traceback:\n{traceback.format_exc()}")
            self._forward_log(f"Command was: {cmd}")
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

        # Record the config so the watchdog / restart() can re-launch on crash,
        # and mark the process RUNNING so the watcher's auto-restart branch
        # (which only fires for state == RUNNING) can actually trigger.
        self._last_config = sc

        # Persist ownership to disk so a later loader session can adopt and
        # stop this server after this GUI exits. Previously the in-memory
        # Popen handle was the *only* record of the child process, so closing
        # and reopening the loader orphaned the server (could not be stopped,
        # and a second server could be spawned on top of it).
        try:
            write_registry(
                self._process.pid,
                sc.host,
                sc.port,
                self._cached_server_path or "",
                sc.model_path or "",
            )
        except Exception as exc:  # registry is best-effort; never block start
            logger.warning("Failed to write server registry: %s", exc)

        self._set_state(ProcessState.RUNNING)
        return True

    def stop(self, grace_period: float = 5.0) -> bool:
        """Gracefully stop the server.

        Sends SIGTERM/terminate(); waits up to ``grace_period`` seconds. On
        timeout, force-kills.

        If this session holds no in-memory process handle (e.g. a previous
        loader session launched the server and we adopted it from the on-disk
        PID registry at startup, or this session never started one), we fall
        back to killing by PID so a lingering server is still stopped instead
        of being orphaned.
        """
        self._set_state(ProcessState.STOPPING)
        self._forward_log("Stopping server...")

        with self._lock:
            proc = self._process

        if proc is None:
            # No handle held by this session — try to adopt the server that a
            # previous session registered, so we can still shut it down.
            proc = self._adopt_from_registry()
            if proc is None:
                self._forward_log("No server handle or registry entry found.")
                self._set_state(ProcessState.IDLE)
                return True

        try:
            proc.terminate()  # SIGTERM on Unix, TerminateProcess on Windows
            try:
                proc.wait(timeout=grace_period)
                self._forward_log("Server stopped gracefully.")
                self._set_state(ProcessState.IDLE)
                return True
            except subprocess.TimeoutExpired:
                self._forward_log(f"Grace period exceeded ({grace_period}s), forcing kill...")
                proc.kill()  # SIGKILL / TerminateProcess
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._forward_log("Force kill did not terminate the process.")
                self._set_state(ProcessState.IDLE)
                return False
        except OSError as exc:
            self._forward_log(f"Stop error: {exc}")
            if proc.poll() is not None:
                self._set_state(ProcessState.IDLE)
            return False
        finally:
            # Drop the on-disk registry entry once we have acted on the server,
            # so a future session does not try to adopt a server we've stopped.
            clear_registry()

    def _adopt_from_registry(self) -> Optional["_PidHandle"]:
        """Reconnect to a server tracked in the on-disk registry.

        Returns a ``_PidHandle`` if the registered PID is still alive, else
        ``None`` (and clears a stale registry entry). Used by ``stop()`` when
        this session holds no real Popen handle.
        """
        data = read_registry()
        if data is None:
            return None
        pid = data.get("pid")
        if pid is None or not _pid_is_llama_server(pid):
            clear_registry()
            return None
        self._forward_log(
            f"Adopting server from registry (PID {pid}, port {data.get('port')})")
        return _PidHandle(pid)

    def recover(self) -> Optional[int]:
        """Adopt a server left running by a previous loader session.

        Called at startup. If the registry points to a live process, this
        session takes ownership (state RUNNING) so the UI's Stop button works
        and the server is not orphaned. Returns the adopted PID or ``None``.
        """
        data = read_registry()
        if data is None:
            return None
        pid = data.get("pid")
        if pid is None or not _pid_is_llama_server(pid):
            clear_registry()
            return None
        with self._lock:
            self._process = _PidHandle(pid)
        self._set_state(ProcessState.RUNNING)
        self._forward_log(
            f"Recovered running server from previous session (PID {pid}, "
            f"port {data.get('port')})")
        return pid

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
        # Determine llama-server executable path
        config_root = ""
        if self._config_store is not None:
            try:
                ui = self._config_store.get_ui_state()
                config_root = ui.llama_server_path or ""
            except Exception:
                pass
        exe = self._cached_server_path or _resolve_llama_server_executable(
            config.model_path, config_root=config_root)
        # Cache for next time
        self._cached_server_path = exe
        cmd = [exe]

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

        # KV cache quantization
        if config.kv_cache and config.kv_cache.lower() != "f16":
            cmd.extend(["--cache-type-k", config.kv_cache.lower()])
            cmd.extend(["--cache-type-v", config.kv_cache.lower()])

        # Vision projector. If the profile explicitly declares an mmproj but the
        # file is missing, FAIL LOUDLY instead of silently launching a text-only
        # server: a silent downgrade is exactly what makes vision "look broken"
        # (the service starts fine but image input does nothing). The raised
        # FileNotFoundError (an OSError subclass) is caught by start()'s
        # try/except, which logs it and moves the server to the ERROR state.
        if config.mmproj:
            if os.path.isfile(config.mmproj):
                cmd.extend(["--mmproj", config.mmproj])
            else:
                raise FileNotFoundError(
                    "Declared mmproj projector not found, refusing to start a "
                    f"vision-less server: {config.mmproj}")

        # MTP speculative decoding (Multi-Token Prediction).
        #  - External draft: a separate draft GGUF is passed via
        #    --spec-draft-model (MTP / DFlash / EAGLE3, type inferred from the
        #    filename).
        #  - Native draft: the draft head is bundled inside the model GGUF
        #    itself (e.g. empero-ai Ridge quants expose blk.<n>.* / nextn
        #    tensors); then only --spec-type draft-mtp is needed, no external
        #    file.  Without --spec-type the server silently ignores the draft,
        #    so it MUST be set explicitly in both cases.
        if config.mtp_enabled:
            if config.mtp_model and os.path.isfile(config.mtp_model):
                stem = config.mtp_model.lower().replace(" ", "-")
                if "dflash" in stem:
                    draft_type = "draft-dflash"
                elif "eagle" in stem:
                    draft_type = "draft-eagle3"
                else:
                    draft_type = "draft-mtp"
                cmd.extend(["--spec-type", draft_type])
                cmd.extend(["--spec-draft-model", config.mtp_model])
                cmd.extend(["--spec-draft-n-max", str(config.mtp_n_max)])
            elif config.mtp_native:
                cmd.extend(["--spec-type", "draft-mtp"])
                cmd.extend(["--spec-draft-n-max", str(config.mtp_n_max)])
            elif config.mtp_model:
                self._forward_log(
                    f"draft model not found, skipping speculative decoding: {config.mtp_model}")

        # Reasoning toggle (llama-server >= b4xxx supports --reasoning on/off).
        # Only emit when enabling reasoning — "off" is the server default, so
        # omitting it keeps us compatible with older llama-server builds that
        # reject the unknown flag.
        if config.reasoning:
            cmd.extend(["--reasoning", "on"])

        # Flash Attention — "auto" lets llama-server enable it when the GPU /
        # driver supports it ("on" forces it, "off" disables).  Always emitted
        # so behaviour is explicit and reproducible across launches.
        if config.flash_attn:
            cmd.extend(["--flash-attn", config.flash_attn])

        # Jinja chat template — REQUIRED for correct chat formatting and tool
        # calling. Verified on 2026-08-14: without --jinja, tool calls
        # silently fail. Safe to enable for every model, so always on.
        cmd.append("--jinja")

        # Sampling (llama-server flags)
        cmd.extend(["--temp", str(config.temperature)])
        cmd.extend(["--top-k", str(config.top_k)])
        cmd.extend(["--top-p", str(config.top_p)])
        cmd.extend(["--repeat-penalty", str(config.repeat_penalty)])

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
                    self._manager._forward_log(
                        f"Server crashed with exit code {returncode}. Auto-restarting...")
                    self._manager.start(self._manager._last_config)  # type: ignore[arg-type]

    def stop_watching(self) -> None:
        """Signal the watcher thread to exit."""
        self._stop_event.set()


# --------------------------------------------------------------------------- ModelProfile reference (imported at runtime)

_ModelProfile = None  # lazily imported to avoid circular imports


def _get_model_profile_class():
    """Import and cache the real ModelProfile class (config.store imports manager?)."""
    global _ModelProfile
    if _ModelProfile is None:
        from llamacpp_loader.config.store import ModelProfile as _MP
        _ModelProfile = _MP
    return _ModelProfile


# --------------------------------------------------------------------------- server registry (on-disk PID file)
#
# The in-memory Popen handle is lost whenever the loader GUI exits, which
# previously left a launched llama-server orphaned (un-stoppable, and a second
# server could be spawned on top of it). We persist ownership to a small JSON
# file so *any* loader session can adopt and stop the server by PID.

def _registry_path() -> Path:
    """Location of the on-disk server PID registry."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or tempfile.gettempdir()
        directory = Path(base) / "llamacpp-loader"
    else:
        directory = Path.home() / ".config" / "llamacpp-loader"
    return directory / "server.pid"


def write_registry(pid: int, host: str, port: int, exe: str, model: str = "") -> None:
    """Record the running server's PID and metadata to disk."""
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pid": pid,
        "host": host,
        "port": port,
        "exe": exe,
        "model": model,
        "started_at": time.time(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def read_registry() -> Optional[dict]:
    """Read the server registry, or ``None`` if absent/corrupt."""
    path = _registry_path()
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "pid" not in data:
        return None
    return data


def clear_registry() -> None:
    """Remove the server registry file (best-effort)."""
    try:
        _registry_path().unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """Return ``True`` if a process with *pid* is currently running."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        except AttributeError:
            return False
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                # STILL_ACTIVE == 259
                return exit_code.value == 0x103
            return False
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _kill_pid(pid: int) -> None:
    """Terminate the process with *pid* (best-effort, cross-platform)."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        except AttributeError:
            return
        PROCESS_TERMINATE = 0x0001
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            kernel32.TerminateProcess(handle, 0)
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _pid_image_name(pid: int) -> Optional[str]:
    """Return the executable image path of *pid*, or ``None`` if unavailable."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        except AttributeError:
            return None
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value
            return None
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None


def _pid_is_llama_server(pid: int) -> bool:
    """True if *pid* is alive AND its image name looks like llama-server.

    Guards against PID reuse: a stale registry pointing at a PID that the OS
    has since reassigned to an unrelated process must not be adopted or killed.
    """
    if not _pid_alive(pid):
        return False
    name = _pid_image_name(pid)
    if not name:
        # Image could not be resolved (e.g. access denied) — trust the PID
        # check alone rather than refuse to manage a server we likely own.
        return True
    base = os.path.basename(name).lower()
    return "llama-server" in base or "llama_server" in base


class _PidHandle:
    """Minimal ``subprocess.Popen`` stand-in backed by an OS PID.

    Lets a ProcessManager control a server it did not spawn (one adopted from
    the on-disk registry). Only the subset of the Popen interface used by
    stop()/is_running()/get_pid() is implemented.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self.stdout = None  # no pipe to read from
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        return None if _pid_alive(self.pid) else 0

    def terminate(self) -> None:
        _kill_pid(self.pid)

    def kill(self) -> None:
        _kill_pid(self.pid)

    def wait(self, timeout: Optional[float] = None) -> int:
        if timeout is None:
            while _pid_alive(self.pid):
                time.sleep(0.1)
            self.returncode = 0
            return 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _pid_alive(self.pid):
                self.returncode = 0
                return 0
            time.sleep(0.1)
        raise subprocess.TimeoutExpired(self.pid, timeout)

