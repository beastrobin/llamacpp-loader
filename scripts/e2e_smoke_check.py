"""Headless end-to-end check of the 'Start -> Smoke Test' flow.

This reuses the REAL ProcessManager + SmokeTestRunner + ConfigStore from the
project, so it proves the logic works against the real llama-server.exe on
the local machine -- NO GUI / tkinter needed. It replicates exactly what the
Smoke Test button does in app.py:

    1. pick the first profile that has a model
    2. start the server (or attach to one already listening on the port)
    3. wait for health (/health -> /v1/models fallback) up to 120s
    4. run the speed test with the endpoint fallback chain
       (/v1/completions -> /v1/chat/completions -> /tokenize)

Run it from the project root (or anywhere) with:
    python scripts/e2e_smoke_check.py

It prints every step so you can see exactly where things break.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make sure we import the project's own package, not any installed copy.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llamacpp_loader.config.store import ConfigStore            # noqa: E402
from llamacpp_loader.process_manager.manager import ProcessManager  # noqa: E402
from llamacpp_loader.smoke_test.runner import SmokeTestRunner   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("e2e")


def pick_profile(store: ConfigStore):
    """Return the first profile that actually has a model attached."""
    names = store.list_profiles()
    log.info("Profiles on disk: %s", names)
    for name in names:
        p = store.load(name)
        if p and (p.gguf_file or p.model_path):
            return p
    return None


def probe_port(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _try_speed(url: str, payload: str, parse):
    req = urllib.request.Request(url, data=payload.encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    elapsed = time.time() - t0
    return parse(json.loads(body), elapsed)


def _parse_completion(data, elapsed):
    n = (data.get("usage", {}) or {}).get("completion_tokens") or 0
    if elapsed > 0:
        return f"[OK] Generated {n} tokens in {elapsed:.1f}s -> {n / elapsed:.1f} tokens/s"
    return "[OK] Generated 0 tokens"


def _parse_chat(data, elapsed):
    n = (data.get("usage", {}) or {}).get("completion_tokens") or 0
    if elapsed > 0:
        return f"[OK] Chat generated {n} tokens in {elapsed:.1f}s -> {n / elapsed:.1f} tokens/s"
    return "[OK] Chat generated 0 tokens"


def _parse_tokenize(data, elapsed):
    n = len(data.get("tokens", []))
    if elapsed > 0:
        return f"[OK] Tokenized {n} tokens in {elapsed:.1f}s -> {n / elapsed:.1f} tok/s (tokenize only)"
    return "[OK] Tokenized 0 tokens"


def speed_test(port: int) -> str:
    """Replicates the Smoke Test speed-test fallback chain in app.py."""
    endpoints = [
        (f"http://127.0.0.1:{port}/v1/completions",
         '{"model": "model", "prompt": "Introduce yourself in 50 words", "max_tokens": 100, "temperature": 0.7}',
         _parse_completion),
        (f"http://127.0.0.1:{port}/v1/chat/completions",
         '{"model": "model", "messages": [{"role": "user", "content": "Introduce yourself in 50 words"}], "max_tokens": 100, "temperature": 0.7}',
         _parse_chat),
        (f"http://127.0.0.1:{port}/tokenize",
         '{"content": "Introduce yourself in 50 words"}',
         _parse_tokenize),
    ]
    last_error = None
    for url, payload, parse in endpoints:
        try:
            return _try_speed(url, payload, parse)
        except urllib.error.HTTPError as exc:
            last_error = exc
            # 404/405 means the endpoint is simply not exposed; try the next one.
            if exc.code not in (404, 405):
                raise
    if last_error:
        raise last_error
    return "[WARN] No compatible generation endpoint available for speed test"


def main() -> int:
    store = ConfigStore()
    profile = pick_profile(store)
    if not profile:
        log.error("No model profile found in settings.json")
        return 1

    port = profile.server.port
    log.info("Selected profile: %s (port %s)", profile.display_name, port)

    mgr = ProcessManager(config_store=store)
    if mgr.is_running():
        log.info("Managed server process is already alive -- will smoke-test it directly")
    elif probe_port(port):
        log.info("A server is already listening on port %s -- will smoke-test it directly", port)
    else:
        log.info("Starting server via ProcessManager.start() ...")
        ok = mgr.start(profile)
        if not ok:
            log.error("ProcessManager.start() returned False -- server did NOT launch. "
                      "Check the llama-server path / model path above.")
            return 1
        log.info("ProcessManager.start() returned True -- subprocess spawned")

    log.info("Waiting for server health (up to 120s) ...")
    runner = SmokeTestRunner(port=port, timeout=120.0)
    res = runner.wait_until_ready()
    log.info("HEALTH: status=%s http=%s latency=%sms detail=%s",
             res.status.value, res.http_code, res.latency_ms, res.detail)
    if res.status.value != "pass":
        log.error("Server never became healthy. If a server IS already running on this "
                  "port from another loader instance, stop it and re-run this script.")
        try:
            mgr.stop()
        except Exception:
            pass
        return 1

    log.info("Health OK -- running speed test (endpoint fallback chain) ...")
    try:
        msg = speed_test(port)
    except Exception as exc:
        log.error("SPEED TEST FAILED: %s", exc)
        try:
            mgr.stop()
        except Exception:
            pass
        return 1

    log.info("SPEED TEST: %s", msg)
    status = "Smoke passed (tokenize only)" if "tokenize" in msg else "Smoke passed"
    log.info("=== SMOKE TEST PASSED (headless): %s ===", status)

    # Stop the server we started so we don't leave a stray process.
    try:
        mgr.stop()
        log.info("Stopped the test server.")
    except Exception as exc:
        log.warning("Could not stop server (already gone?): %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
