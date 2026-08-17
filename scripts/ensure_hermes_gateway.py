"""Ensure the Hermes gateway daemon is running (idempotent).

Used as a pre-launch command by llamacpp-loader's Agent Launch:
  * If `hermes gateway status` reports the gateway is running, do nothing.
  * Otherwise spawn `hermes gateway run` detached (no console window) so
    Telegram/Discord/Weixin polling starts in the background.

Exit code 0 in all normal cases so the loader always proceeds to launch the
main agent.
"""

import os
import subprocess
import sys

HERMES_EXE = r"C:\Users\qiaoj\AI\.hermes-venv-new\Scripts\hermes.exe"
GATEWAY_LOG_DIR = r"C:\Users\qiaoj\AppData\Local\hermes\logs"

# Clean NODE_OPTIONS: WorkBuddy injects a safe-delete hook via NODE_OPTIONS
# which breaks child processes started from this machine's tooling.
ENV = dict(os.environ)
ENV.pop("NODE_OPTIONS", None)


def gateway_running() -> bool:
    """Return True when `hermes gateway status` says the gateway is up."""
    try:
        proc = subprocess.run(
            [HERMES_EXE, "gateway", "status"],
            capture_output=True, text=True, timeout=30, env=ENV,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        return "is running" in combined.lower()
    except Exception as exc:  # noqa: BLE001
        print(f"[ensure_gateway] status check failed: {exc}", file=sys.stderr)
        return False


def start_gateway_detached() -> None:
    """Launch `hermes gateway run` fully detached (no console window)."""
    os.makedirs(GATEWAY_LOG_DIR, exist_ok=True)
    out_log = os.path.join(GATEWAY_LOG_DIR, "gateway-run-manual.log")
    err_log = out_log + ".err"
    flags = 0
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        flags = 0x00000200 | 0x00000008
    with open(out_log, "a", encoding="utf-8") as out_f, \
            open(err_log, "a", encoding="utf-8") as err_f:
        subprocess.Popen(
            [HERMES_EXE, "gateway", "run"],
            env=ENV, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
            creationflags=flags, close_fds=True,
        )
    print("[ensure_gateway] hermes gateway run spawned (detached)")


def main() -> None:
    if gateway_running():
        print("[ensure_gateway] gateway already running - skip")
        return
    start_gateway_detached()


if __name__ == "__main__":
    main()
