"""Live smoke test: verify server health + measure tokens/s with a fixed prompt.

Uses the project's SmokeTestRunner to check 8080 server health,
then sends a fixed prompt and measures generation speed (tokens/s).
"""

from __future__ import annotations

import sys
import time
import json
import socket
from pathlib import Path

# Add project root to path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llamacpp_loader.smoke_test.runner import SmokeTestRunner, SmokeResult, SmokeTestResult


def main() -> None:
    # --- Phase 1: Health check ---
    runner = SmokeTestRunner(host="127.0.0.1", port=8080, timeout=30.0)
    result = runner.run()

    print("=== Smoke Test Result ===")
    print(f"Status: {result.status.value}")
    if result.http_code is not None:
        print(f"HTTP Code: {result.http_code}")
    if result.latency_ms is not None:
        print(f"Latency: {result.latency_ms} ms")
    print(f"Detail: {result.detail}")

    if result.status != SmokeResult.PASS:
        print("ERROR: Server health check failed. Aborting.")
        sys.exit(1)

    # --- Phase 2: Send a real generation request and measure tokens/s ---
    prompt = "Introduce yourself in 50 words"
    gen_start = time.time()
    generated_tokens = None
    detail = ""

    try:
        body = json.dumps({
            "prompt": prompt,
            "n_predict": 64,
            "temperature": 0.0,
            "stream": False,
        }).encode("utf-8")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        sock.connect(("127.0.0.1", 8080))

        # Real /v1/completions request with a JSON body and Content-Length.
        request = (
            "POST /v1/completions HTTP/1.1\r\n"
            "Host: 127.0.0.1:8080\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("utf-8") + body
        sock.sendall(request)

        # Read the full response until the server closes the connection.
        response = b""
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            response += chunk
        sock.close()

        gen_time = time.time() - gen_start

        if b"\r\n\r\n" not in response:
            detail = "No HTTP response received from /v1/completions"
        else:
            header_end = response.index(b"\r\n\r\n")
            body_bytes = response[header_end + 4:]
            try:
                data = json.loads(body_bytes.decode("utf-8", "replace"))
            except ValueError:
                data = None

            usage = (data or {}).get("usage") or {}
            if usage.get("completion_tokens"):
                generated_tokens = int(usage["completion_tokens"])
                detail = "from server usage.completion_tokens"
            elif data and data.get("choices"):
                text = data["choices"][0].get("text", "")
                # Rough heuristic: ~4 chars per token (labelled as an estimate).
                generated_tokens = max(1, len(text) // 4)
                detail = "estimated from generated text length (~4 chars/token)"

        print(f"\n=== Performance Measurement ===")
        print(f"Prompt: {prompt!r}")
        if generated_tokens is not None:
            print(f"Generated tokens: {generated_tokens} ({detail})")
            print(f"Generation time: {gen_time:.3f}s")
            if gen_time > 0:
                print(f"Tokens/s (real): {generated_tokens / gen_time:.1f}")
        else:
            print(f"Generation time: {gen_time:.3f}s")
            print(f"Note: {detail or 'could not parse a generation response'}")

    except Exception as e:
        print(f"\nError during performance measurement: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()