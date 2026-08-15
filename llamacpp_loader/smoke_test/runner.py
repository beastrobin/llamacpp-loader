"""Smoke test runner: verifies a launched llama.cpp server is healthy."""

from __future__ import annotations

import logging
import socket
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- enums / data


class SmokeResult(Enum):
    """Outcome of a smoke test run."""
    PASS = "pass"
    FAIL_HTTP = "fail_http"        # non-200 response
    TIMEOUT = "timeout"            # didn\'t respond within timeout
    CONNECTION_ERROR = "conn_error"  # connection refused / network error


@dataclass(slots=True)
class SmokeTestResult:
    """Detailed result from a single smoke test execution.

    Attributes:
        status:      PASS, FAIL_HTTP, TIMEOUT, or CONNECTION_ERROR.
        http_code:   HTTP response code (None on timeout/connection error).
        latency_ms:  Time to first byte in milliseconds (or None).
        detail:      Human-readable description of the result.
    """
    status: SmokeResult
    http_code: Optional[int] = None
    latency_ms: Optional[float] = None
    detail: str = ""


# --------------------------------------------------------------------------- runner


class SmokeTestRunner:
    """Runs health checks against a running llama.cpp server.

    Health check endpoints (in order of preference)::
        1. GET /health          -> expects HTTP 200 with JSON body
        2. GET /v1/models       -> expects HTTP 200 with models list (OpenAI compat)

    Usage::
        runner = SmokeTestRunner(host="127.0.0.1", port=8080, timeout=30)
        result = runner.run()   # blocks until PASS/FAIL or TIMEOUT
        if result.status == SmokeResult.PASS:
            print("Server is healthy!")

    Thread safety::
        All methods are safe to call from any thread. Uses raw socket + HTTP
        parsing to avoid external dependencies (no requests/httpx needed).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, *,
                 timeout: float = 30.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    def run(self) -> SmokeTestResult:
        """Execute the full smoke test sequence.

        Tries /health first, falls back to /v1/models on 404.
        Returns a SmokeTestResult with status and timing details.
        """
        start_time = time.monotonic()
        self._start_time = start_time

        # Try health endpoint first
        result = self._check_endpoint("/health")
        if result.status == SmokeResult.PASS:
            return result

        # If we got a 404, try the OpenAI-compatible endpoint
        if result.status == SmokeResult.FAIL_HTTP and result.http_code == 404:
            result = self._check_endpoint("/v1/models")
            return result

        return result

    def wait_until_ready(self, *, poll_interval: float = 0.5) -> SmokeTestResult:
        """Poll the server periodically until it responds or timeout.

        Checks every ``poll_interval`` seconds until either a PASS result
        is returned or the total elapsed time exceeds self._timeout.

        Returns the first non-timeout result, or a TIMEOUT result if
        the server never becomes reachable within the timeout window.
        """
        start_time = time.monotonic()
        self._start_time = start_time

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._timeout:
                return SmokeTestResult(
                    status=SmokeResult.TIMEOUT,
                    detail=f"Timed out after {self._timeout}s waiting for server",
                )

            result = self.run()
            if result.status == SmokeResult.PASS:
                return result

            # Give server time to start up before retrying
            time.sleep(poll_interval)

    def _elapsed(self) -> float:
        """Return seconds since the smoke test started."""
        return time.monotonic() - getattr(self, "_start_time", time.monotonic())

    def _check_endpoint(self, endpoint: str) -> SmokeTestResult:
        """Check a single HTTP endpoint.

        Sends raw HTTP GET request to http://<host>:<port><endpoint>.
        Measures response time in milliseconds.
        Returns PASS if status code is 200, FAIL_HTTP otherwise.
        Catches connection errors and converts them to CONNECTION_ERROR results.
        """
        start = time.monotonic()

        try:
            # Create raw TCP socket (no external dependencies)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(min(self._timeout, 5.0))  # per-operation timeout
            sock.connect((self._host, self._port))

            # Send HTTP request
            request = f"GET {endpoint} HTTP/1.1\r\nHost: {self._host}:{self._port}\r\nConnection: close\r\n\r\n"
            sock.sendall(request.encode("utf-8"))

            # Receive response headers (first line contains status)
            response = b""
            while b"\r\n\r\n" not in response:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk

            latency_ms = (time.monotonic() - start) * 1000.0
            sock.close()

            # Parse HTTP status line
            if response:
                if b"\r\n\r\n" not in response:
                    return SmokeTestResult(
                        status=SmokeResult.CONNECTION_ERROR,
                        latency_ms=round(latency_ms),
                        detail="Server closed connection before sending headers",
                    )
                header_end = response.index(b"\r\n\r\n")
                headers = response[:header_end].decode("utf-8", errors="replace")
                status_line = headers.split("\r\n")[0]  # "HTTP/1.1 200 OK"

                try:
                    http_code = int(status_line.split()[1])
                except (IndexError, ValueError):
                    return SmokeTestResult(
                        status=SmokeResult.FAIL_HTTP,
                        latency_ms=latency_ms,
                        detail=f"Parsed incomplete status line: {status_line}",
                    )

                if http_code == 200:
                    return SmokeTestResult(
                        status=SmokeResult.PASS,
                        http_code=http_code,
                        latency_ms=round(latency_ms),
                        detail="Endpoint responded with HTTP 200",
                    )
                else:
                    return SmokeTestResult(
                        status=SmokeResult.FAIL_HTTP,
                        http_code=http_code,
                        latency_ms=round(latency_ms),
                        detail=f"HTTP {http_code}",
                    )

            # No response at all (empty headers)
            return SmokeTestResult(
                status=SmokeResult.CONNECTION_ERROR,
                latency_ms=round(latency_ms),
                detail="Empty response from server",
            )

        except socket.timeout:
            elapsed = round((time.monotonic() - start) * 1000.0)
            return SmokeTestResult(
                status=SmokeResult.TIMEOUT,
                latency_ms=elapsed,
                detail=f"Connection timed out after {elapsed}ms",
            )

        except (socket.error, OSError):
            elapsed = round((time.monotonic() - start) * 1000.0)
            return SmokeTestResult(
                status=SmokeResult.CONNECTION_ERROR,
                latency_ms=elapsed,
                detail="Connection refused or network error",
            )


# --------------------------------------------------------------------------- health checker


class ServerHealthChecker:
    """Lightweight synchronous checker for integration tests.

    Unlike SmokeTestRunner, this does NOT use threads or timers.
    It makes a single blocking request and returns immediately.

    Used by pytest to verify the process manager starts a healthy server.
    """

    def check(self, host: str = "127.0.0.1", port: int = 8080) -> tuple[bool, str]:
        """Single health check. Returns (is_ok: bool, detail: str)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((host, port))

            request = f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
            sock.sendall(request.encode("utf-8"))

            response = b""
            while b"\r\n\r\n" not in response:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk

            sock.close()

            if response:
                headers = response[:response.index(b"\r\n\r\n")].decode("utf-8", errors="replace")
                status_line = headers.split("\r\n")[0]
                code = int(status_line.split()[1])
                return (code == 200, f"HTTP {code}" if code != 200 else "OK")

            return (False, "Empty response")

        except socket.timeout:
            return (False, "Connection timed out")
        except (socket.error, OSError) as exc:
            return (False, str(exc))


# --------------------------------------------------------------------------- CLI helper

def main() -> None:  # pragma: no cover
    """Quick smoke test from command line."""
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    runner = SmokeTestRunner(timeout=5.0)
    result = runner.run()
    print(f"Result: {result.status.value} - {result.detail}")


if __name__ == "__main__":
    main()

