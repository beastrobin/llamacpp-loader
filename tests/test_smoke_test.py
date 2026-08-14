"""Tests for smoke_test.runner module."""

import pytest
from llamacpp_loader.smoke_test.runner import (
    SmokeTestRunner,
    SmokeTestResult,
    ServerHealthChecker,
    SmokeResult,
)


class TestSmokeTestResult:

    def test_pass_result(self):
        r = SmokeTestResult(
            status=SmokeResult.PASS,
            http_code=200,
            latency_ms=15.3,
            detail="Server healthy",
        )
        assert r.status == SmokeResult.PASS
        assert r.http_code == 200

    def test_timeout_result(self):
        r = SmokeTestResult(
            status=SmokeResult.TIMEOUT,
            latency_ms=None,
            detail="Timed out after 30s",
        )
        assert r.status == SmokeResult.TIMEOUT


class TestSmokeTestRunnerInit:

    def test_defaults(self):
        runner = SmokeTestRunner()
        assert runner._host == "127.0.0.1"
        assert runner._port == 8080
        assert runner._timeout == 30.0


class TestServerHealthChecker:

    def test_check_returns_tuple(self):
        checker = ServerHealthChecker()
        result = checker.check(port=9999)  # unlikely to respond
        assert isinstance(result, tuple)
        ok, detail = result
        assert isinstance(ok, bool)
        assert isinstance(detail, str)


class TestSmokeRunnerEndpoints:

    @pytest.mark.parametrize("endpoint", ["/health", "/v1/models"])
    def test_check_endpoint_returns_result(self, endpoint):
        runner = SmokeTestRunner(port=9999)
        result = runner._check_endpoint(endpoint)
        assert isinstance(result, SmokeTestResult)


class TestSmokeTestStates:

    def test_all_enum_values_exist(self):
        states = list(SmokeResult)
        assert len(states) == 4
        names = {s.value for s in states}
        assert "pass" in names
        assert "fail_http" in names
        assert "timeout" in names
        assert "conn_error" in names


class TestSmokeTestResultFields:

    def test_minimum_result(self):
        r = SmokeTestResult(status=SmokeResult.CONNECTION_ERROR, detail="refused")
        assert r.status == SmokeResult.CONNECTION_ERROR
        assert r.http_code is None
        assert r.latency_ms is None
