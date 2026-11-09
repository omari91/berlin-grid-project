"""Tests for resilience circuit breakers and compliance audit logging."""
import os
import tempfile
import time
import pytest

from src.resilience import SolverCircuitBreaker, SCADACircuitBreaker, CircuitBreakerState
from src.observability import AuditLogger, AuditRecord


@pytest.mark.chaos
def test_solver_circuit_breaker_healthy_execution():
    """Verify solver circuit breaker executes healthy solver and caches last good state."""
    breaker = SolverCircuitBreaker(max_consecutive_failures=3)

    def healthy_solver():
        return {
            "voltage_min_pu": 0.98,
            "voltage_max_pu": 1.01,
            "trafo_loading_max_pct": 45.0,
            "line_loading_max_pct": 18.0,
        }

    res = breaker.execute(healthy_solver, load_mw=30.0)
    assert res["voltage_min_pu"] == 0.98
    assert res["fallback_active"] == 0.0
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.chaos
def test_solver_circuit_breaker_trips_after_max_failures():
    """Verify breaker transitions from CLOSED to OPEN after 3 consecutive failures."""
    breaker = SolverCircuitBreaker(max_consecutive_failures=3, recovery_timeout_s=0.5)

    def failing_solver():
        raise RuntimeError("Newton-Raphson solver diverged")

    # Failure 1
    res1 = breaker.execute(failing_solver, load_mw=40.0)
    assert res1["fallback_active"] == 1.0
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.consecutive_failures == 1

    # Failure 2
    res2 = breaker.execute(failing_solver, load_mw=40.0)
    assert res2["fallback_active"] == 1.0
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.consecutive_failures == 2

    # Failure 3 -> Trips to OPEN
    res3 = breaker.execute(failing_solver, load_mw=40.0)
    assert res3["fallback_active"] == 1.0
    assert breaker.state == CircuitBreakerState.OPEN
    assert breaker.is_open() is True

    # Call while OPEN: failing_solver is never called, fallback returned immediately
    called = []
    def spy_solver():
        called.append(True)
        return None

    res4 = breaker.execute(spy_solver, load_mw=40.0)
    assert len(called) == 0  # Did not invoke solver
    assert res4["fallback_active"] == 1.0


@pytest.mark.chaos
def test_solver_circuit_breaker_recovery_half_open():
    """Verify breaker transitions to HALF_OPEN after timeout and recovers to CLOSED on success."""
    breaker = SolverCircuitBreaker(max_consecutive_failures=2, recovery_timeout_s=0.2)

    # Force trip
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.OPEN

    # Wait for recovery timeout
    time.sleep(0.25)
    assert breaker.is_open() is False
    assert breaker.state == CircuitBreakerState.HALF_OPEN

    # Successful recovery call
    def recovered_solver():
        return {
            "voltage_min_pu": 0.99,
            "voltage_max_pu": 1.01,
            "trafo_loading_max_pct": 40.0,
            "line_loading_max_pct": 15.0,
        }

    res = breaker.execute(recovered_solver, load_mw=30.0)
    assert res["voltage_min_pu"] == 0.99
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.chaos
def test_scada_circuit_breaker_timeout_and_staleness():
    """Verify SCADA breaker degrades gracefully upon MMS timeout and flags staleness."""
    scada_breaker = SCADACircuitBreaker(timeout_ms=500.0, max_staleness_s=0.2)

    # Initial update
    scada_breaker.update_measurement({"TotW": 35.0, "PPV.phsA": 0.98})

    # Normal read
    assert scada_breaker.is_stale is False

    # First timeout: returns degraded cached telemetry
    degraded = scada_breaker.handle_timeout()
    assert degraded["_degraded"] is True
    assert degraded["_stale"] is False
    assert degraded["TotW"] == 35.0

    # Wait until staleness threshold exceeds
    time.sleep(0.25)
    stale_data = scada_breaker.handle_timeout()
    assert stale_data["_stale"] is True
    assert scada_breaker.is_stale is True

    # Reconnect: restores health
    scada_breaker.update_measurement({"TotW": 36.0, "PPV.phsA": 0.99})
    assert scada_breaker.is_stale is False
    assert scada_breaker.staleness_count == 0


@pytest.mark.chaos
def test_audit_logger_signature_and_tamper_detection():
    """Verify audit records have valid cryptographic SHA-256 signatures that detect tampering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = os.path.join(tmpdir, "audit.jsonl")
        logger = AuditLogger(log_filepath=logfile)

        record = logger.log_control_action(
            event_type="control_action",
            action="curtailment",
            reason="VDE_AR_N_4110_VOLTAGE_STABILITY",
            load_mw=45.0,
            curtailed_mw=38.887,
            voltage_pu=0.915,
            trafo_loading_pct=81.3,
            solver_confidence=0.99,
        )

        assert record.cryptographic_signature != ""
        # Valid signature verification
        assert AuditLogger.verify_record(record) is True

        # Tampering test: modify load_mw
        tampered_record = AuditRecord(
            timestamp=record.timestamp,
            event_type=record.event_type,
            action=record.action,
            reason=record.reason,
            load_mw=999.0,  # Tampered!
            curtailed_mw=record.curtailed_mw,
            voltage_pu=record.voltage_pu,
            trafo_loading_pct=record.trafo_loading_pct,
            solver_confidence=record.solver_confidence,
            cryptographic_signature=record.cryptographic_signature,
        )
        assert AuditLogger.verify_record(tampered_record) is False

        # File persistence check
        assert os.path.exists(logfile)
        with open(logfile, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "VDE_AR_N_4110_VOLTAGE_STABILITY" in lines[0]

