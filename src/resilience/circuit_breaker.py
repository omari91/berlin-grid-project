"""Circuit Breaker & Fallback System for Berlin Grid Digital Twin.

Protects against:
1. Divergent or hanging power flow solvers
2. SCADA MMS communication timeouts and staleness
3. Cascading control failure modes
"""
from enum import Enum
import time
from typing import Any, Callable, Optional


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"        # Normal healthy operation
    OPEN = "OPEN"            # Tripped; fallback active
    HALF_OPEN = "HALF_OPEN"  # Probing recovery


class SolverCircuitBreaker:
    """Circuit breaker guarding against AC power flow non-convergence and numerical instabilities."""

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        recovery_timeout_s: float = 5.0,
        default_voltage_pu: float = 1.00,
        default_limit_mw: float = 45.0,
    ) -> None:
        self.max_failures = max_consecutive_failures
        self.recovery_timeout = recovery_timeout_s
        self.state: CircuitBreakerState = CircuitBreakerState.CLOSED

        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0
        self.last_good_state: dict[str, float] = {
            "voltage_min_pu": default_voltage_pu,
            "voltage_max_pu": default_voltage_pu,
            "trafo_loading_max_pct": 50.0,
            "line_loading_max_pct": 20.0,
        }
        self.limit_mw: float = default_limit_mw

    def is_open(self) -> bool:
        """Check if circuit breaker is open or eligible to probe recovery (half-open)."""
        if self.state == CircuitBreakerState.OPEN:
            if (time.time() - self.last_failure_time) >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return False
            return True
        return False

    def record_success(self, results: dict[str, float]) -> None:
        """Record successful power flow computation and reset breaker."""
        self.last_good_state = results.copy()
        self.consecutive_failures = 0
        self.state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        """Record solver divergence or failure."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.max_failures:
            self.state = CircuitBreakerState.OPEN

    def safe_fallback(self, load_mw: float) -> dict[str, float]:
        """Generate safe fallback state under solver outage:
        
        1. Employs conservative curtailment (80% of normal limit).
        2. Returns last-known-good voltage profile.
        3. Asserts elevated conservative transformer loading.
        """
        fallback_curtailment = min(load_mw, self.limit_mw * 0.80)
        return {
            "voltage_min_pu": self.last_good_state.get("voltage_min_pu", 0.95),
            "voltage_max_pu": self.last_good_state.get("voltage_max_pu", 1.02),
            "trafo_loading_max_pct": 85.0,  # Conservative buffer
            "line_loading_max_pct": 75.0,
            "curtailed_setpoint_mw": fallback_curtailment,
            "fallback_active": 1.0,
        }

    def execute(
        self,
        solver_func: Callable[..., Optional[dict[str, float]]],
        load_mw: float,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Execute solver with fallback and circuit breaking."""
        if self.is_open():
            return self.safe_fallback(load_mw)

        try:
            results = solver_func(*args, **kwargs)
            if results is None:
                self.record_failure()
                return self.safe_fallback(load_mw)

            # Sanity checks on physical realism
            v_min = results.get("voltage_min_pu", 0.0)
            if v_min < 0.50 or v_min > 1.50:
                self.record_failure()
                return self.safe_fallback(load_mw)

            self.record_success(results)
            res_dict = results.copy()
            res_dict["fallback_active"] = 0.0
            return res_dict

        except Exception:
            self.record_failure()
            return self.safe_fallback(load_mw)


class SCADACircuitBreaker:
    """Circuit breaker handling SCADA MMS telemetry dropouts and staleness."""

    def __init__(
        self,
        timeout_ms: float = 500.0,
        max_staleness_s: float = 30.0,
    ) -> None:
        self.timeout_ms: float = timeout_ms
        self.max_staleness: float = max_staleness_s
        self.last_good_measurements: dict[str, Any] = {
            "TotW": 30.0,
            "TotVAr": 9.0,
            "PPV.phsA": 1.0,
            "A.phsA": 0.0,
        }
        self.last_update_time: float = time.time()
        self.staleness_count: int = 0
        self.is_stale: bool = False

    def update_measurement(self, measurements: dict[str, Any]) -> None:
        """Update last good measurement on successful MMS cycle."""
        self.last_good_measurements = measurements.copy()
        self.last_update_time = time.time()
        self.staleness_count = 0
        self.is_stale = False

    def handle_timeout(self) -> dict[str, Any]:
        """Gracefully handle MMS read timeout by returning cached telemetry with degraded flag."""
        self.staleness_count += 1
        elapsed = time.time() - self.last_update_time
        if elapsed > self.max_staleness:
            self.is_stale = True

        cached = self.last_good_measurements.copy()
        cached["_degraded"] = True
        cached["_stale"] = self.is_stale
        cached["_staleness_count"] = self.staleness_count
        return cached

