"""Zero-Downtime Hot Update Manager for Grid Control Kernels.

Implements safe controller updates:
1. Background staging of target controller
2. Warm-up evaluation on historical data (cache population & sanity verification)
3. Atomic swap at grid cycle boundary (< 100 µs duration)
4. Guard metric evaluation with auto-rollback if metrics degrade > 10%
"""
from enum import Enum
import time
from typing import Any, Callable, Optional
import numpy as np


class UpdateState(str, Enum):
    IDLE = "IDLE"
    STAGED = "STAGED"
    WARMED_UP = "WARMED_UP"
    ACTIVE = "ACTIVE"
    ROLLED_BACK = "ROLLED_BACK"


class HotUpdateManager:
    """Manages zero-downtime atomic swaps of control kernels."""

    def __init__(self, initial_controller: Any, version_tag: str = "v1.0.0") -> None:
        self.active_controller: Any = initial_controller
        self.active_version: str = version_tag

        self.staged_controller: Optional[Any] = None
        self.staged_version: Optional[str] = None

        self.previous_controller: Optional[Any] = None
        self.previous_version: Optional[str] = None

        self.state: UpdateState = UpdateState.IDLE
        self.switch_timestamp: float = 0.0
        self.last_switch_latency_us: float = 0.0

        # Baseline performance metric (e.g. latency, energy efficiency)
        self.baseline_metric: float = 0.0

    def stage_update(self, new_controller: Any, version_tag: str) -> None:
        """Stage new controller in background without interrupting active control."""
        self.staged_controller = new_controller
        self.staged_version = version_tag
        self.state = UpdateState.STAGED

    def warm_up(self, historical_loads: np.ndarray, n_ticks: int = 100) -> bool:
        """Run shadow evaluation to warm CPU caches and verify numerical sanity."""
        if self.staged_controller is None or self.state != UpdateState.STAGED:
            return False

        sample_slice = historical_loads[:n_ticks]
        try:
            # Perform shadow evaluations
            for load in sample_slice:
                if hasattr(self.staged_controller, "fuzzy_control"):
                    _ = self.staged_controller.fuzzy_control(load)
                elif hasattr(self.staged_controller, "fuzzy_logic"):
                    _ = self.staged_controller.fuzzy_logic(load)

            self.state = UpdateState.WARMED_UP
            return True
        except Exception:
            self.staged_controller = None
            self.staged_version = None
            self.state = UpdateState.IDLE
            return False

    def atomic_switch(self) -> bool:
        """Atomically swap the active controller with the staged controller at cycle boundary."""
        if self.state != UpdateState.WARMED_UP or self.staged_controller is None:
            return False

        t0 = time.perf_counter()

        # Atomic pointer swap
        self.previous_controller = self.active_controller
        self.previous_version = self.active_version

        self.active_controller = self.staged_controller
        self.active_version = str(self.staged_version)

        self.staged_controller = None
        self.staged_version = None

        dt_us = (time.perf_counter() - t0) * 1e6
        self.last_switch_latency_us = dt_us
        self.switch_timestamp = time.time()
        self.state = UpdateState.ACTIVE
        return True

    def guard_and_evaluate(
        self,
        metric_eval_fn: Callable[[Any], float],
        max_degradation_pct: float = 10.0,
    ) -> bool:
        """Evaluate active controller against performance metric; rollback if degraded > 10%."""
        if self.state != UpdateState.ACTIVE or self.previous_controller is None:
            return True

        current_val = metric_eval_fn(self.active_controller)
        baseline_val = metric_eval_fn(self.previous_controller)

        # Assuming higher is worse for latency/jitter, or lower is worse for efficiency
        # We check relative deviation
        if baseline_val > 0:
            degradation = ((current_val - baseline_val) / baseline_val) * 100.0
            if degradation > max_degradation_pct:
                self.rollback()
                return False

        return True

    def rollback(self) -> bool:
        """Revert immediately to previous controller."""
        if self.previous_controller is None:
            return False

        self.active_controller = self.previous_controller
        self.active_version = str(self.previous_version)
        self.previous_controller = None
        self.previous_version = None
        self.state = UpdateState.ROLLED_BACK
        return True

