"""Chaos Engineering & Automated Fault Injection Test Suite (Phase 1B).

Verifies resilience against 3 primary distributed production failure modes:
1. SCADA MMS offline / timeout for prolonged periods (60s simulation)
2. Network partition (split-brain prevention, quorum consensus, auto-rejoin)
3. Zero-downtime hot update mid-cycle with auto-rollback protection
"""
import time
import pytest
import numpy as np

from src.core import Controller
from src.resilience import SCADACircuitBreaker, SolverCircuitBreaker, CircuitBreakerState
from src.distribution import DistributedCoordinator, NodeRole
from src.deployment import HotUpdateManager, UpdateState


@pytest.mark.chaos
def test_scada_offline_60s():
    """Verify system survives 60-second SCADA dropout using cached telemetry and conservative mode."""
    breaker = SCADACircuitBreaker(timeout_ms=500.0, max_staleness_s=10.0)
    breaker.update_measurement({"TotW": 34.0, "PPV.phsA": 0.99, "TotVAr": 10.2})

    ctrl = Controller(limit_mw=45.0)

    # Simulate 60 time ticks under SCADA disconnect
    degraded_records = []
    control_setpoints = []

    for tick in range(60):
        # MMS timeout occurs
        meas = breaker.handle_timeout()
        degraded_records.append(meas)

        # In degraded mode, apply conservative curtailment
        load = meas["TotW"]
        setpoint = ctrl.fuzzy_control(load)
        control_setpoints.append(setpoint)

    assert len(degraded_records) == 60
    assert all(r["_degraded"] is True for r in degraded_records)
    # Beyond staleness threshold (10s), marked stale
    assert degraded_records[-1]["_staleness_count"] == 60
    # Safe control setpoints maintained throughout
    assert all(s <= 45.0 for s in control_setpoints)


@pytest.mark.chaos
def test_network_partition_then_heal():
    """Verify 3-node cluster handles partition without split-brain and auto-resyncs on healing."""
    nodes = ["Berlin-01", "Berlin-02", "Berlin-03"]
    b1 = DistributedCoordinator("Berlin-01", nodes, detection_timeout_s=0.2, anti_flapping_window_s=0.1)
    b2 = DistributedCoordinator("Berlin-02", nodes, detection_timeout_s=0.2, anti_flapping_window_s=0.1)
    b3 = DistributedCoordinator("Berlin-03", nodes, detection_timeout_s=0.2, anti_flapping_window_s=0.1)

    # Initial full mesh sync
    for _ in range(2):
        hb1 = b1.generate_heartbeat()
        hb2 = b2.generate_heartbeat()
        hb3 = b3.generate_heartbeat()

        b1.receive_heartbeat(hb2); b1.receive_heartbeat(hb3)
        b2.receive_heartbeat(hb1); b2.receive_heartbeat(hb3)
        b3.receive_heartbeat(hb1); b3.receive_heartbeat(hb2)

    # Berlin-01 (lowest ID) is PRIMARY
    assert b1.role == NodeRole.PRIMARY
    assert b2.role == NodeRole.STANDBY
    assert b3.role == NodeRole.STANDBY

    # PARTITION EVENT: Berlin-03 is disconnected from Berlin-01 & Berlin-02
    time.sleep(0.3)  # Exceed detection timeout (0.2s)

    # Only b1 and b2 exchange heartbeats
    hb1 = b1.generate_heartbeat()
    hb2 = b2.generate_heartbeat()
    b1.receive_heartbeat(hb2)
    b2.receive_heartbeat(hb1)

    # b3 receives nothing
    b3.evaluate_quorum_and_role()

    # Quorum check:
    # b1 and b2 have 2/3 nodes -> Majority quorum maintained
    assert b1.role == NodeRole.PRIMARY
    assert b2.role == NodeRole.STANDBY

    # b3 has 1/3 nodes -> Minority partition! Must drop to ISOLATED_LOCAL (no split-brain primary)
    assert b3.role == NodeRole.ISOLATED_LOCAL
    assert b3.current_primary is None

    # HEAL EVENT: Network reconnects, full mesh resumes
    time.sleep(0.15)
    hb1 = b1.generate_heartbeat()
    hb2 = b2.generate_heartbeat()
    hb3 = b3.generate_heartbeat()

    b1.receive_heartbeat(hb3)
    b2.receive_heartbeat(hb3)
    b3.receive_heartbeat(hb1)
    b3.receive_heartbeat(hb2)

    # b3 rejoins cluster as STANDBY without conflict
    assert b3.role == NodeRole.STANDBY
    assert b3.current_primary == "Berlin-01"


@pytest.mark.chaos
def test_hot_update_mid_cycle_and_rollback():
    """Verify zero-downtime hot update swaps within < 100 µs and triggers auto-rollback on metric degradation."""
    initial_ctrl = Controller(limit_mw=45.0)
    manager = HotUpdateManager(initial_ctrl, version_tag="v1.0.0")

    # Synthetic streaming data
    historical_loads = np.random.uniform(20.0, 50.0, 200)

    # 1. Stage a new controller
    staged_ctrl = Controller(limit_mw=45.0)
    manager.stage_update(staged_ctrl, version_tag="v1.1.0-candidate")
    assert manager.state == UpdateState.STAGED

    # 2. Warm up
    success = manager.warm_up(historical_loads, n_ticks=50)
    assert success is True
    assert manager.state == UpdateState.WARMED_UP

    # 3. Mid-cycle atomic switch
    switch_ok = manager.atomic_switch()
    assert switch_ok is True
    assert manager.state == UpdateState.ACTIVE
    assert manager.active_version == "v1.1.0-candidate"
    # Verify swap was extremely fast (< 100 µs)
    assert manager.last_switch_latency_us < 100.0

    # 4. Guard test: Simulate degraded performance metric (e.g. latency spiked by 50%)
    def degraded_metric_eval(ctrl):
        if ctrl == staged_ctrl:
            return 15.0  # Degraded latency (15 µs vs baseline 5 µs)
        return 5.0

    guarded = manager.guard_and_evaluate(degraded_metric_eval, max_degradation_pct=10.0)
    # Rollback must be triggered!
    assert guarded is False
    assert manager.state == UpdateState.ROLLED_BACK
    assert manager.active_version == "v1.0.0"
    assert manager.active_controller == initial_ctrl
