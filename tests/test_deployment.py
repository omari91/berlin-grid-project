"""Unit tests for Zero-Downtime Hot Update Manager (Stage B/C)."""
import pytest
import numpy as np
from src.core import Controller
from src.deployment import HotUpdateManager, UpdateState


def test_hot_update_lifecycle():
    """Verify state transitions: IDLE -> STAGED -> WARMED_UP -> ACTIVE."""
    c1 = Controller(limit_mw=45.0)
    mgr = HotUpdateManager(c1, version_tag="v1.0.0")

    assert mgr.state == UpdateState.IDLE
    assert mgr.active_version == "v1.0.0"

    # Cannot switch without staging and warming up
    assert mgr.atomic_switch() is False

    # Stage
    c2 = Controller(limit_mw=45.0)
    mgr.stage_update(c2, version_tag="v1.1.0")
    assert mgr.state == UpdateState.STAGED
    assert mgr.staged_version == "v1.1.0"

    # Warm up with data
    data = np.random.uniform(20.0, 50.0, 100)
    ok = mgr.warm_up(data, n_ticks=20)
    assert ok is True
    assert mgr.state == UpdateState.WARMED_UP

    # Switch
    swapped = mgr.atomic_switch()
    assert swapped is True
    assert mgr.state == UpdateState.ACTIVE
    assert mgr.active_version == "v1.1.0"
    assert mgr.active_controller == c2
    assert mgr.previous_controller == c1


def test_hot_update_guard_eval_passes_when_healthy():
    """Verify guard evaluation maintains active version when degradation is within bounds."""
    c1 = Controller(limit_mw=45.0)
    c2 = Controller(limit_mw=45.0)
    mgr = HotUpdateManager(c1, version_tag="v1.0.0")

    mgr.stage_update(c2, version_tag="v1.1.0")
    mgr.warm_up(np.array([30.0, 35.0]))
    mgr.atomic_switch()

    # Metric within bounds (e.g. 5.1 vs 5.0 -> +2% deviation, <10% max)
    eval_fn = lambda ctrl: 5.1 if ctrl == c2 else 5.0
    res = mgr.guard_and_evaluate(eval_fn, max_degradation_pct=10.0)

    assert res is True
    assert mgr.state == UpdateState.ACTIVE
    assert mgr.active_version == "v1.1.0"


def test_manual_rollback():
    """Verify manual rollback reverts to previous controller."""
    c1 = Controller(limit_mw=45.0)
    c2 = Controller(limit_mw=45.0)
    mgr = HotUpdateManager(c1, version_tag="v1.0.0")

    mgr.stage_update(c2, version_tag="v1.1.0")
    mgr.warm_up(np.array([30.0]))
    mgr.atomic_switch()

    assert mgr.rollback() is True
    assert mgr.state == UpdateState.ROLLED_BACK
    assert mgr.active_version == "v1.0.0"
    assert mgr.active_controller == c1

    # Second rollback fails (no older version cached)
    assert mgr.rollback() is False

