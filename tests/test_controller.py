"""Unit tests for Fuzzy Logic Controller and Vectorized Soft Sigmoid Control."""
import pytest
import numpy as np
from src.core import Controller as CoreController
from main import Controller as MainController


@pytest.fixture
def core_ctrl():
    return CoreController(limit_mw=45.0)


@pytest.fixture
def main_ctrl():
    return MainController(limit=45.0)


def test_fuzzy_control_scalar_types(core_ctrl):
    """Verify scalar inputs return Python floats."""
    val = core_ctrl.fuzzy_control(30.0)
    assert isinstance(val, float)
    assert np.isclose(val, 29.9156, atol=0.01)


def test_fuzzy_control_vectorized_types(core_ctrl):
    """Verify numpy array inputs return numpy ndarrays."""
    arr = np.array([10.0, 30.0, 45.0, 55.0])
    res = core_ctrl.fuzzy_control(arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == arr.shape


def test_vectorized_matches_scalar_elementwise(core_ctrl):
    """Verify vectorized call returns identical values to element-wise scalar loop."""
    loads = np.linspace(0.0, 70.0, 100)
    vectorized_res = core_ctrl.fuzzy_control(loads)

    scalar_res = np.array([core_ctrl.fuzzy_control(float(l)) for l in loads])
    assert np.allclose(vectorized_res, scalar_res, rtol=1e-5, atol=1e-8)


def test_dimming_factor_bounds(core_ctrl):
    """Verify that dimming factor is strictly between 0.80 and 1.00 (max 20% curtailment)."""
    # Test very high stress (load >> limit)
    high_load = 500.0  # stress > 10
    out = core_ctrl.fuzzy_control(high_load)
    # Output should be capped at limit_mw
    assert out <= core_ctrl.limit
    # Dimming factor component is max 20%
    stress = high_load / core_ctrl.limit
    k = 15.0
    s_ref = 0.95
    activation = 1.0 / (1.0 + np.exp(-k * (stress - s_ref)))
    dimming = 1.0 - (activation * 0.20)
    assert 0.80 <= dimming <= 1.00


def test_monotonic_curtailment(core_ctrl):
    """Verify curtailment percentage increases monotonically as stress increases."""
    stresses = np.linspace(0.5, 1.5, 50)
    curtailment_pcts = []
    for s in stresses:
        load = s * core_ctrl.limit
        out = core_ctrl.fuzzy_control(load)
        curt_pct = (load - out) / load
        curtailment_pcts.append(curt_pct)

    # Check non-decreasing
    diffs = np.diff(curtailment_pcts)
    assert (diffs >= -1e-12).all(), "Curtailment curve is not monotonic increasing"


def test_hard_limit_enforcement(core_ctrl):
    """Verify controller output never exceeds the physical transformer limit."""
    extreme_loads = np.array([45.0, 45.1, 50.0, 100.0, 1000.0])
    res = core_ctrl.fuzzy_control(extreme_loads)
    assert (res <= core_ctrl.limit + 1e-9).all()


def test_zero_and_negative_loads(core_ctrl):
    """Verify zero and edge case loads are handled gracefully."""
    assert core_ctrl.fuzzy_control(0.0) == 0.0
    # Negative load (e.g. reverse power flow from local solar)
    neg_res = core_ctrl.fuzzy_control(-10.0)
    assert neg_res < 0.0


def test_controller_ablation_soft_vs_hard(core_ctrl):
    """Verify fuzzy controller provides smooth continuous response vs hard cutoff discontinuity."""
    loads = np.linspace(40.0, 50.0, 50)
    fuzzy_out = core_ctrl.fuzzy_control(loads)
    hard_out = core_ctrl.hard_cutoff(loads)

    # Hard cutoff has a sharp gradient change at limit=45.0
    hard_diff = np.diff(hard_out)
    # After 45 MW, hard cutoff diff is exactly 0
    assert np.allclose(hard_diff[loads[:-1] >= 45.0], 0.0)

    # Fuzzy logic has smooth, non-zero gradient across the threshold
    fuzzy_diff = np.diff(fuzzy_out)
    assert (fuzzy_diff > 0).all(), "Fuzzy controller should maintain smooth non-zero derivative"


def test_main_and_core_controller_parity(core_ctrl, main_ctrl):
    """Verify src/core.py Controller and main.py Controller behave identically."""
    loads = np.array([10.0, 25.0, 42.0, 45.0, 50.0, 60.0])
    core_out = core_ctrl.fuzzy_control(loads)
    main_out = main_ctrl.fuzzy_logic(loads)
    assert np.allclose(core_out, main_out)
