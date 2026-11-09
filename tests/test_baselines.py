"""Phase 1: Baseline Regression Tests.

Ensures result preservation across all refactoring and verification phases.
Tolerance bands:
- Physics voltage: ±0.5%
- Transformer loading: ±1.0%
- Controller output: ±1.0%
- AR(1) lag-1 autocorrelation: ±5.0%
- IEC 61850 protocol transitions: exact match
"""
import json
import os
import pytest
import numpy as np

from src.core import PhysicalTwin as CoreTwin, Controller as CoreCtrl, StochasticEngine
from src.iec61850_layer import IEC61850Station
from main import PhysicalTwin as MainTwin, Controller as MainCtrl


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "verification_baseline.json")


@pytest.fixture(scope="module")
def baseline():
    """Load golden reference verification baseline."""
    with open(FIXTURE_PATH, "r") as f:
        return json.load(f)


@pytest.mark.regression
def test_physics_core_baseline(baseline):
    """Verify src/core.py PhysicalTwin matches power flow baseline within ±0.5% / ±1.0%."""
    twin = CoreTwin()
    assert twin.net is not None, "Physical network model failed to initialize"

    # Test load points
    test_cases = [
        ("load_0.1mw", 0.0, 0.1),
        ("load_0.5mw", 0.2, 0.5),
        ("load_1.0mw", 0.0, 1.0),
    ]

    for key, gen, load in test_cases:
        res = twin.run_power_flow(gen, load)
        assert res is not None, f"Power flow failed for {key}"
        expected = baseline["physics_core"][key]

        assert np.isclose(res["voltage_min_pu"], expected["voltage_min_pu"], rtol=0.005), (
            f"Voltage min drift in {key}: got {res['voltage_min_pu']}, expected {expected['voltage_min_pu']}"
        )
        assert np.isclose(res["voltage_max_pu"], expected["voltage_max_pu"], rtol=0.005), (
            f"Voltage max drift in {key}: got {res['voltage_max_pu']}, expected {expected['voltage_max_pu']}"
        )
        assert np.isclose(res["trafo_loading_max_pct"], expected["trafo_loading_max_pct"], rtol=0.01), (
            f"Trafo loading drift in {key}: got {res['trafo_loading_max_pct']}, expected {expected['trafo_loading_max_pct']}"
        )


@pytest.mark.regression
def test_physics_main_baseline(baseline):
    """Verify main.py PhysicalTwin matches power flow baseline at key loads (10, 30, 45, 55 MW)."""
    twin = MainTwin(feeder_length_km=5.0)

    for load in [10.0, 30.0, 45.0, 55.0]:
        key = f"load_{load}mw"
        expected = baseline["physics_main"][key]
        loading_pct, v_min = twin.step(load)

        assert np.isclose(v_min, expected["voltage_min_pu"], rtol=0.005), (
            f"Voltage min drift for {load}MW: got {v_min}, expected {expected['voltage_min_pu']}"
        )
        assert np.isclose(loading_pct, expected["trafo_loading_pct"], rtol=0.01), (
            f"Trafo loading drift for {load}MW: got {loading_pct}, expected {expected['trafo_loading_pct']}"
        )


@pytest.mark.regression
def test_controller_baseline(baseline):
    """Verify sigmoid controller output matches baseline within ±1.0% across all stress levels."""
    ctrl = CoreCtrl(limit_mw=45.0)

    for stress_key, expected in baseline["controller"].items():
        load_mw = expected["load_mw"]
        curtailed = ctrl.fuzzy_control(load_mw)
        if load_mw > 0:
            assert np.isclose(curtailed, expected["curtailed_mw"], rtol=0.01), (
                f"Controller drift for {stress_key}: got {curtailed}, expected {expected['curtailed_mw']}"
            )
        else:
            assert np.isclose(curtailed, 0.0, atol=1e-6)


@pytest.mark.regression
def test_stochastic_baseline(baseline):
    """Verify AR(1) noise generator reproduces statistics and lag-1 autocorrelation with fixed seed."""
    np.random.seed(baseline["stochastic"]["seed"])
    engine = StochasticEngine()
    samples = engine.generate_ar1_noise(
        n=baseline["stochastic"]["n_samples"],
        sigma=baseline["stochastic"]["sigma"],
        phi=baseline["stochastic"]["phi"]
    )

    lag1 = np.corrcoef(samples[:-1], samples[1:])[0, 1]
    assert np.isclose(lag1, baseline["stochastic"]["lag1_autocorr"], rtol=0.05), (
        f"AR(1) lag-1 autocorr drift: got {lag1}, expected {baseline['stochastic']['lag1_autocorr']}"
    )
    assert np.isclose(np.mean(samples), baseline["stochastic"]["mean"], atol=0.2)
    assert np.isclose(np.std(samples), baseline["stochastic"]["std"], rtol=0.05)


@pytest.mark.regression
def test_iec61850_baseline(baseline):
    """Verify IEC 61850 MMS and GOOSE trip logic matches baseline state transitions."""
    expected = baseline["iec61850"]
    station = IEC61850Station()

    # Normal voltage cycle (0.95 pu) -> No trip
    mms_norm, tripped_norm = station.process_scada_cycle(load_mw=30.0, voltage_pu=expected["normal_voltage_pu"])
    assert tripped_norm == expected["normal_tripped"]
    assert station.xcbr.data_objects["Pos.stVal"] is True
    for field in expected["scada_cycle_fields"]:
        assert field in mms_norm

    # Undervoltage cycle (0.88 pu) -> GOOSE trip (<0.90 pu)
    _, tripped_under = station.process_scada_cycle(load_mw=55.0, voltage_pu=expected["under_voltage_pu"])
    assert tripped_under == expected["under_tripped"]
    assert station.xcbr.data_objects["Pos.stVal"] == expected["breaker_pos_after_trip"]
    assert len(station.protection_log) == 1

