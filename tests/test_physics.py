"""Physics Model & AC Power Flow Verification Test Suite (pandapower Newton-Raphson)."""
import pytest
import numpy as np

from src.core import PhysicalTwin as CorePhysicalTwin
from main import PhysicalTwin as MainPhysicalTwin, PANDAPOWER_AVAILABLE


@pytest.fixture
def core_twin():
    return CorePhysicalTwin()


@pytest.fixture
def main_twin():
    return MainPhysicalTwin(feeder_length_km=5.0)


def test_core_network_topology(core_twin):
    """Verify core physical network has expected buses, lines, and transformers."""
    assert core_twin.net is not None
    # 5 buses: External Grid, 2 MV feeder buses, 2 LV load buses
    assert len(core_twin.net.bus) == 5
    assert len(core_twin.net.line) == 2
    assert len(core_twin.net.trafo) == 2
    assert len(core_twin.net.load) == 2
    assert len(core_twin.net.sgen) == 2


def test_main_network_topology(main_twin):
    """Verify main physical network has HV, MV, and Aggregated Load buses."""
    assert main_twin.net is not None
    assert len(main_twin.net.bus) == 3
    assert len(main_twin.net.trafo) == 1
    assert len(main_twin.net.line) == 1
    assert len(main_twin.net.load) == 1


def test_core_power_flow_load_sweep(core_twin):
    """Verify power flow converges across normal load range and voltages stay within bounds."""
    for load_mw in [0.0, 0.2, 0.5, 0.8, 1.0]:
        res = core_twin.run_power_flow(gen_mw=0.1, load_mw=load_mw)
        assert res is not None, f"Power flow failed to converge for load={load_mw} MW"
        assert 0.90 <= res["voltage_min_pu"] <= 1.10
        assert 0.90 <= res["voltage_max_pu"] <= 1.10
        assert res["trafo_loading_max_pct"] >= 0.0


def test_main_power_flow_load_sweep(main_twin):
    """Verify main network step function runs power flow and thermal loading increases with load."""
    loads = [10.0, 25.0, 40.0, 45.0, 50.0]
    prev_loading = 0.0
    prev_v_min = 1.05

    for load in loads:
        loading_pct, v_min = main_twin.step(load)
        assert loading_pct > prev_loading, "Transformer loading must increase monotonically with load"
        assert v_min < prev_v_min, "Minimum voltage must decrease monotonically with load"
        prev_loading = loading_pct
        prev_v_min = v_min


def test_feeder_length_voltage_drop():
    """Verify longer feeder length induces greater voltage depression."""
    twin_short = MainPhysicalTwin(feeder_length_km=2.0)
    twin_long = MainPhysicalTwin(feeder_length_km=15.0)

    test_load = 40.0
    _, v_short = twin_short.step(test_load)
    _, v_long = twin_long.step(test_load)

    assert v_long < v_short, f"Long feeder should experience larger voltage drop: {v_long} vs {v_short}"


def test_extreme_overload_protection_trigger():
    """Verify overload load (55 MW) depresses voltage below 0.90 pu and triggers protection."""
    twin = MainPhysicalTwin(feeder_length_km=10.0)
    loading, voltage = twin.step(55.0)

    # At 55 MW with a 10km line, voltage dips below the 0.90 pu threshold
    assert voltage < 0.90
    assert twin.iec61850.pdis.data_objects["Op"] is True
    assert twin.iec61850.xcbr.data_objects["Pos.stVal"] is False
    assert len(twin.protection_trips) > 0


def test_power_flow_solver_divergence_handling():
    """Verify solver handles physically infeasible inputs safely without crashing."""
    twin = CorePhysicalTwin()
    # Extremely massive load that causes power flow non-convergence
    res = twin.run_power_flow(gen_mw=0.0, load_mw=100000.0)
    # PhysicalTwin catches divergence and returns None
    assert res is None or res.get("voltage_min_pu", 0) == 0.0

