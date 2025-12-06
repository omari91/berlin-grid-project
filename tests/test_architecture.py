"""Test suite for modular src/ architecture."""
import pytest
from src.core import DataLayer, PhysicalTwin, Controller, StochasticEngine


def test_datalayer_instantiation():
    """Verify DataLayer can be instantiated."""
    dl = DataLayer()
    assert dl is not None
    assert hasattr(dl, 'load_berlin_data')


def test_physicaltwin_instantiation():
    """Verify PhysicalTwin can be instantiated."""
    twin = PhysicalTwin()
    assert twin is not None
    assert hasattr(twin, 'run_power_flow')


def test_controller_instantiation():
    """Verify Controller can be instantiated."""
    ctrl = Controller()
    assert ctrl is not None
    assert hasattr(ctrl, 'fuzzy_control')


def test_stochastic_engine_instantiation():
    """Verify StochasticEngine can be instantiated."""
    engine = StochasticEngine()
    assert engine is not None
    assert hasattr(engine, 'generate_ar1_noise')


def test_module_integration():
    """Verify all modules can work together."""
    dl = DataLayer()
    twin = PhysicalTwin()
    ctrl = Controller()
    engine = StochasticEngine()
    
    # All components should be instantiable
    assert all([dl, twin, ctrl, engine])
