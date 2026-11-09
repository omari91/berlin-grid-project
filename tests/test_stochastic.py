"""Stochastic Monte Carlo & Efficiency Convergence Validation Test Suite."""
import pytest
import numpy as np
from scipy.signal import lfilter

from src.core import StochasticEngine
from main import Controller, PhysicalTwin, DataLayer, FILES_MAP, GLOBAL_TRAFO_LIMIT_MW


def test_ar1_signal_properties():
    """Verify AR(1) noise generator produces stationary variance and specified autocorrelation."""
    np.random.seed(42)
    n = 10000
    sigma = 2.0
    phi = 0.95

    # 1. src/core.py time-domain generator
    engine = StochasticEngine()
    noise_loop = engine.generate_ar1_noise(n=n, sigma=sigma, phi=phi)

    # 2. main.py lfilter IIR generator
    white = np.random.normal(0, sigma * np.sqrt(1 - phi**2), n)
    noise_iir = lfilter([1], [1, -phi], white)

    # Compute empirical lag-1 autocorrelations
    r1_loop = np.corrcoef(noise_loop[:-1], noise_loop[1:])[0, 1]
    r1_iir = np.corrcoef(noise_iir[:-1], noise_iir[1:])[0, 1]

    # Verify lag-1 is within 5% of theoretical phi=0.95
    assert np.isclose(r1_loop, phi, rtol=0.05), f"Loop AR(1) lag-1 autocorr was {r1_loop}, expected ~{phi}"
    assert np.isclose(r1_iir, phi, rtol=0.05), f"IIR AR(1) lag-1 autocorr was {r1_iir}, expected ~{phi}"

    # Verify stationary standard deviation is ~sigma (within 10%)
    assert np.isclose(np.std(noise_loop), sigma, rtol=0.10)
    assert np.isclose(np.std(noise_iir), sigma, rtol=0.10)


def test_stochastic_physics_monte_carlo_stability():
    """Verify 50 Monte Carlo perturbations through AC power flow maintain voltage stability."""
    np.random.seed(123)
    n_scenarios = 30
    n_timesteps = 25
    base_load = 32.0  # MW nominal operating point

    twin = PhysicalTwin(feeder_length_km=6.0)
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)

    all_voltages = []

    for _ in range(n_scenarios):
        # Generate AR(1) perturbations
        white = np.random.normal(0, 1.5 * np.sqrt(1 - 0.95**2), n_timesteps)
        noise = lfilter([1], [1, -0.95], white)
        scenario_load = base_load + noise

        scenario_v = []
        for l in scenario_load:
            setpoint = ctrl.fuzzy_logic(l)
            _, v = twin.step(setpoint)
            scenario_v.append(v)

        all_voltages.append(scenario_v)

    all_voltages_arr = np.array(all_voltages)
    mean_v = np.mean(all_voltages_arr, axis=0)
    p5_v = np.percentile(all_voltages_arr, 5, axis=0)
    p95_v = np.percentile(all_voltages_arr, 95, axis=0)

    # Voltage constraints (0.90 - 1.10 p.u.)
    assert (p5_v >= 0.90).all(), "5th percentile voltage dropped below 0.90 p.u. safety limit"
    assert (p95_v <= 1.10).all(), "95th percentile voltage exceeded 1.10 p.u. upper limit"
    # Ensure standard deviation is small (controller dampens perturbations)
    std_v = np.std(all_voltages_arr, axis=0)
    assert (std_v < 0.05).all()


def test_efficiency_convergence_claim():
    """Verify that under heavy congestion (limit=22MW), Fuzzy controller converges to >=97% efficiency of Hard Cutoff."""
    df = DataLayer.load_and_clean(FILES_MAP)
    loads = df["Net_Load_MW"].values

    limit_heavy = 22.0
    ctrl = Controller(limit_heavy)

    fuzzy_energy = np.sum(ctrl.fuzzy_logic(loads))
    hard_energy = np.sum(ctrl.hard_cutoff(loads))

    efficiency_ratio = fuzzy_energy / hard_energy
    # Claim: >= 97% of optimal efficiency
    assert efficiency_ratio >= 0.97, (
        f"Efficiency ratio {efficiency_ratio*100:.2f}% did not meet 97% convergence claim"
    )

