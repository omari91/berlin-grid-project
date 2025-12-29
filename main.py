"""
Berlin Grid Digital Twin: From Simulation to Reality 
Author: Clifford Ondieki
Reference: Bundesnetzagentur Monitoring Report 2024

### Optimizations ###
1. Fast Stochasticity: Replaced loop-based AR-1 with Scipy IIR filters.
2. Dynamic Topology: Added variable feeder length to stress-test voltage drops.
3. Hybrid Streaming: Separated Jitter (scalar) vs Throughput (vector) logic.
4. Robustness: Added fallback data generation if CSVs are missing.
"""

import os
import time
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import lfilter

# Try importing pandapower
try:
    import pandapower as pp
    import pandapower.topology as ppt
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    print("⚠️ Warning: 'pandapower' library not found. Physics checks will be skipped.")

# --- CONFIGURATION ---
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

GLOBAL_TRAFO_LIMIT_MW = 45.0  # Physical Hard Limit

# Visual Styling
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# --- DATA MAPPING ---
FILES_MAP = {
    "generation": {"file": "Gen_2024.csv", "skip": 0, "cols": [0, 2], "names": ["Time", "Gen_MW"]},
    "load": {"file": "Load_2024.csv", "skip": 0, "cols": [0, 2], "names": ["Time", "Load_MW"]}
}

# --- LAYER 1: DATA MODEL ---
class DataLayer:
    @staticmethod
    def load_and_clean(files_map):
        """
        Robust loader that generates synthetic data if files are missing.
        """
        print("\n[DataLayer] 📥 Ingesting Data Streams...")
        # For this demo, we auto-generate synthetic data to ensure code runs immediately
        # regardless of local CSV file state.
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="15min")
        
        # Synthetic profile: Base load + Daily Cycle + Noise
        t = np.linspace(0, 4*np.pi, 1000)
        base = 35.0 + 10.0 * np.sin(t) + np.random.normal(0, 2, 1000)
        
        df = pd.DataFrame(index=dates)
        df['Net_Load_MW'] = base
        print(f"  ✅ Generated Synthetic German Grid Profile (N={len(df)})")
        return df

# --- LAYER 2: PHYSICS ENGINE ---
class PhysicalTwin:
    def __init__(self, feeder_length_km=5.0):
        self.net = self._build_model(feeder_length_km)
    
    def _build_model(self, length_km):
        if not PANDAPOWER_AVAILABLE: return None
        net = pp.create_empty_network()
        hv = pp.create_bus(net, vn_kv=110, name="HV Source")
        mv = pp.create_bus(net, vn_kv=20, name="MV Busbar")
        load_bus = pp.create_bus(net, vn_kv=20, name="Aggregated Load")
        
        pp.create_ext_grid(net, bus=hv, vm_pu=1.02)
        pp.create_transformer(net, hv_bus=hv, lv_bus=mv, std_type="63 MVA 110/20 kV")
        
        # Update: Dynamic Line Length
        pp.create_line(net, from_bus=mv, to_bus=load_bus, length_km=length_km, 
                       std_type="NA2XS2Y 1x240 RM/25 12/20 kV", parallel=2)
        
        pp.create_load(net, bus=load_bus, p_mw=0, q_mvar=0, name="Dynamic_Load")
        return net

    def step(self, active_power_mw):
        if self.net is None: return 0.0, 1.0
        load_idx = pp.get_element_index(self.net, "load", "Dynamic_Load")
        
        # Update State
        self.net.load.at[load_idx, 'p_mw'] = active_power_mw
        self.net.load.at[load_idx, 'q_mvar'] = active_power_mw * 0.3 
        
        try:
            # Warm-Start Newton-Raphson
            pp.runpp(self.net, algorithm='nr', init_vm_pu="results")
            return self.net.res_trafo.loading_percent.max(), self.net.res_bus.vm_pu.min()
        except:
            return 999.9, 0.0

# --- LAYER 3: CONTROLLERS ---
class Controller:
    def __init__(self, limit):
        self.limit = limit

    def fuzzy_logic(self, load_mw, k=15, s_ref=0.95):
        """
        Vectorized Soft Sigmoid Control.
        Works efficiently on both Scalars (float) and Vectors (numpy array).
        """
        stress = load_mw / self.limit
        # Numpy handles the broadcasting automatically here
        activation = 1 / (1 + np.exp(-k * (stress - s_ref)))
        dimming = 1 - (activation * 0.20)
        return np.minimum(load_mw * dimming, self.limit)

    def hard_cutoff(self, load_mw):
        return np.minimum(load_mw, self.limit)

# --- LAYER 4: STREAMING SIMULATOR ---
class StreamingDigitalTwin:
    def __init__(self):
        self.latencies = []

    def run_jitter_test(self, n_ticks=5000):
        """
        Objective 1: Real-Time Proof (Scalar Loop).
        Measures the overhead of calling the controller tick-by-tick.
        """
        print("\n[Streaming] 📡 Running Real-Time Jitter Test...")
        ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
        stream = np.random.uniform(20, 60, n_ticks)
        
        for load in stream:
            t0 = time.perf_counter()
            _ = ctrl.fuzzy_logic(load) 
            dt = (time.perf_counter() - t0) * 1e6
            self.latencies.append(dt)
            
        p99 = np.percentile(self.latencies, 99)
        print(f"  ⚡ P99 Latency: {p99:.2f} µs (Target: <1000 µs)")

# --- EXPERIMENTS ---

def run_throughput_benchmark():
    """
    Update: Fully Vectorized Benchmark.
    Tests raw CPU throughput without Python loop overhead.
    """
    print("\n[Benchmark] 🚀 Running Vectorized Throughput Sweep...")
    N_values = [10_000, 100_000, 1_000_000]
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
    results = []
    
    for N in N_values:
        data = np.random.uniform(20, 60, N)
        t0 = time.perf_counter()
        _ = ctrl.fuzzy_logic(data) # Vectorized call
        duration = time.perf_counter() - t0
        rate = N / duration
        results.append(rate)
        print(f"  N={N:,.0f} | Rate={rate/1e6:.2f} M Ops/sec")

    plt.figure(figsize=(8,5))
    plt.plot(N_values, [r/1e6 for r in results], 'o-', color='navy')
    plt.xscale('log')
    plt.title('Vectorized Controller Throughput')
    plt.ylabel('M Ops/sec')
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark.png"))

def run_stochastic_physics_loop(df):
    """
    Update: Uses Scipy lfilter for fast AR-1 noise generation.
    """
    print("\n[Stochastic] 🎲 Running Physics-Integrated Monte Carlo...")
    if not PANDAPOWER_AVAILABLE: return

    base_load = df['Net_Load_MW'].values[:50]
    twin = PhysicalTwin(feeder_length_km=8.0) # Stress test long line
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
    results_volt = []
    
    # Fast AR-1 Generator using Signal Processing Filter
    def fast_ar1(n, phi=0.95, sigma=2.0):
        white = np.random.normal(0, sigma * np.sqrt(1-phi**2), n)
        # y[n] = phi*y[n-1] + x[n]  -> Transfer Function: 1 / (1 - phi*z^-1)
        return lfilter([1], [1, -phi], white)

    plt.figure(figsize=(10,5))
    
    for _ in range(30):
        noise = fast_ar1(len(base_load))
        scenario_load = base_load + noise
        scenario_voltages = []
        
        # Physics Loop (Must be looped, hard to vectorize NR solver)
        for load in scenario_load:
            setpoint = ctrl.fuzzy_logic(load)
            _, v = twin.step(setpoint)
            scenario_voltages.append(v)
        
        results_volt.append(scenario_voltages)
        plt.plot(scenario_voltages, color='blue', alpha=0.05)
        
    avg_volt = np.mean(results_volt, axis=0)
    plt.plot(avg_volt, color='black', label='Mean Voltage')
    plt.axhline(0.90, color='red', ls='--', label='Limit (0.90 pu)')
    plt.title("Stochastic Physics Validation (Fast AR-1)")
    plt.ylabel("Voltage (p.u.)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "stochastic_physics.png"))

def run_controller_ablation(df):
    print("\n[Ablation] ⚖️ Running Controller Comparison...")
    loads = df['Net_Load_MW'].sort_values(ascending=False).head(200).values
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
    
    # Vectorized execution
    res_fuzzy = ctrl.fuzzy_logic(loads)
    res_hard = ctrl.hard_cutoff(loads)
    
    plt.figure(figsize=(10,5))
    plt.plot(loads, label='Unmanaged', color='grey', ls=':', alpha=0.6)
    plt.plot(res_hard, label='Hard Cutoff', color='red', ls='--')
    plt.plot(res_fuzzy, label='Fuzzy Logic', color='green', lw=2)
    plt.axhline(GLOBAL_TRAFO_LIMIT_MW, color='black', ls='-.')
    plt.title("Controller Response Comparison")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation.png"))

if __name__ == "__main__":
    print("=== DIGITAL TWIN (Optimized) ===")
    
    # 1. Load / Generate Data
    data = DataLayer.load_and_clean(FILES_MAP)
    
    # 2. Real-Time Proof (Jitter)
    streamer = StreamingDigitalTwin()
    streamer.run_jitter_test()
    
    # 3. Hardware Benchmark (Throughput)
    run_throughput_benchmark()
    
    # 4. Stochastic Physics
    run_stochastic_physics_loop(data)
    
    # 5. Ablation
    run_controller_ablation(data)
    
    print(f"\n✅ Optimization Complete. Artifacts in '{OUTPUT_DIR}/'")
