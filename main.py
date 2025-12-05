### 2. `main.py` (The Golden Master Code)
"""
Berlin Grid Digital Twin: From Simulation to Reality
Author: Clifford Ondieki
Reference: Bundesnetzagentur Monitoring Report 2024

Purpose: Engineering Proof for LinkedIn Series.
Demonstrates:
1. Handling the €110bn Grid Expansion Challenge (Report p.14).
2. Managing the 2.04M §14a Devices (Report p.16).
3. Scalability of Edge Intelligence (Redispatch 3.0).
"""

import os
import time
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing pandapower
try:
    import pandapower as pp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    print("⚠️ Warning: 'pandapower' library not found. Physics checks will be skipped.")

# --- CONFIGURATION ---
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Global Transformer Limit (Matches standard 63 MVA Transformer @ 0.9 PF)
GLOBAL_TRAFO_LIMIT_MW = 45.0

# Styling for Professional Graphs
sns.set_theme(style="white", context="talk")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']

FILES_MAP = {
    "generation": {"file": "§23c_Abs.3_Nr.6_EnWG_Einspeisungen_aus_Erzeugungsanlagen_2024.csv", "skip": 9, "names": ["Date", "Time", "Gen_MS_kW", "Gen_NS_kW"]},
    "upstream": {"file": "§23c_Abs.3_Nr.5_EnWG_Entnahme aus der vorgelagerten Spannungsebene_2024.csv", "skip": 9, "names": ["Date", "Time", "Grid_Import_MS_kW", "Grid_Import_NS_kW"]},
    "total_load": {"file": "§23c_Abs.3_Nr.1_EnWG_Lastverlauf der Jahreshöchstlast_2024.csv", "skip": 10, "names": ["Date", "Time", "Total_Load_MS_kW", "Total_Load_NS_kW"]}
}

# --- HELPER FUNCTIONS ---

def clean_german_float(x):
    if pd.isna(x): return 0.0
    if isinstance(x, str):
        clean = x.replace('.', '').replace(',', '.')
        try: return float(clean)
        except: return 0.0
    return float(x)

def add_branding(ax):
    """Adds professional footer/watermark."""
    ax.text(1, -0.25, 'Simulation: C. Ondieki | Data: Energienetze Berlin 2024',
            transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#777777')

# --- SECTION: CONTROLLER LOGIC (BENCHMARKING) ---

class GridController:
    """
    Modular Controller Architecture for Comparative Benchmarking.
    """
    def __init__(self, limit_mw):
        self.limit = limit_mw

    def hard_cutoff(self, load_mw):
        """Baseline 1: Binary Switch (Relay). Instant cut-off."""
        return min(load_mw, self.limit)

    def linear_droop(self, load_mw):
        """Baseline 2: Standard Linear Droop (P(f) or P(U) proxy)."""
        if load_mw < self.limit * 0.9:
            return load_mw
        # Linear ramp down starting at 90% load
        excess = load_mw - (self.limit * 0.9)
        return load_mw - (excess * 0.5)

    def fuzzy_logic(self, load_mw, k=15, s_ref=0.95):
        """
        The Proposed Solution: Sigmoid Smoothing.
        Equation: $$ \alpha = \frac{1}{1 + e^{-k(S - S_{ref})}} $$
        """
        stress = load_mw / self.limit
        dimming_factor = 1 / (1 + np.exp(-k * (stress - s_ref)))
        soft_cap = load_mw * (1 - dimming_factor * 0.3)
        return min(soft_cap, self.limit * 1.02)

# --- PHASE 1: ETL & VALIDATION ---

def load_and_validate_data():
    print("\n🚀 Starting ETL Pipeline...")
    dfs = {}

    for key, info in FILES_MAP.items():
        filepath = os.path.join(DATA_DIR, info["file"])
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: File not found: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath, skiprows=info["skip"], sep=';', encoding='latin1', header=0, dtype=str)
            df = df.iloc[:, :len(info["names"])]
            df.columns = info["names"]
            for col in df.columns:
                if "kW" in col: df[col] = df[col].apply(clean_german_float)

            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d.%m.%Y %H:%M:%S')
            df = df.set_index('datetime').drop(columns=['Date', 'Time'])
            df = df[~df.index.duplicated(keep='first')]
            dfs[key] = df
            print(f"   ✅ Loaded {key}")
        except Exception as e:
            print(f"   ❌ Error loading {key}: {e}")

    if not dfs: raise ValueError("No data loaded.")

    merged = pd.concat(dfs.values(), axis=1).fillna(0)
    merged['Total_Gen_MW'] = (merged.get('Gen_MS_kW', 0) + merged.get('Gen_NS_kW', 0)) / 1000.0
    merged['Total_Load_MW'] = (merged.get('Total_Load_MS_kW', 0) + merged.get('Total_Load_NS_kW', 0)) / 1000.0
    merged['Grid_Import_MW'] = (merged.get('Grid_Import_MS_kW', 0) + merged.get('Grid_Import_NS_kW', 0)) / 1000.0
    merged['Net_Load_MW'] = merged['Total_Load_MW'] - merged['Total_Gen_MW']

    return merged

# --- PHASE 2: REAL-TIME ARCHITECTURE & SCALABILITY ---

class StreamingDigitalTwin:
    """
    Simulates a 50Hz real-time streaming environment.
    Demonstrates ability to handle continuous data ingestion.
    """
    def __init__(self, data_stream):
        self.stream = data_stream
        self.controller = GridController(GLOBAL_TRAFO_LIMIT_MW)
        self.latencies = []

    def run_stream(self, ticks=1000):
        print(f"\n📡 Initiating Real-Time Stream Simulation ({ticks} ticks)...")
        print(f"   Hardware: {platform.processor()} | System: {platform.system()}")

        simulated_stream = np.resize(self.stream, ticks)

        start_global = time.perf_counter()

        for load in simulated_stream:
            t0 = time.perf_counter()
            # The decision kernel
            _ = self.controller.fuzzy_logic(load)
            t1 = time.perf_counter()
            self.latencies.append((t1 - t0) * 1e6) # Microseconds

        duration = time.perf_counter() - start_global

        avg_lat = np.mean(self.latencies)
        p99_lat = np.percentile(self.latencies, 99)
        ops_sec = 1_000_000 / avg_lat

        print(f"   ⏱️  Avg Latency: {avg_lat:.2f} µs | P99 Jitter: {p99_lat:.2f} µs")
        print(f"   🚀 Throughput: {int(ops_sec):,} Ops/Sec (Single Core)")

        if avg_lat < 20000: # 20ms = 50Hz cycle
            print("   ✅ Grid Code Compliance: Real-time capable (<20ms cycle time).")

        return ops_sec

# --- PHASE 3: PHYSICS VALIDATION (PANDAPOWER) ---


# Image of electrical substation diagram


def run_pandapower_validation(df):
    """
    Expanded to check Line Loading as per Feedback.
    """
    if not PANDAPOWER_AVAILABLE: return

    print("\n⚡ Running AC Physics Validation (Pandapower)...")
    peak_mw = df['Net_Load_MW'].max()
    print(f"   Simulating Peak Load: {peak_mw:.2f} MW")

    net = pp.create_empty_network()
    b_hv = pp.create_bus(net, vn_kv=110, name="HV Grid")
    b_mv = pp.create_bus(net, vn_kv=20, name="MV Busbar")
    b_load = pp.create_bus(net, vn_kv=20, name="Remote Node")

    pp.create_ext_grid(net, bus=b_hv, vm_pu=1.02)
    pp.create_transformer(net, hv_bus=b_hv, lv_bus=b_mv, std_type="63 MVA 110/20 kV")

    # Define line with specific thermal limit (0.65 kA ~ 22MW capacity per line)
    # Using 2 parallel cables to reach ~45MW capacity
    pp.create_line(net, from_bus=b_mv, to_bus=b_load, length_km=5.0,
                   std_type="NA2XS2Y 1x240 RM/25 12/20 kV", parallel=2)

    pp.create_load(net, bus=b_load, p_mw=peak_mw, q_mvar=peak_mw*0.1)

    try:
        pp.runpp(net)

        # 1. Voltage Check
        voltage = net.res_bus.vm_pu.at[b_load]
        # 2. Line Loading Check
        line_load = net.res_line.loading_percent.max()
        # 3. Trafo Loading Check
        trafo_load = net.res_trafo.loading_percent.max()

        print(f"   📊 Results: Voltage={voltage:.3f} p.u. | Line={line_load:.1f}% | Trafo={trafo_load:.1f}%")

        if 0.90 < voltage < 1.10 and line_load < 100.0:
            print("   ✅ Grid Constraints: Feasible operation.")
        else:
            print("   ⚠️ CRITICAL: Grid Constraint Violation detected.")

    except Exception as e:
        print(f"   ❌ Power Flow Failed: {e}")

# --- PHASE 4: ANALYTICAL SCENARIOS ---

def compare_baselines(df):
    """
    New Scenario: Benchmark Fuzzy Logic against standard industry approaches.
    """
    print("\n⚖️  Running Controller Comparison (Ablation Study)...")
    ctrl = GridController(GLOBAL_TRAFO_LIMIT_MW)

    # Create synthetic ramp to show response behavior
    ramp = np.linspace(GLOBAL_TRAFO_LIMIT_MW * 0.8, GLOBAL_TRAFO_LIMIT_MW * 1.2, 100)

    y_hard = [ctrl.hard_cutoff(x) for x in ramp]
    y_droop = [ctrl.linear_droop(x) for x in ramp]
    y_fuzzy = [ctrl.fuzzy_logic(x) for x in ramp]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ramp, ramp, 'k:', label='Unmanaged (Risk)', alpha=0.3)
    ax.plot(ramp, y_hard, 'r--', label='Hard Cutoff (Relay)', linewidth=2)
    ax.plot(ramp, y_droop, 'b-.', label='Linear Droop', linewidth=2)
    ax.plot(ramp, y_fuzzy, 'g-', label='Fuzzy Logic (Proposed)', linewidth=3)

    ax.set_title('Controller Response Benchmark', fontweight='bold')
    ax.set_xlabel('Input Load (MW)')
    ax.set_ylabel('Managed Load (MW)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '03_controller_benchmark.png'))

def sensitivity_analysis():
    """
    New Scenario: Sensitivity Analysis of Hyperparameter k.
    Demonstrates tunability of the algorithm.
    """
    print("\n🎛️  Running Hyperparameter Sensitivity Analysis...")
    ctrl = GridController(GLOBAL_TRAFO_LIMIT_MW)
    load_range = np.linspace(35, 55, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Testing different 'k' (Steepness) values
    k_values = [5, 15, 30]
    colors = ['#A8D5BA', '#4DAF7C', '#1E4733']

    for k, c in zip(k_values, colors):
        resp = [ctrl.fuzzy_logic(l, k=k) for l in load_range]
        ax.plot(load_range, resp, color=c, label=f'k={k} (Steepness)')

    ax.axhline(GLOBAL_TRAFO_LIMIT_MW, color='red', linestyle='--', label='Limit')
    ax.set_title('Sensitivity Analysis: Impact of Gain (k)', fontweight='bold')
    ax.set_xlabel('Input Load (MW)')
    ax.set_ylabel('Output Power (MW)')
    ax.legend()
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '08_sensitivity_analysis.png'))


# Image of Monte Carlo simulation distribution


def run_monte_carlo_stress_test(df):
    """
    Enhanced Stochastic Model.
    Assumptions:
    1. PV Forecast Error: Normal Dist (Sigma=2.0) - Based on standard RMSE.
    2. EV Variability: Uniform Dist - Represents random arrival times.
    3. Note: Independence assumed for simplicity (Correlation=0).
    """
    print("\n🎲 Running Scenario 4: Monte Carlo Stress Test (n=100)...")
    ctrl = GridController(GLOBAL_TRAFO_LIMIT_MW)
    base_load = df['Net_Load_MW'].values
    iterations = 100
    results = []

    for i in range(iterations):
        # 1. PV Uncertainty (Gaussian)
        pv_noise = np.random.normal(0, 2.0, size=len(base_load))
        # 2. EV/Demand Uncertainty (Uniform)
        ev_noise = np.random.uniform(-1.0, 1.0, size=len(base_load))

        noisy_load = base_load + pv_noise + ev_noise
        managed_load = np.array([ctrl.fuzzy_logic(l) for l in noisy_load])
        results.append(managed_load)

    results = np.array(results)
    p05 = np.percentile(results, 5, axis=0)
    p95 = np.percentile(results, 95, axis=0) # 95th Percentile Risk
    median = np.median(results, axis=0)

    zoom_start, zoom_end = 2000, 2500
    x = np.arange(zoom_end - zoom_start)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, median[zoom_start:zoom_end], color='blue', label='Median')
    ax.fill_between(x, p05[zoom_start:zoom_end], p95[zoom_start:zoom_end],
                    color='blue', alpha=0.2, label='95% Confidence Interval')
    ax.axhline(GLOBAL_TRAFO_LIMIT_MW, color='red', linestyle='--', label='Physical Limit')

    ax.set_title('Scenario 4: Stochastic Robustness (95% CI)', fontweight='bold')
    ax.legend(loc='upper right', frameon=False)
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '05_probabilistic_stress_test.png'))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Please create a '{DATA_DIR}' folder.")
    else:
        # 1. ETL
        grid_data = load_and_validate_data()

        # 2. Real-Time Architecture Check
        twin = StreamingDigitalTwin(grid_data['Net_Load_MW'].values)
        ops_rate = twin.run_stream(ticks=5000)

        # 3. Physics & Constraints
        run_pandapower_validation(grid_data)

        # 4. Controller Benchmarking (New)
        compare_baselines(grid_data)
        sensitivity_analysis()

        # 5. Stochastic Verification
        run_monte_carlo_stress_test(grid_data)

        print(f"\n🎉 Simulation Complete. Results in '{OUTPUT_DIR}'.")
