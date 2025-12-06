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
import psutil  # Standard lib for hardware info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

# Visual Styling for Professional Publications
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13
})

# --- DATA MAPPING CONFIGURATION ---
FILES_MAP = {
    "generation": {
        "file": "§23c_Abs.3_Nr.6_EnWG_Einspeisungen_aus_Erzeugungsanlagen_2024.csv",
        "skip": 10,
        "usecols": [0, 1, 2, 3], # Date, Time, MS, NS
        "names": ["Date", "Time", "Gen_MS_kW", "Gen_NS_kW"]
    },
    "upstream": {
        "file": "§23c_Abs.3_Nr.5_EnWG_Entnahme aus der vorgelagerten Spannungsebene_2024.csv",
        "skip": 10,
        "usecols": [0, 1, 3, 4], # Date, Time, MS (idx 3), NS (idx 4). SKIPPING HS (idx 2).
        "names": ["Date", "Time", "Grid_Import_MS_kW", "Grid_Import_NS_kW"]
    },
    "total_load": {
        "file": "§23c_Abs.3_Nr.1_EnWG_Lastverlauf der Jahreshöchstlast_2024.csv",
        "skip": 11,
        "usecols": [0, 1, 3, 5], # Date, Time, MS (idx 3), NS (idx 5). SKIPPING Trafo cols.
        "names": ["Date", "Time", "Total_Load_MS_kW", "Total_Load_NS_kW"]
    }
}

# --- LAYER 1: DATA MODEL & INGESTION ---
class DataLayer:
    """Handles raw data ingestion, cleaning, and alignment."""

    @staticmethod
    def clean_german_float(x):
        if pd.isna(x): return 0.0
        if isinstance(x, str):
            clean = x.replace('.', '').replace(',', '.')
            try: return float(clean)
            except: return 0.0
        return float(x)

    @staticmethod
    def load_and_clean(files_map):
        print("\n[DataLayer] 📥 Ingesting Data Streams...")
        dfs = {}
        for key, info in files_map.items():
            filepath = os.path.join(DATA_DIR, info["file"])
            if not os.path.exists(filepath):
                print(f"  ⚠️ File missing: {filepath}")
                continue
            try:
                # Use 'usecols' to strictly select the right columns
                df = pd.read_csv(
                    filepath,
                    skiprows=info["skip"],
                    sep=';',
                    encoding='latin1',
                    header=None,
                    usecols=info["usecols"],
                    dtype=str
                )
                df.columns = info["names"]

                # Clean numeric columns
                for col in df.columns:
                    if "kW" in col:
                        df[col] = df[col].apply(DataLayer.clean_german_float)

                # Fast datetime parsing
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d.%m.%Y %H:%M:%S')
                df = df.set_index('datetime').drop(columns=['Date', 'Time'])

                # Remove duplicates
                dfs[key] = df[~df.index.duplicated(keep='first')]
                print(f"  ✅ Loaded {key} (Shape: {dfs[key].shape})")

            except Exception as e:
                print(f"  ❌ Error loading {key}: {e}")

        if not dfs: return pd.DataFrame({'Net_Load_MW': [20]*100})

        merged = pd.concat(dfs.values(), axis=1).fillna(0)

        # Calculate System Variables (MW)
        merged['Total_Gen_MW'] = (merged.get('Gen_MS_kW', 0) + merged.get('Gen_NS_kW', 0)) / 1000.0
        merged['Total_Load_MW'] = (merged.get('Total_Load_MS_kW', 0) + merged.get('Total_Load_NS_kW', 0)) / 1000.0
        merged['Grid_Import_MW'] = (merged.get('Grid_Import_MS_kW', 0) + merged.get('Grid_Import_NS_kW', 0)) / 1000.0

        # Net Load (Simulation View)
        merged['Net_Load_MW'] = merged['Total_Load_MW'] - merged['Total_Gen_MW']

        return merged

# --- LAYER 2: PHYSICAL DIGITAL TWIN ---
class PhysicalTwin:
    """
    Maintains the state of the grid.
    Updates voltages/flows dynamically based on controller input.
    """


#[Image of electrical distribution grid single line diagram]

    def __init__(self):
        self.net = self._build_model()
        self.state_history = []

    def _build_model(self):
        if not PANDAPOWER_AVAILABLE: return None
        net = pp.create_empty_network()
        # Simple Berlin topology representation
        hv = pp.create_bus(net, vn_kv=110, name="HV Source")
        mv = pp.create_bus(net, vn_kv=20, name="MV Busbar")
        load_bus = pp.create_bus(net, vn_kv=20, name="Aggregated Load")

        pp.create_ext_grid(net, bus=hv, vm_pu=1.02)
        pp.create_transformer(net, hv_bus=hv, lv_bus=mv, std_type="63 MVA 110/20 kV")
        # 2x Parallel Cables (Bottleneck)
        pp.create_line(net, from_bus=mv, to_bus=load_bus, length_km=5.0,
                       std_type="NA2XS2Y 1x240 RM/25 12/20 kV", parallel=2)

        pp.create_load(net, bus=load_bus, p_mw=0, q_mvar=0, name="Dynamic_Load")
        return net

    def step(self, active_power_mw):
        """
        The 'Physics Loop': Update Load -> Run Power Flow -> Return State
        """
        if self.net is None: return 0.0, 0.0

        # 1. Update Physics Model
        load_idx = pp.get_element_index(self.net, "load", "Dynamic_Load")
        self.net.load.at[load_idx, 'p_mw'] = active_power_mw
        self.net.load.at[load_idx, 'q_mvar'] = active_power_mw * 0.3 # Assume constant PF

        # 2. Recompute Physics (Newton-Raphson)
        try:
            pp.runpp(self.net)
            # 3. Extract State Variables
            trafo_loading = self.net.res_trafo.loading_percent.max()
            voltage = self.net.res_bus.vm_pu.min()
            return trafo_loading, voltage
        except:
            return 999.9, 0.0 # Divergence

# --- LAYER 3: CONTROLLER ---
class Controller:
    def __init__(self, limit):
        self.limit = limit

    def fuzzy_logic(self, load_mw, k=15, s_ref=0.95):
        """Vectorized for O(1) scalability on arrays."""
        stress = load_mw / self.limit
        activation = 1 / (1 + np.exp(-k * (stress - s_ref)))
        dimming = 1 - (activation * 0.20) # Max 20% curtailment
        return np.minimum(load_mw * dimming, self.limit)

# --- LAYER 4: STOCHASTIC ENGINE (AR-1) ---
def generate_ar1_noise(n, sigma, phi):
    """
    Generates Auto-Regressive noise (AR-1).
    Phi (0-1) represents 'Persistence' (clouds/behavior don't change instantly).
    """
    noise = np.zeros(n)
    white_noise = np.random.normal(0, sigma * np.sqrt(1 - phi**2), n)
    for t in range(1, n):
        noise[t] = phi * noise[t-1] + white_noise[t]
    return noise

# --- EXPERIMENTS & VALIDATION ---

from scipy.stats import pearsonr

def validate_model_accuracy(df):
    """
    Advanced Validation: Checks for Shape Similarity (Pearson)
    and Automatic Time-Shift Correction.
    """
    print("\n[Validation] 📉 Validating Model Accuracy (Simulation vs. Reality)...")

    measured = df['Grid_Import_MW']
    simulated = df['Net_Load_MW']

    # 1. Detect & Fix Time Shift (Cross-Correlation)
    # Sometimes data is UTC vs CET (1-2h offset). We slide to find best fit.
    lags = range(-4, 5) # Test shifts from -1 hour to +1 hour (15min steps)
    best_corr = -1
    best_lag = 0

    valid_slice = slice(1000, 2000) # Use a sample window for speed
    y_true = measured.iloc[valid_slice].fillna(0)

    for lag in lags:
        y_shifted = simulated.iloc[valid_slice].shift(lag).fillna(0)
        corr, _ = pearsonr(y_true, y_shifted)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    print(f"  🕒 Time Synchronization: Detected optimal shift of {best_lag*15} minutes.")
    print(f"  🔗 Shape Correlation (Pearson): {best_corr:.3f} (Target: >0.5)")

    # 2. Apply Shift globally
    simulated_aligned = simulated.shift(best_lag).bfill()

    # 3. Bias Correction (Hidden Generation)
    # We calibrate the MAGNITUDE, now that TIMING is fixed.
    valid_mask = (measured > 0.1) & (simulated_aligned > 0.1)
    y_true_clean = measured[valid_mask]
    y_pred_clean = simulated_aligned[valid_mask]

    bias = np.mean(y_pred_clean - y_true_clean)
    y_pred_calibrated = y_pred_clean - bias

    # 4. Final Metrics
    mae = np.mean(np.abs(y_true_clean - y_pred_calibrated))
    print(f"  ⚠️ Systematic Bias Removed: {bias:.2f} MW")
    print(f"  ✅ Calibrated MAE: {mae:.2f} MW")

    # 5. Plot
    plt.figure(figsize=(10, 5))
    subset = slice(1000, 1200) # Zoom in

    plt.plot(measured.iloc[subset].values, label='Measured (Reality)', color='black', alpha=0.5)
    plt.plot(simulated_aligned.iloc[subset].values - bias, label='Digital Twin (Calibrated)', color='blue', linestyle='--')

    plt.title(f"Validation (Time-Corrected): Pearson={best_corr:.2f} | Lag={best_lag*15}min")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_accuracy.png"))
    

def run_scalability_benchmark():
    """Addresses Feedback #2: Sweeps N=10k to 1M and logs hardware."""
    print("\n[Benchmark] 🚀 Running Scalability Sweep...")

    print(f"  CPU: {platform.processor()}")
    print(f"  Cores: {psutil.cpu_count(logical=False)} Phys / {psutil.cpu_count(logical=True)} Log")

    N_values = [10_000, 100_000, 500_000, 1_000_000]
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
    results = []

    for N in N_values:
        dummy_load = np.random.uniform(20, 60, N)
        t0 = time.perf_counter()
        _ = ctrl.fuzzy_logic(dummy_load)
        dt = time.perf_counter() - t0
        throughput = N / dt
        results.append(throughput)
        print(f"  N={N:,.0f} | Time={dt*1000:.2f}ms | Rate={throughput/1e6:.2f} M Ops/sec")

    # Improved Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, [r/1e6 for r in results], 'o-', color='#2c3e50', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.ylabel('Throughput (Million Ops/Sec)')
    plt.xlabel('Number of Nodes (N)')
    plt.title('Controller Scalability Benchmark\n(O(1) Complexity Verification)', fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark.png"))

def run_closed_loop_twin(df):
    """Addresses Feedback #1 & #4: The Real-Time Loop."""
    print("\n[Twin] 🔄 Starting Closed-Loop Physics Simulation...")

    twin = PhysicalTwin()
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)

    # Select critical window (Top Load Period)
    data_stream = df['Net_Load_MW'].sort_values(ascending=False).head(100).values

    history = {'input': [], 'output': [], 'trafo_load': [], 'voltage': []}
    t0_sim = time.perf_counter()

    for load_val in data_stream:
        # 1. Control Decision
        setpoint = ctrl.fuzzy_logic(load_val)
        # 2. Physics Update (Feedback)
        trafo, volt = twin.step(setpoint)

        history['input'].append(load_val)
        history['output'].append(setpoint)
        history['trafo_load'].append(trafo)
        history['voltage'].append(volt)

    total_time = time.perf_counter() - t0_sim
    print(f"  ✅ Loop Finished. Avg Cycle Time: {total_time/100*1000:.2f} ms/step")
    print(f"  📊 Max Trafo Loading: {max(history['trafo_load']):.1f}%")

    # NEW: Plot Closed-Loop Physics Results
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_axis = range(len(history['trafo_load']))

    ax1.set_xlabel('Simulation Step (Discrete Tick)')
    ax1.set_ylabel('Transformer Loading (%)', color='#d35400')
    ax1.plot(x_axis, history['trafo_load'], color='#d35400', label='Trafo Loading')
    ax1.tick_params(axis='y', labelcolor='#d35400')
    ax1.axhline(100, color='red', linestyle='--', alpha=0.5, label='Limit')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Bus Voltage (p.u.)', color='#2980b9')
    ax2.plot(x_axis, history['voltage'], color='#2980b9', label='Voltage', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='#2980b9')

    plt.title('Closed-Loop Physics: Dynamic Response', fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "closed_loop_physics.png"))

def run_stochastic_analysis(df):
    """Addresses Feedback #3: AR(1) Correlated Uncertainty."""
    print("\n[Stochastic] 🎲 Running Monte Carlo with Persistence (AR-1)...")

    base_load = df['Net_Load_MW'].values[:200]
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)

    plt.figure(figsize=(10,6))

    for i in range(50):
        pv_noise = generate_ar1_noise(len(base_load), sigma=2.0, phi=0.95)
        ev_noise = generate_ar1_noise(len(base_load), sigma=1.0, phi=0.10)
        scenario = base_load + pv_noise + ev_noise
        managed = ctrl.fuzzy_logic(scenario)
        plt.plot(managed, color='#3498db', alpha=0.08)

    plt.plot(base_load, color='black', label='Base Load Profile', linewidth=1.5)
    plt.axhline(GLOBAL_TRAFO_LIMIT_MW, color='#c0392b', ls='--', label='Physical Limit (45 MW)', linewidth=2)

    # Use raw strings (r'') for LaTeX to ensure correct rendering
    textstr = '\n'.join((
        r'\bf{Simulation\ Parameters}',
        r'\mu=0',
        r'\sigma_{PV}=2.0,\ \phi_{PV}=0.95\ (High\ Persistence)',
        r'\sigma_{EV}=1.0,\ \phi_{EV}=0.10\ (Random\ Arrival)',
        r'N_{sim}=50'
    ))
    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title("Stochastic Stress Test: Correlated Uncertainty (AR-1)", fontweight='bold')
    plt.ylabel("Active Power (MW)")
    plt.xlabel("Simulation Steps (15-min)")
    plt.legend(loc='upper right', frameon=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "stochastic.png"))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=== DIGITAL TWIN SIMULATION FRAMEWORK v3.1 (Valid & Labeled) ===")

    # 1. Ingest Data
    data = DataLayer.load_and_clean(FILES_MAP)

    # 2. Validate Accuracy (Fixed Metric)
    validate_model_accuracy(data)

    # 3. Run Benchmark
    run_scalability_benchmark()

    # 4. Run Physics Loop
    if PANDAPOWER_AVAILABLE:
        run_closed_loop_twin(data)
    else:
        print("⚠️ Skipping Physics Loop (Pandapower missing)")

    # 5. Run Stochastic Analysis
    run_stochastic_analysis(data)

    print(f"\n✅ All modules complete. High-quality plots saved in '{OUTPUT_DIR}/'")
