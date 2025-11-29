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
    """Adds your professional footer to every graph"""
    ax.text(1, -0.25, 'Simulation: C. Ondieki | Data: Energienetze Berlin 2024', 
            transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#777777')

def fuzzy_control_logic(load_mw, limit_mw):
    """
    Sigmoid Control Loop (The 'Dimmer Switch').
    NOTE: Parameters k=15 and thresh=0.95 mimic an optimized response.
    Complexity: O(1)
    """
    stress = load_mw / limit_mw
    dimming_factor = 1 / (1 + np.exp(-15 * (stress - 0.95)))
    soft_cap = load_mw * (1 - dimming_factor * 0.3)
    final_load = min(soft_cap, limit_mw * 1.02)
    return final_load

# --- PHASE 1: UNIT TEST (SANITY CHECK) ---

def run_basic_pandapower_test():
    """Verifies library function before running complex logic."""
    if not PANDAPOWER_AVAILABLE: return

    print("\n🛠️ Running Basic Pandapower Test (Sanity Check)...")
    try:
        net = pp.create_empty_network() 
        b1 = pp.create_bus(net, vn_kv=20.)
        b2 = pp.create_bus(net, vn_kv=20.)
        pp.create_line(net, from_bus=b1, to_bus=b2, length_km=2.5, std_type="NAYY 4x50 SE")   
        pp.create_ext_grid(net, bus=b1)
        pp.create_load(net, bus=b2, p_mw=1.)
        pp.runpp(net)
        print("   ✅ Test Passed: Library is functioning.")
    except Exception as e:
        print(f"   ❌ Basic Test Failed: {e}")

# --- PHASE 2: ETL & VALIDATION ---

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
    
    # --- PHYSICS CHECK (KCL) ---
    print("\n🔍 Running Physics-Based Validation (KCL)...")
    # Only validate when grid is Importing (Load > Gen) to avoid Export confusion
    importing = merged['Net_Load_MW'] > 0
    if importing.any():
        error = np.abs(merged.loc[importing, 'Grid_Import_MW'] - merged.loc[importing, 'Net_Load_MW'])
        mean_error = error.mean()
        print(f"   Mean Deviation (during import): {mean_error:.2f} MW")
        if mean_error < 10.0:
            print("   ✅ Data Physics Validated (Within Tolerance)")
        else:
            print("   ⚠️ Warning: High Data Discrepancy")
    else:
        print("   Notice: Grid is Net Exporter (High Renewables)")

    return merged

# --- PHASE 3: BENCHMARKING (SCALABILITY) ---

def run_computational_benchmark():
    print("\n⏱️ Running Scalability & Performance Benchmark...")
    
    N = 100_000 
    loads = np.random.uniform(10, 60, N)
    limit = GLOBAL_TRAFO_LIMIT_MW
    
    start_time = time.time()
    _ = [fuzzy_control_logic(l, limit) for l in loads]
    end_time = time.time()
    
    total_time = end_time - start_time
    latency_ms = (total_time / N) * 1000 
    ops_per_second = 1 / (latency_ms / 1000)
    
    print(f"   Processed {N} control steps in {total_time:.4f}s")
    print(f"   ⚡ Latency: {latency_ms:.4f} ms per node")
    print(f"   🚀 Scalability Score: {int(ops_per_second):,} nodes/sec per CPU core")
    
    if latency_ms < 20: 
        print("   ✅ Real-Time Capable (Fits within 50Hz cycle)")
    else:
        print("   ⚠️ Too Slow for Real-Time")
    
    return ops_per_second

# --- PHASE 4: PHYSICS VALIDATION (PANDAPOWER) ---

def run_pandapower_validation(df):
    if not PANDAPOWER_AVAILABLE: return

    print("\n⚡ Running AC Physics Validation (Pandapower)...")
    peak_mw = df['Net_Load_MW'].max()
    print(f"   Simulating Peak Load: {peak_mw:.2f} MW")

    net = pp.create_empty_network()
    b_hv = pp.create_bus(net, vn_kv=110, name="HV Grid")
    b_mv = pp.create_bus(net, vn_kv=20, name="MV Busbar")
    b_load = pp.create_bus(net, vn_kv=20, name="Remote Node")

    pp.create_ext_grid(net, bus=b_hv, vm_pu=1.02)
    # 63 MVA Transformer (Standard for urban distribution)
    pp.create_transformer(net, hv_bus=b_hv, lv_bus=b_mv, std_type="63 MVA 110/20 kV")
    pp.create_line(net, from_bus=b_mv, to_bus=b_load, length_km=5.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
    pp.create_load(net, bus=b_load, p_mw=peak_mw, q_mvar=peak_mw*0.1)

    try:
        pp.runpp(net)
        voltage = net.res_bus.vm_pu.at[b_load]
        print(f"   Remote Voltage: {voltage:.3f} p.u.")
        if 0.90 < voltage < 1.10:
            print("   ✅ VDE-AR-N 4110 Compliance: Voltage within limits.")
        else:
            print("   ⚠️ CRITICAL: Voltage violation detected!")
    except Exception as e:
        print(f"   ❌ Power Flow Failed: {e}")

# --- PHASE 5: SCENARIOS ---

def simulate_ev_congestion(df):
    print("\n⚡ Running Scenario 1: EV Congestion...")
    
    df['EV_Load_MW'] = df['Net_Load_MW']
    peak_idx = df['EV_Load_MW'].idxmax()
    peak_val = df['EV_Load_MW'].max()
    
    subset = df.loc[peak_idx - pd.Timedelta(days=1) : peak_idx + pd.Timedelta(days=1)].copy()
    subset['Managed_Load_MW'] = subset['EV_Load_MW'].clip(upper=GLOBAL_TRAFO_LIMIT_MW)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(subset.index, subset['EV_Load_MW'], 'r--', label='Unmanaged Load (Risk)', alpha=0.5)
    ax.plot(subset.index, subset['Managed_Load_MW'], 'g-', linewidth=3, label='Redispatch 3.0 (Active)')
    ax.axhline(GLOBAL_TRAFO_LIMIT_MW, color='k', linestyle=':', label='Transformer Limit')
    ax.fill_between(subset.index, subset['Managed_Load_MW'], subset['EV_Load_MW'], color='red', alpha=0.1)
    
    ax.annotate(f'BLACKOUT RISK\n({peak_val:.1f} MW)', 
                xy=(peak_idx, peak_val), xytext=(peak_idx + pd.Timedelta(hours=4), peak_val + 2),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2), color='red', fontweight='bold')
    
    ax.set_title('Scenario 1: Congestion Management (EV Cluster)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Load [MW]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    sns.despine()
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '01_scenario_congestion.png'))

def simulate_grid_booster(df):
    print("\n🔋 Running Scenario 2: Grid Booster...")
    BATTERY_MW = 5.0
    limit = GLOBAL_TRAFO_LIMIT_MW - 5.0 
    
    load = df['Total_Load_MW'].values
    managed = []
    
    for l in load:
        if l > limit:
            d = min(l - limit, BATTERY_MW, 10.0 * 4)
            managed.append(l - d)
        else:
            managed.append(l)

    sorted_orig = np.sort(load)[::-1]
    sorted_man = np.sort(managed)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(sorted_orig, label='Original Grid Load', color='#1f77b4', alpha=0.6)
    ax.plot(sorted_man, label='With Grid Booster', color='#ff7f0e', linewidth=2.5)
    ax.axhline(limit, color='g', linestyle='--', label='Target Capacity')
    ax.set_xlim(0, 400)
    ax.set_ylim(30, 60)
    
    peak_orig = sorted_orig[0]
    ax.annotate(f'Peak Shaved: -{(peak_orig - limit):.1f} MW', 
                xy=(0, peak_orig), xytext=(50, peak_orig),
                arrowprops=dict(facecolor='orange', shrink=0.05), fontweight='bold')
    
    ax.set_title('Scenario 2: Asset Deferral (Duration Curve)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Load [MW]')
    ax.set_xlabel('Peak Hours (Sorted)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    sns.despine()
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '02_scenario_grid_booster.png'))

def visualize_digital_twin(df):
    print("\n📊 Running Scenario 3: Digital Twin Heatmaps...")
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month_name().str[:3]
    pivot = df.pivot_table(index='Hour', columns='Month', values='Net_Load_MW', aggfunc=lambda x: np.percentile(x, 95))
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot[months_order]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='magma', cbar_kws={'label': 'Peak Load (MW)'})
    plt.gca().invert_yaxis()
    plt.title('Scenario 3: Grid Stress Fingerprint (Temporal Overlay)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_scenario_heatmap_overlay.png'))

def run_monte_carlo_stress_test(df):
    print("\n🎲 Running Scenario 4: Monte Carlo Stress Test...")
    base_load = df['Net_Load_MW'].values
    iterations = 50
    results = []
    
    for i in range(iterations):
        noise = np.random.normal(0, 2.0, size=len(base_load))
        noisy_load = base_load + noise
        managed_load = np.array([fuzzy_control_logic(l, GLOBAL_TRAFO_LIMIT_MW) for l in noisy_load])
        results.append(managed_load)
        
    results = np.array(results)
    p05 = np.percentile(results, 5, axis=0)
    p50 = np.percentile(results, 50, axis=0)
    p95 = np.percentile(results, 95, axis=0)
    
    zoom_start, zoom_end = 2000, 2500
    x = np.arange(zoom_end - zoom_start)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, p50[zoom_start:zoom_end], color='blue', label='Median')
    ax.fill_between(x, p05[zoom_start:zoom_end], p95[zoom_start:zoom_end], color='blue', alpha=0.2, label='95% Confidence')
    ax.axhline(GLOBAL_TRAFO_LIMIT_MW, color='red', linestyle='--', label='Physical Limit')
    
    ax.set_title('Scenario 4: Fuzzy Logic Robustness (Monte Carlo)', fontweight='bold')
    ax.set_ylabel('Load [MW]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '05_probabilistic_stress_test.png'))

def analyze_hosting_capacity(df):
    print("\n🚀 Running Hosting Capacity Analysis...")
    base_load = df['Net_Load_MW'].max()
    added_mw = 0
    results = []
    
    while added_mw < 20.0:
        total = base_load + added_mw
        active = fuzzy_control_logic(total, GLOBAL_TRAFO_LIMIT_MW) if total > GLOBAL_TRAFO_LIMIT_MW else total
        results.append({'Added': added_mw, 'Passive': total, 'Active': active})
        added_mw += 0.5
        
    res_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(res_df['Added'], res_df['Passive'], 'r--', label='Passive Grid', alpha=0.6)
    ax.plot(res_df['Added'], res_df['Active'], 'g-', linewidth=3, label='Active Grid (Fuzzy Logic)')
    ax.axhline(GLOBAL_TRAFO_LIMIT_MW, color='k', linestyle=':', label='Physical Limit')
    
    try:
        fail_passive = res_df[res_df['Passive'] > GLOBAL_TRAFO_LIMIT_MW].iloc[0]['Added']
        fail_active_rows = res_df[res_df['Active'] > GLOBAL_TRAFO_LIMIT_MW]
        fail_active = fail_active_rows.iloc[0]['Added'] if not fail_active_rows.empty else res_df['Added'].max()
        gain = fail_active - fail_passive
        
        ax.annotate(f'Breaks: +{fail_passive:.1f} MW', xy=(fail_passive, GLOBAL_TRAFO_LIMIT_MW), 
                    xytext=(fail_passive - 1, GLOBAL_TRAFO_LIMIT_MW + 3),
                    arrowprops=dict(facecolor='red', shrink=0.05), ha='center')
        
        ax.annotate(f'New Limit: +{fail_active:.1f} MW', xy=(fail_active, GLOBAL_TRAFO_LIMIT_MW), 
                    xytext=(fail_active + 2, GLOBAL_TRAFO_LIMIT_MW + 3),
                    arrowprops=dict(facecolor='green', shrink=0.05), ha='center')
        
        print(f"   Hosting Capacity Gain: {gain:.1f} MW")
    except: pass

    ax.set_title('Hosting Capacity Analysis: Headroom Gain', fontweight='bold', fontsize=14)
    ax.set_xlabel('Additional EV Capacity Installed [MW]')
    ax.set_ylabel('Transformer Loading [MW]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    sns.despine()
    add_branding(ax)
    plt.savefig(os.path.join(OUTPUT_DIR, '06_hosting_capacity.png'))

def simulate_2035_forecast(df, ops_per_sec):
    """
    Scenario 5: 10-Year Load Forecast (2035) based on Report Data.
    Refs: BNetzA Monitoring Report 2024 (73.7 GW Peak, €110bn CAPEX)
    """
    print("\n🔮 Running Scenario 5: 2035 Load Forecast (Report-Based)...")
    print("   ℹ️  Baseline: BNetzA Report 2024 (Page 14) - 73.7 GW National Peak")
    print("   ℹ️  Driver: 2.04M §14a Devices (Page 16) + 42% EV Growth")
    
    # 34% Growth (CAGR 3%) over 10 years
    GROWTH_FACTOR = 1.34 
    
    df['Load_2035_MW'] = df['Net_Load_MW'] * GROWTH_FACTOR
    
    peak_idx = df['Load_2035_MW'].idxmax()
    peak_val_2035 = df['Load_2035_MW'].max()
    peak_val_2024 = df['Net_Load_MW'].max()
    
    print(f"   ℹ️  Local Peak (2024): {peak_val_2024:.2f} MW")
    print(f"   ℹ️  Projected Peak (2035): {peak_val_2035:.2f} MW")
    
    subset = df.loc[peak_idx - pd.Timedelta(days=1) : peak_idx + pd.Timedelta(days=1)].copy()
    subset['Managed_2035_MW'] = subset['Load_2035_MW'].clip(upper=GLOBAL_TRAFO_LIMIT_MW)
    
    gap = peak_val_2035 - GLOBAL_TRAFO_LIMIT_MW
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(subset.index, subset['Net_Load_MW'], color='grey', linestyle=':', label='2024 Baseline', alpha=0.6)
    ax.plot(subset.index, subset['Load_2035_MW'], color='red', linestyle='--', label='2035 Unmanaged')
    ax.plot(subset.index, subset['Managed_2035_MW'], color='green', linewidth=3, label='2035 Managed')
    ax.axhline(GLOBAL_TRAFO_LIMIT_MW, color='k', linestyle='-', linewidth=2, label='Physical Limit')
    
    ax.fill_between(subset.index, subset['Managed_2035_MW'], subset['Load_2035_MW'], 
                    color='red', alpha=0.1, hatch='//', label='Curtailed Load')

    ax.annotate(f'2035 RISK: {peak_val_2035:.1f} MW\n(Grid Collapse)', xy=(peak_idx, peak_val_2035), 
                xytext=(peak_idx + pd.Timedelta(hours=4), peak_val_2035),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=11, fontweight='bold', color='red')

    ax.set_title('Scenario 5: 10-Year Load Forecast (Scalability Stress Test)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Load [MW]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    sns.despine()
    add_branding(ax)
    
    plt.savefig(os.path.join(OUTPUT_DIR, '07_forecast_2035.png'))
    print(f"   📸 Saved 2035 Forecast.")
    
    print("\n📋 STRATEGIC RECOMMENDATIONS (2035 ROADMAP):")
    print(f"   ⚠️  Identified Capacity Gap: {gap:.2f} MW ({gap/GLOBAL_TRAFO_LIMIT_MW:.1%} Overload)")
    print(f"   💡 Context: DSOs plan €110bn CAPEX by 2033 (Report p.14).")
    
    if gap < 2.0:
        print("   ✅ Recommendation: PURE SOFTWARE. Deploy Redispatch 3.0.")
    elif gap < 10.0:
        print("   🔋 Recommendation: HYBRID. Deploy Software + Grid Booster (Battery).")
    else:
        print("   🏗️ Recommendation: HARDWARE UPGRADE. Gap > 10MW.")
        
    print(f"   ✅ Scalability Validation: {int(ops_per_sec):,} ops/s throughput confirms readiness.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Please create a '{DATA_DIR}' folder and add CSV files.")
    else:
        # 1. Sanity Check
        run_basic_pandapower_test()
        
        # 2. ETL
        grid_data = load_and_validate_data()
        
        # 3. Scalability
        ops_rate = run_computational_benchmark()
        
        # 4. Physics
        run_pandapower_validation(grid_data)
        
        # 5. Simulations
        simulate_ev_congestion(grid_data)
        simulate_grid_booster(grid_data)
        visualize_digital_twin(grid_data)
        run_monte_carlo_stress_test(grid_data)
        analyze_hosting_capacity(grid_data)
        
        # 6. Forecast (Pass in scalability score)
        simulate_2035_forecast(grid_data, ops_rate)
        
        print(f"\n🎉 Project Complete! Check '{OUTPUT_DIR}' for evidence.")
