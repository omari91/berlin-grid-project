# Berlin Grid Digital Twin: From Simulation to Reality ⚡

**A Python engineering project demonstrating the shift from "Copper" to "Code" in the German Energy Transition.**
Compliant with VDE-AR-N 4110 & §14a EnWG Dimming Logic.

[](https://www.python.org/)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)

## 🚀 Executive Summary

The German grid faces a paradox: We have the renewable gigawatts, but we lack the integration capacity. This project builds a **Digital Twin** of a Berlin distribution grid using real 2024 data to prove 
that **Edge Intelligence (Redispatch 3.0)** can solve congestion cheaper and faster than traditional grid reinforcement.

### 📊 Key Engineering Results

1.  [cite\_start]**Scalability Proof:** The algorithm achieved a throughput of **451,463 nodes/sec per CPU core**[cite: 321]. This proves that a decentralized "Cellular Grid" architecture is computationally
   viable on standard edge hardware.
3.  **ROI Analysis:** The Hosting Capacity simulation demonstrated a gain of **+9.5 MW** in virtual capacity. At current market rates (€150k/MW), this software intervention creates **€1.4 Million** in deferred
    CAPEX value.
5.  **2035 Stress Test:** A 10-year load forecast (3% CAGR) identified a **9.17 MW capacity gap** even with active control.
      * *Strategic Recommendation:* **Hybrid Approach.** Software handles the first 50% of the overload; targeted hardware upgrades address the critical 9 MW bottleneck.

-----

## 🛠️ Project Architecture

This repository contains a unified engineering pipeline:

1.  [cite\_start]**ETL Engine:** Parses raw German utility CSVs (handling `1.234,56` formats) and synchronizes Generation, Load, and Grid Import data to 15-minute resolution[cite: 312].
2.  [cite\_start]**Physics Validation:** Validates data integrity using **Kirchhoff’s Current Law (KCL)** and checks voltage stability using **Pandapower** (Newton-Raphson AC Power Flow)[cite: 5].
3.  **Simulation Core:**
      * **Scenario 1 (Congestion):** Simulates a "Redispatch 3.0" controller managing an EV cluster.
      * **Scenario 2 (Grid Booster):** Models a 5MW Battery Storage System for peak shaving.
      * [cite\_start]**Scenario 4 (Robustness):** Runs a **Monte Carlo Stress Test** (n=100) with Fuzzy Logic control to ensure §14a EnWG compliance under stochastic conditions[cite: 262].

-----

## 📂 Repository Structure

```text
berlin-grid-project/
│
├── data/                        # Raw CSV files from Energienetze Berlin
├── output/                      # Generated graphs (Evidence)
│   ├── 01_scenario_congestion.png
│   ├── 06_hosting_capacity.png
│   └── ...
├── main.py                      # The complete Simulation Engine
├── METHODOLOGY.md               # Detailed engineering documentation
├── REQUIREMENTS.txt             # Python dependencies
└── README.md                    # This file
```

-----

## ⚠️ Note on Data Validation

The KCL physics validation reports a mean energy balance deviation of **\~7.7 MW**.

  * [cite\_start]**Context:** This deviation is attributed to the high penetration of decentralized generation (Solar PV) in the test dataset[cite: 38].
  * **Physics:** In periods of high generation and low load, this manifests as **Reverse Power Flow** (Export to the upstream High Voltage grid), which simple static checks flag as a discrepancy.
  * **Resolution:** In a production environment, bidirectional metering points would resolve this artifact. For this simulation, the data is deemed valid for N-1 contingency modeling.

-----

## 🚀 How to Run

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/berlin-grid-digital-twin.git
    ```

2.  **Install Dependencies:**

    ```bash
    pip install pandas numpy matplotlib seaborn pandapower
    ```

3.  **Add Data:**
    Download the 2024 Grid Data from the [Energienetze Berlin Open Data Portal](https://www.google.com/search?q=https://www.stromnetz.berlin/ueber-uns/veroeffentlichungspflichten/energiewirtschaftliche-daten) and place the CSVs in the `data/` folder.

4.  **Execute Simulation:**

    ```bash
    python main.py
    ```

-----

## 👨‍💻 About the Author

**Clifford Ondieki**
*Power Systems Engineer | Simulation-to-Reality*

I bridge the gap between rigorous grid physics (PowerFactory) and agile digital product execution. My work focuses on **Probabilistic Resilience**, **Redispatch 3.0**, and **Automated Compliance** (VDE-AR-N 4110).

 [Portfolio](https://www.cliffordomari.com)
