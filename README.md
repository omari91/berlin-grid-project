Markdown

# Berlin Grid Digital Twin: From Simulation to Reality ⚡

**A Python engineering project demonstrating the shift from "Copper" to "Code" in the German Energy Transition.**

Compliant with VDE-AR-N 4110 & §14a EnWG Dimming Logic.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 🚀 Executive Summary
The German grid faces a paradox: We have the renewable gigawatts, but we lack the integration capacity. This project builds a **Digital Twin** of a Berlin distribution grid using real 2024 data to prove that **Edge Intelligence (Redispatch 3.0)** can solve congestion cheaper and faster than traditional grid reinforcement.

### 📊 Key Engineering Results
1.  **Scalability Proof:** The algorithm achieved a throughput of **>450,000 nodes/sec per CPU core**. This proves that a decentralized "Cellular Grid" architecture is computationally viable on standard edge hardware (e.g., WAGO/Phoenix Contact PLCs).
2.  **ROI Analysis:** The Hosting Capacity simulation demonstrated a gain of **+9.5 MW** in virtual capacity. At current market rates (€150k/MW), this software intervention creates **€1.4 Million** in deferred CAPEX value per substation.
3.  **The 2035 Stress Test:** Using the **Bundesnetzagentur Monitoring Report 2024** growth data (3% CAGR), we simulated the 2035 load profile.
    * *Result:* A **9.17 MW capacity gap** remains even with active control.
    * *Strategic Recommendation:* **Hybrid Approach.** Software handles the first 50% of the overload; targeted hardware upgrades address the critical bottleneck.

---

## 🛠️ Project Architecture

This repository contains a unified engineering pipeline:

1.  **ETL Engine:** Parses raw German utility CSVs (handling `1.234,56` formats) and synchronizes Generation, Load, and Grid Import data to 15-minute resolution.
2.  **Physics Validation:** Validates data integrity using **Kirchhoff’s Current Law (KCL)** and checks voltage stability using **Pandapower** (Newton-Raphson AC Power Flow).
3.  **Simulation Core:**
    * **Scenario 1 (Congestion):** Simulates a "Redispatch 3.0" controller managing an EV cluster.
    * **Scenario 2 (Grid Booster):** Models a 5MW Battery Storage System for peak shaving.
    * **Scenario 4 (Robustness):** Runs a **Monte Carlo Stress Test** (n=100) with Fuzzy Logic control to ensure §14a EnWG compliance under stochastic conditions.
    * **Scenario 5 (2035 Forecast):** Projects the €110bn grid expansion challenge based on federal reporting data.

---

## 📂 Repository Structure

```text
berlin-grid-digital-twin/
│
├── data/                        # Raw CSV files (User must populate this)
├── output/                      # Generated graphs (Evidence)
│   ├── 01_scenario_congestion.png
│   ├── 07_forecast_2035.png
│   └── ...
├── main.py                      # The Master Simulation Engine
├── METHODOLOGY.md               # Detailed engineering documentation (PDF-linked)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
⚠️ Note on Data Validation
The KCL physics validation reports a mean energy balance deviation of ~7.7 MW.

Context: This deviation is attributed to the high penetration of decentralized generation (Solar PV) in the test dataset.

Physics: In periods of high generation and low load, this manifests as Reverse Power Flow (Export to the upstream High Voltage grid), which simple static checks flag as a discrepancy.

Resolution: In a production environment, bidirectional metering points would resolve this artifact. For this simulation, the data is deemed valid for N-1 contingency modeling.

🚀 How to Run
Clone the Repository:

Bash

git clone [https://github.com/yourusername/berlin-grid-digital-twin.git](https://github.com/yourusername/berlin-grid-digital-twin.git)
Install Dependencies:

Bash

pip install pandas numpy matplotlib seaborn pandapower
Add Data: The script will automatically create a data/ folder on the first run. Download the 2024 Grid Data from the Energienetze Berlin Open Data Portal and place the CSVs inside.

Execute Simulation:

Bash

python main.py
🔗 Related Research
This project is part of a broader Power Systems Engineering portfolio:

Optimization Engine: ev-optimization-nsga2 (Deriving the Fuzzy Logic parameters)

RMS Modeling: grid-modeling-powerfactory (Transient stability analysis)

👨‍💻 About the Author
Clifford Ondieki Power Systems Engineer | Simulation-to-Reality

I bridge the gap between rigorous grid physics (PowerFactory) and agile digital product execution. My work focuses on Probabilistic Resilience, Redispatch 3.0, and Automated Compliance (VDE-AR-N 4110).

LinkedIn Profile | Portfolio(www.cliffordomari.com)
