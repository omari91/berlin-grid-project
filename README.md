# Berlin Grid Digital Twin — Enterprise Edition

**Author:** Clifford Ondieki  
**Purpose:** Demonstrate scalable real-time edge control (Redispatch 3.0) and grid hosting-capacity analytics using real German grid data.  
Compliant targets: **VDE-AR-N 4110**, **§14a EnWG**.

## 🚀 Key Features

This repository implements a **streaming digital twin** with real-time validation:

### Core Capabilities
- **Real-Time Streaming Architecture** – Tick-by-tick data processing with sub-20ms latency (50Hz grid-compliant)
- **Hardware Benchmarking** – Empirical performance metrics (avg latency: 2.22 µs, P99 jitter: 2.65 µs, ~450k ops/sec)
- **Multi-Strategy Controller Comparison** – Ablation study comparing:
  - Hard Cutoff (binary relay)
  - Linear Droop (P(f)/P(U) proxy)
  - Fuzzy Logic (proposed sigmoid)
- **Hyperparameter Sensitivity Analysis** – Systematic k-factor tuning (k=5, 15, 30)
- **Multi-Constraint Physics Validation** – Pandapower AC power flow checking:
  - Voltage stability (0.90–1.10 p.u.)
  - Line thermal loading
  - Transformer capacity
- **Monte Carlo Robustness Testing** – 100 stochastic runs with:
  - PV forecast error (Gaussian, σ=2.0 MW)
  - EV arrival variability (Uniform, [-1, 1] MW)
- **2035 Strategic Forecasting** – Gap-based intervention planning (software/hybrid/hardware)

### Technical Stack
- Typed Python modules with Pydantic data models
- Pandapower for AC network simulation
- Fuzzy (sigmoid) smoothing control algorithm
- CI/CD pipeline (pytest, mypy, Docker)
- Comprehensive documentation (METHODOLOGY.md)

---

## ⏱️ Real-Time Performance

The `StreamingDigitalTwin` class validates edge-readiness:

```python
# Measured on x86_64 Linux
Average Latency: 2.22 µs
P99 Jitter: 2.65 µs
Throughput: 450,379 Operations/Sec (Single Core)
Grid Cycle Compliance: <0.02% of 20ms 50Hz cycle
```

✅ **Conclusion:** ARM gateways (e.g., Raspberry Pi) can host this logic without latency violations.

---

## 📈 Controller Benchmarking

Systematic comparison against industry baselines:

| Strategy | Behavior | Hosting Capacity | Stability |
|----------|----------|------------------|------------|
| **Hard Cutoff** | Binary relay, instant clamp | Low | Relay chatter risk |
| **Linear Droop** | Proportional reduction | Medium | Premature curtailment |
| **Fuzzy Logic** | Sigmoid soft landing | High | Smooth, optimized |

See `compare_baselines()` and `sensitivity_analysis()` functions in `main.py`.

---

## ⚡ Multi-Constraint Validation

Pandapower AC power flow reveals hidden bottlenecks:

```
Peak Load: 48.2 MW
✓ Voltage: 0.961 p.u. (within 0.90–1.10 range)
✓ Transformer: 67.0% of 63 MVA capacity
⚠️ Line Loading: 144.9% (thermal overload on NA2XS2Y cable)
```

**Insight:** Software (Redispatch 3.0) must be paired with targeted cable reinforcement.

---

## 🎲 Stochastic Robustness

Monte Carlo simulation (n=100) with documented uncertainty models:

- **PV Generation Error:** Normal(μ=0, σ=2.0 MW) based on day-ahead RMSE
- **EV Charging Variability:** Uniform([-1, 1] MW) for random plug-in times
- **Result:** Controller output remains bounded within 95% confidence interval despite ±2 MW perturbations

---

## Quick Start

1. **Clone repository:**
```bash
git clone https://github.com/omari91/berlin-grid-project.git
cd berlin-grid-project
```

2. **Create virtual environment & install:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Add data:**  
Create a `data/` folder and add the required Energienetze Berlin CSVs (file names specified in `main.py`).

4. **Run simulation:**
```bash
python main.py
```

5. **Run tests:**
```bash
pytest -q
```

---

## Project Structure

```
berlin-grid-project/
├── data/                   # (excluded from repo) raw CSVs
├── docs/                   # mkdocs documentation
├── output/                 # generated graphs & artifacts
├── src/                    # typed source modules
├── tests/                  # pytest tests
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile
├── METHODOLOGY.md          # Academic methodology (6 sections)
├── PORTFOLIO.md            # Portfolio highlights
├── requirements.txt
├── requirements-dev.txt
├── main.py                 # Golden master (working analysis)
├── mkdocs.yml
└── README.md
```

**Note:** The root `main.py` contains the validated analysis script with all features described above. The `src/` directory provides a refactored, enterprise-grade modular version with CI/CD.

---

## 📚 Documentation

- **METHODOLOGY.md** – Complete engineering methodology aligned with ISO/IEC Digital Twin standards
- **PORTFOLIO.md** – Recruiter-friendly project highlights
- **docs/** – MkDocs technical documentation

---

## 🛠️ Skills Demonstrated

- **Programming:** Python (pandas, numpy, pandapower, matplotlib, seaborn)
- **Power Systems Engineering:** Grid resilience, voltage stability, Redispatch 3.0, §14a EnWG
- **Real-Time Systems:** Streaming data processing, latency optimization, edge computing
- **Software Engineering:** Testing (pytest), CI/CD (GitHub Actions), Containerization (Docker)
- **Stochastic Modeling:** Monte Carlo simulation, uncertainty quantification
- **German Energy Regulations:** EnWG §14a, VDE-AR-N 4110

---

## Licensing

MIT License recommended.

---

## Contact

Clifford Ondieki  
📧 ondiekiclifford05@gmail.com  
🎓 M.Sc. Electrical Engineering (graduating 2026)  
🔗 [LinkedIn](https://www.linkedin.com/in/clifford-ondieki-tpm/) | [GitHub](https://github.com/omari91) www.cliffordomari.com
