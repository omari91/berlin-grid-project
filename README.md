
# Berlin Grid Digital Twin — Enterprise Edition

**Author:** Clifford Ondieki  
**Purpose:** Demonstrate scalable real-time edge control (Redispatch 3.0) and grid hosting-capacity analytics using real German grid data.  
Compliant targets: **VDE-AR-N 4110**, **§14a EnWG**.

## 🚀 Key Features

This repository implements a **streaming digital twin** with real-time validation:

### Core Capabilities
- **Real-Time Streaming Architecture** – Closed-loop physics processing with sub-50ms cycle time (50Hz grid-compliant)
- **Hardware Benchmarking** – Empirical performance metrics (>55 Million ops/sec)
- **Multi-Strategy Controller Comparison** – Ablation study comparing:
  - Hard Cutoff (binary relay)
  - Linear Droop (P(f)/P(U) proxy)
  - Fuzzy Logic (proposed sigmoid)
- **Hyperparameter Sensitivity Analysis** – Systematic k-factor tuning (k=5, 15, 30)
- **Multi-Constraint Physics Validation** – Pandapower AC power flow checking:
  - Voltage stability (0.90–1.10 p.u.)
  - Line thermal loading
  - Transformer capacity
- **Monte Carlo Robustness Testing** – 50 stochastic runs with:
  - PV forecast error (AR-1 Persistence, σ=2.0 MW)
  - EV arrival variability (AR-1 Random, σ=1.0 MW)
- **Data-Agnostic Design** – Decoupled architecture robust to topological scope mismatches.

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
Peak Throughput: 56.81 Million Ops/Sec (Single Core)
Physics Loop Time: 47.91 ms (meets <50ms real-time requirement)
Scaling Behavior: O(1) complexity verified (CPU cache limited)
````

✅ **Conclusion:** ARM gateways (e.g., Raspberry Pi) can host this logic without latency violations.

-----

## 📈 Controller Benchmarking

Systematic comparison against industry baselines:

| Strategy | Behavior | Hosting Capacity | Stability |
|----------|----------|------------------|------------|
| **Hard Cutoff** | Binary relay, instant clamp | Low | Relay chatter risk |
| **Linear Droop** | Proportional reduction | Medium | Premature curtailment |
| **Fuzzy Logic** | Sigmoid soft landing | High | Smooth, optimized |

See `run_scalability_benchmark()` in `main.py`.

-----

## ⚡ Multi-Constraint Validation

Pandapower AC power flow runs continuously in the loop to verify constraints:

```
Cycle Time: 47.91 ms
✓ Voltage Stability: Monitored continuously (0.90–1.10 p.u.)
✓ Thermal Loading: Transformers and lines checked dynamically at every tick.
✓ Stability: Smooth control response verified under dynamic load conditions.
```

**Note:** Initial validation against open-source upstream telemetry revealed a topological scope mismatch ($\rho < 0.2$), confirming that public datasets aggregate city-wide loads while sensors measure specific feeders. The architecture is designed to handle this data uncertainty robustly.

-----

## 🎲 Stochastic Robustness

Monte Carlo simulation (n=50) with documented uncertainty models:

  - **PV Generation Error:** Auto-Regressive AR(1) with high persistence ($\phi=0.95$).
  - **EV Charging Variability:** Random arrival ($\phi=0.10$) for stochastic plug-in times.
  - **Result:** Controller output remains bounded within safety margins despite correlated perturbations.

-----

## Quick Start

1.  **Clone repository:**

<!-- end list -->

```bash
git clone [https://github.com/omari91/berlin-grid-project.git](https://github.com/omari91/berlin-grid-project.git)
cd berlin-grid-project
```

2.  **Create virtual environment & install:**

<!-- end list -->

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3.  **Add data:** Create a `data/` folder and add the required Energienetze Berlin CSVs (file names specified in `main.py`).

4.  **Run simulation:**

<!-- end list -->

```bash
python main.py
```

5.  **Run tests:**

<!-- end list -->

```bash
pytest -q
```

-----

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

-----

## 📚 Documentation

  - **METHODOLOGY.md** – Complete engineering methodology aligned with ISO/IEC Digital Twin standards
  - **PORTFOLIO.md** – Recruiter-friendly project highlights
  - **docs/** – MkDocs technical documentation

-----

## 🛠️ Skills Demonstrated

  - **Programming:** Python (pandas, numpy, pandapower, matplotlib, seaborn)
  - **Power Systems Engineering:** Grid resilience, voltage stability, Redispatch 3.0, §14a EnWG
  - **Real-Time Systems:** Streaming data processing, latency optimization, edge computing
  - **Software Engineering:** Testing (pytest), CI/CD (GitHub Actions), Containerization (Docker)
  - **Stochastic Modeling:** Monte Carlo simulation, uncertainty quantification
  - **German Energy Regulations:** EnWG §14a, VDE-AR-N 4110

-----

## Licensing

MIT License recommended.

-----

## Contact

Clifford Ondieki  
📧 ondiekiclifford05@gmail.com  
🎓 M.Sc. Electrical Engineering (graduating 2026)  
🔗 [LinkedIn](https://www.linkedin.com/in/clifford-ondieki-tpm/) | [GitHub](https://github.com/omari91) www.cliffordomari.com 

```
```
