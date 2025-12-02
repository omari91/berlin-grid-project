# Berlin Grid Digital Twin — Enterprise Edition

**Author:** Clifford Ondieki  
**Purpose:** Demonstrate scalable edge control (Redispatch 3.0) and grid hosting-capacity analytics using real German grid data.  
Compliant targets: **VDE-AR-N 4110**, **§14a EnWG**.

This repo includes:
- Typed Python modules + Pydantic data models
- Pandapower physics validation
- Fuzzy (sigmoid) dimming control algorithm
- Scenario simulations (congestion, grid booster, Monte Carlo, 2035 forecast)
- PowerFactory COM integration placeholder for Windows workflows
- Tests, CI pipeline, Dockerfile and docs (mkdocs)

---

## Quick Start

1. Clone repository:
```bash
git clone https://github.com/omari91/berlin-grid-project.git
cd berlin-grid-project
```

2. Create virtual environment & install:
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Add data:  
Create a `data/` folder and add the required Energienetze Berlin CSVs (sample names in `src/main.py`).

4. Run tests:
```bash
pytest -q
```

5. Run the pipeline:
```bash
python -m src.main
```

> Windows: to use PowerFactory integration, install DIgSILENT PowerFactory and set `POWERFACTORY_PATH` per instructions in `src/powerfactory_integration.py`.

---

## Project Structure

```
berlin-grid-project/
├── data/                 # (excluded from repo) raw CSVs
├── docs/                 # mkdocs documentation
├── output/               # generated graphs & artifacts
├── src/                  # typed source modules
├── tests/                # pytest tests
├── .github/
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── mkdocs.yml
└── README.md
```

**Note:** The root `main.py` contains the original working analysis script. The `src/` directory provides a refactored, enterprise-grade modular version with CI/CD pipeline.

---

## Licensing

MIT License recommended.

---

## Contact

Clifford Ondieki — ondiekiclifford05@gmail.com
