# Test Execution & Verification Guide

This repository implements a multi-tiered verification test suite ensuring unit correctness, strict type safety, physics consistency, real-time performance, and distributed resilience.

---

## 1. Quick Start

Run the entire verification suite with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Run static type analysis:

```bash
mypy src --config-file mypy.ini
```

---

## 2. Test Suite Breakdown

The test suite consists of 11 modules organized into verification tiers:

| Test Module | Verification Tier | What It Validates |
| :--- | :--- | :--- |
| `tests/test_baselines.py` | Baseline Preservation | AC power flow voltages ($\pm 0.5\%$), loading ($\pm 1\%$), fuzzy curtailment ($\pm 1\%$), AR(1) autocorrelation ($\pm 5\%$), and IEC 61850 trip transitions. |
| `tests/test_physics.py` | Physics Correctness | Newton-Raphson convergence, voltage limits ($0.90 \le V_{pu} \le 1.10$), dynamic line length drop, transformer overload protection, solver divergence handling. |
| `tests/test_controller.py` | Control Logic | Soft sigmoid monotonicity, dimming bounds ($0.8 \le \text{dimming} \le 1.0$), vectorization parity, hard cutoff ablation. |
| `tests/test_datalayer.py` | Data Ingestion | German float notation parsing, synthetic profile generation, multi-CSV ingestion. |
| `tests/test_iec61850.py` | Substation Automation | MMS metering (`MMXU`), GOOSE breaker switching (`XCBR`), undervoltage protection relay (`PDIS`), cyclic station execution. |
| `tests/test_performance.py` | Real-Time Latency | P50, P99 ($< 10\ \mu\text{s}$), P99.9 jitter analysis, vectorized throughput ($> 10\ \text{MOps/s}$). |
| `tests/test_stochastic.py` | Monte Carlo & Convergence | AR(1) noise generation, 50-run Monte Carlo 95% CI voltage stability, 97% efficiency convergence verification. |
| `tests/test_resilience.py` | Single-Site Resilience | Solver circuit breaker (tripping, fallback, half-open recovery), SCADA MMS timeout/staleness handling, VDE-AR-N 4110 audit logging with SHA-256 signatures. |
| `tests/test_distribution.py` | Distributed Coordination | Substation telemetry sync, Lamport clock incrementation, Last-Write-Wins conflict resolution, proportional redispatch allocation. |
| `tests/test_deployment.py` | Zero-Downtime Deployment | Hot update lifecycle (IDLE $\to$ STAGED $\to$ WARMED_UP $\to$ ACTIVE), sub-100µs atomic swap, guard metric evaluation, automatic rollback on degradation. |
| `tests/test_chaos_suite.py` | Chaos Engineering | 60-second SCADA dropout resilience, 3-substation network partition & healing without split-brain, mid-cycle hot update fault injection. |

---

## 3. Running Specific Subsets

### Regression Suite Only
```bash
pytest -m regression -v
```

### Real-Time Performance Benchmarks
```bash
pytest -m benchmark -v
```

### Chaos & Resilience Tests
```bash
pytest -m chaos -v
```

---

## 4. Continuous Integration (CI/CD)

The GitHub Actions workflow (`.github/workflows/ci.yml`) enforces the following gates on every pull request:
1. Complete test execution with XML and terminal coverage reports (`--cov=src`).
2. Strict type verification with `mypy` without error suppression.
3. Zero tolerance for regression outside the specified tolerance bands.
