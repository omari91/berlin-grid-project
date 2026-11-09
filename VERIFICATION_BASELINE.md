# Berlin Grid Simulation Framework: Verification Baseline & Reference Metrics

This document formalizes the **golden reference baselines** captured prior to verification improvements. All regression tests enforce result preservation within designated tolerance bands.

---

## 1. Regression Tolerance Bands

| Domain | Metric | Permissible Drift | Rationale |
| :--- | :--- | :--- | :--- |
| **Physics / AC Power Flow** | Bus Voltage ($V_{pu}$) | $\pm 0.5\%$ | Numerical precision of Newton-Raphson solver |
| **Physics / Loading** | Transformer Loading ($I_{\%}$) | $\pm 1.0\%$ | Line and transformer admittance variations |
| **Control Logic** | Soft Sigmoid Curtailment ($P_{curt}$) | $\pm 1.0\%$ | Floating-point exponential evaluation |
| **Stochastic Model** | AR(1) Lag-1 Autocorrelation ($\phi$) | $\pm 5.0\%$ | Sample autocorrelation windowing |
| **Real-Time Jitter** | P99 Microkernel Jitter | $\le 10.0\ \mu\text{s}$ | Hard real-time edge execution gate |
| **Throughput** | Vectorized Control Ops/Sec | $\ge 10.0\ \text{MOps/s}$ | Streaming batch scalability |
| **IEC 61850 Protocol** | Protection Trip / GOOSE State | $0\%$ (Exact match) | Safety-critical relay logic |

---

## 2. Physics & Power Flow Golden Baseline

AC Newton-Raphson power flow solutions across varying load points on the Berlin Grid model:

| Operating Regime | Net Active Load | Min Voltage ($V_{min}$) | Trafo Loading ($I_{trafo}$) | GOOSE Protection State |
| :--- | :--- | :--- | :--- | :--- |
| **Light Load** | 10.0 MW | $1.0019\ \text{p.u.}$ | $16.41\%$ | Normal (`Pos.stVal = True`) |
| **Medium Load** | 30.0 MW | $0.9569\ \text{p.u.}$ | $51.82\%$ | Normal (`Pos.stVal = True`) |
| **Nominal Limit** | 45.0 MW | $0.9156\ \text{p.u.}$ | $81.32\%$ | Normal (`Pos.stVal = True`) |
| **Severe Overload** | 55.0 MW | $0.8828\ \text{p.u.}$ | $103.12\%$ | **TRIP** (`Pos.stVal = False`) |

---

## 3. Vectorized Fuzzy Controller Golden Baseline

Sigmoid control action $\alpha(S) = \frac{1}{1 + e^{-k(S - S_{ref})}}$ with $k=15$, $S_{ref}=0.95$, and rated limit $P_{max} = 45.0\ \text{MW}$:

| Stress Level ($S$) | Input Load ($P_{in}$) | Setpoint Output ($P_{out}$) | Curtailment ($\%$) | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **0.00** | 0.00 MW | 0.0000 MW | 0.00% | Idle |
| **0.50** | 22.50 MW | 22.4947 MW | 0.02% | Linear passthrough |
| **0.80** | 36.00 MW | 35.3135 MW | 1.91% | Anticipatory dampening |
| **0.95** | 42.75 MW | 38.4750 MW | 10.00% | Midpoint soft curtailment |
| **1.00** | 45.00 MW | 38.8874 MW | 13.58% | Continuous boundary action |
| **1.05** | 47.25 MW | 39.5239 MW | 16.35% | Progressive tightening |
| **1.20** | 54.00 MW | 43.4482 MW | 19.54% | Approaching 20% max dimming |
| **1.50** | 67.50 MW | 45.0000 MW | 33.33% | Hard limit clamp |
| **2.00** | 90.00 MW | 45.0000 MW | 50.00% | Severe congestion clamp |

### Efficiency Convergence Proof
In deep congestion ($P_{max} = 22.0\ \text{MW}$), total energy delivered is **98.83%** of theoretical binary hard-cutoff optimal, satisfying the $\ge 97\%$ efficiency requirement while eliminating binary relay oscillation chatter.

---

## 4. Real-Time Performance Baseline

Evaluated on single-core execution with warm-start caching:

| Metric | Measured Baseline | Target Gate | Status |
| :--- | :--- | :--- | :--- |
| **P50 Jitter** | $0.54\ \mu\text{s}$ | $< 5.0\ \mu\text{s}$ | Pass |
| **P99 Jitter** | $0.71\ \mu\text{s}$ | $< 10.0\ \mu\text{s}$ | Pass |
| **P99.9 Jitter** | $1.83\ \mu\text{s}$ | $< 50.0\ \mu\text{s}$ | Pass |
| **Vectorized Throughput (10k ops)** | $154.2\ \text{MOps/s}$ | $> 10.0\ \text{MOps/s}$ | Pass |
| **Vectorized Throughput (100k ops)** | $118.8\ \text{MOps/s}$ | $> 10.0\ \text{MOps/s}$ | Pass |
| **Vectorized Throughput (1M ops)** | $133.6\ \text{MOps/s}$ | $> 10.0\ \text{MOps/s}$ | Pass |

---

## 5. Stochastic Uncertainty Baseline

Parameters: $N = 1000$ samples, seed = 42, $\sigma = 2.0\ \text{MW}$, $\phi = 0.95$:

- **Empirical Mean**: $0.250\ \text{MW}$
- **Empirical Standard Deviation**: $1.976\ \text{MW}$
- **Lag-1 Autocorrelation ($\hat{\phi}$)**: $0.9510$ (relative error: $+0.1\%$ from nominal $0.95$)
- **5th Percentile ($P_5$)**: $-2.944\ \text{MW}$
- **95th Percentile ($P_{95}$)**: $+3.819\ \text{MW}$

---

## 6. IEC 61850 Substation Integration Baseline

- **MMS Client-Server Telemetry**: Provides cyclic metering objects `TotW`, `TotVAr`, `PPV.phsA`, `A.phsA`.
- **PDIS Protection Threshold**: $0.90\ \text{p.u.}$ undervoltage trip boundary.
- **GOOSE Multicast**: $< 4.0\ \text{ms}$ peer-to-peer trip command changing `XCBR.Pos.stVal` from `True` (Closed) to `False` (Open).

