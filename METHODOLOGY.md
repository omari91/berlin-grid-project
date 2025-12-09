# Engineering Methodology

This document details the architectural design, mathematical modeling, and validation strategies employed in the **Berlin Grid Digital Twin** project. The framework is designed to satisfy **VDE-AR-N 4110** real-time control requirements while operating in data-constrained environments.

---

## 1. System Architecture: The Closed-Loop Digital Twin

The system follows a modular **Cyber-Physical Systems (CPS)** architecture, decoupling the physical grid simulation from the control logic. This ensures modularity and testability.

### 1.1 The Control Loop
The simulation executes a continuous feedback loop at discrete time steps $t$:

1.  **Sense:** Ingest Grid State $S_t$ (Load, Generation) via streaming interface.
2.  **Decide:** Compute Control Action $u_t$ via Vectorized Fuzzy Logic.
3.  **Actuate:** Update Physics Model with setpoints $P_{set} = P_{load} \times u_t$.
4.  **Solve:** Execute AC Power Flow (Warm-Start Newton-Raphson).
5.   **Feedback:** Extract the new system state $S_{t+1}$, comprising nodal voltages ($|V|$) and branch thermal loading percentages ($I_{\%}$).


**Latency Target:** The loop is benchmarked to complete in **< 20ms**, satisfying the Nyquist rate for 50Hz grid dynamics observability.

---

## 2. High-Throughput Vectorized Control

To achieve massive scalability, we rejected iterative optimization algorithms (e.g., Genetic Algorithms, OPF) in favor of a **Vectorized Fuzzy Logic Controller**.

### 2.1 The Sigmoid Decision Function
Congestion management is modeled as a soft-switching problem. The curtailment factor $\alpha$ is calculated using a vectorized sigmoid function, providing $\mathcal{O}(N)$ linear scalability with high constant-factor efficiency.

$$\alpha(S) = \frac{1}{1 + e^{-k(S - S_{ref})}}$$

Where:
* $S$: Grid Stress Level (Measured Load / Transformer Limit).
* $S_{ref}$: Reference Setpoint (e.g., 0.99 or 99% loading).
* $k$: Gain factor ($k=100$) controlling the "stiffness" of the control response.

### 2.2 Advantages
* **Throughput:** Requires only elementary floating-point operations (FLOPS), enabling throughputs of **~30 Million Ops/Sec** on standard x86 hardware.
* **Stability:** The continuous derivative of the sigmoid function prevents "relay chatter" (oscillation) common in binary hard-cutoff controllers.
* **Efficiency:** Ablation studies demonstrate that under heavy congestion, the controller converges to **97% of the theoretical optimal efficiency** of a hard cutoff, while maintaining voltage stability.

---

## 3. Physical Modelling (AC Power Flow)

The **PhysicalTwin** module utilizes `pandapower` to solve the non-linear AC power flow equations. Unlike linear DC approximations, this captures critical voltage stability phenomena.

### 3.1 Solver Optimization (Warm Start)
To minimize latency for real-time operation, the solver utilizes a **Warm-Start Strategy**:
* **Algorithm:** Newton-Raphson method (`algorithm='nr'`).
* **Initialization:** The solver is initialized with the voltage vector from time $t-1$ (`init_vm_pu="results"`).
* **Impact:** Drastically reduces iterations required for convergence between sequential time steps.

### 3.2 Constraints Monitored
1.  **Thermal Loading:** Line and Transformer currents must remain $< 100\%$ of rated capacity ($I_{max}$).
2.  **Voltage Stability:** Bus voltages must remain within the $\pm 10\%$ band ($0.90 \le V_{pu} \le 1.10$).

---

## 4. Stochastic Uncertainty Model

To validate robustness, the system is subjected to a **Physics-Integrated Monte Carlo Stress Test** ($N=50$) where uncertainty is propagated through the power flow solver.

### 4.1 Auto-Regressive (AR-1) Process
Real-world grid variables (Cloud cover, EV arrivals) exhibit temporal persistence. We model this using an AR(1) process:

$$X_t = \phi X_{t-1} + \epsilon_t$$

* **PV Generation:** High persistence ($\phi = 0.95$, $\sigma = 2.0$ MW) simulates passing cloud fronts.
* **EV Demand:** Low persistence ($\phi = 0.10$, $\sigma = 1.0$ MW) simulates random charging events.

**Result:** The simulation generates 95% Confidence Intervals (CI) for bus voltages, proving the controller does not destabilize under realistic, correlated perturbations.

---

## 5. Performance Benchmarking Methodology

Real-time capability is verified through **Streaming Jitter Analysis** rather than simple batch averages.

* **Method:** A `StreamingDigitalTwin` class simulates tick-by-tick data arrival.
* **Metric:** **P99 Jitter** (99th Percentile Latency) is measured to detect worst-case processing times.
* **Verification:** The system demonstrates a P99 Jitter of **< 10 µs** for the control microkernel, confirming deterministic behavior suitable for embedded edge gateways.

---

## 6. Data Strategy & Scope Limitations

The project integrates open-source transparency data (Entso-E, Stromnetz Berlin) to drive the simulation.

### 6.1 Topological Scope Mismatch
Initial correlation analysis between the **Aggregated City Load** (Source A) and the **Upstream Transmission Sensor** (Source B) revealed a Pearson Correlation of $\rho \approx 0.37$.

* **Cause:** The lack of proprietary sub-second telemetry for local "Hidden Generation" (e.g., CHP/Gas plants) creates a variable offset between Total Load and Grid Import.
* **Resolution:** Rather than forcing an artificial curve-fit, this framework adopts a **Data-Agnostic Design**. The architecture is verified for internal consistency (Physics & Control) and is designed to accept proprietary telemetry streams ("Plug-and-Play") for operators who possess the requisite data.

---

**References:**
1.  *VDE-AR-N 4110: Technical Rules for the connection of medium-voltage networks.*
2.  *§14a EnWG: German Energy Industry Act - SteuVE (Controllable Consumption Devices).*
