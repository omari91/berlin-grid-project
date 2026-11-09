"""Prometheus & OpenMetrics Telemetry Exporter for Grid Edge Digital Twins.

Implements Stage A Observability:
- Bus voltages (Gauge)
- Transformer & line loadings (Gauge)
- Control loop latency histogram (Summary/Histogram)
- Circuit breaker state (Gauge: 0=CLOSED, 1=HALF_OPEN, 2=OPEN)
- Protection trip counters (Counter)
"""
from dataclasses import dataclass, field
import time
from typing import Any, Optional


class GridMetricsCollector:
    """Collects and formats Prometheus OpenMetrics telemetry for grid monitoring NOCs."""

    def __init__(self, site_id: str = "Berlin-01") -> None:
        self.site_id = site_id

        # Gauges
        self.voltage_pu: float = 1.00
        self.trafo_loading_pct: float = 0.0
        self.line_loading_pct: float = 0.0
        self.circuit_breaker_state_code: int = 0  # 0: CLOSED, 1: HALF_OPEN, 2: OPEN

        # Counters
        self.total_cycles: int = 0
        self.protection_trips_total: int = 0
        self.solver_fallbacks_total: int = 0

        # Latency statistics (µs)
        self.latencies_us: list[float] = []
        self.max_latency_history: int = 1000

    def record_cycle(
        self,
        voltage_pu: float,
        trafo_loading_pct: float,
        latency_us: float,
        line_loading_pct: float = 0.0,
        circuit_breaker_state: str = "CLOSED",
        tripped: bool = False,
        fallback_used: bool = False,
    ) -> None:
        """Record measurements and state from a digital twin control cycle."""
        self.total_cycles += 1
        self.voltage_pu = float(voltage_pu)
        self.trafo_loading_pct = float(trafo_loading_pct)
        self.line_loading_pct = float(line_loading_pct)

        state_map = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}
        self.circuit_breaker_state_code = state_map.get(circuit_breaker_state, 0)

        if tripped:
            self.protection_trips_total += 1
        if fallback_used:
            self.solver_fallbacks_total += 1

        self.latencies_us.append(float(latency_us))
        if len(self.latencies_us) > self.max_latency_history:
            self.latencies_us.pop(0)

    def generate_prometheus_text(self) -> str:
        """Generate standard Prometheus OpenMetrics text format."""
        avg_latency = (
            sum(self.latencies_us) / len(self.latencies_us) if self.latencies_us else 0.0
        )
        labels = f'site="{self.site_id}"'

        lines = [
            f"# HELP grid_bus_voltage_pu Minimum nodal voltage in per-unit",
            f"# TYPE grid_bus_voltage_pu gauge",
            f"grid_bus_voltage_pu{{{labels}}} {self.voltage_pu:.4f}",
            "",
            f"# HELP grid_trafo_loading_percent Transformer thermal loading percentage",
            f"# TYPE grid_trafo_loading_percent gauge",
            f"grid_trafo_loading_percent{{{labels}}} {self.trafo_loading_pct:.2f}",
            "",
            f"# HELP grid_circuit_breaker_state Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
            f"# TYPE grid_circuit_breaker_state gauge",
            f"grid_circuit_breaker_state{{{labels}}} {self.circuit_breaker_state_code}",
            "",
            f"# HELP grid_protection_trips_total Cumulative protection trip operations",
            f"# TYPE grid_protection_trips_total counter",
            f"grid_protection_trips_total{{{labels}}} {self.protection_trips_total}",
            "",
            f"# HELP grid_control_cycle_latency_us Average control cycle execution latency in microseconds",
            f"# TYPE grid_control_cycle_latency_us gauge",
            f"grid_control_cycle_latency_us{{{labels}}} {avg_latency:.2f}",
            "",
            f"# HELP grid_control_cycles_total Total simulation and control cycles executed",
            f"# TYPE grid_control_cycles_total counter",
            f"grid_control_cycles_total{{{labels}}} {self.total_cycles}",
            "",
        ]
        return "\n".join(lines)

