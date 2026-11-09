"""Unit tests for Prometheus OpenMetrics Collector & Exporter."""
import pytest
from src.observability import GridMetricsCollector


def test_metrics_collector_record_and_export():
    """Verify metrics collector records telemetry and formats standard Prometheus metrics."""
    collector = GridMetricsCollector(site_id="Berlin-01")

    collector.record_cycle(
        voltage_pu=0.985,
        trafo_loading_pct=65.4,
        latency_us=4.2,
        line_loading_pct=22.1,
        circuit_breaker_state="CLOSED",
        tripped=False,
    )

    prom_text = collector.generate_prometheus_text()

    assert 'grid_bus_voltage_pu{site="Berlin-01"} 0.9850' in prom_text
    assert 'grid_trafo_loading_percent{site="Berlin-01"} 65.40' in prom_text
    assert 'grid_circuit_breaker_state{site="Berlin-01"} 0' in prom_text
    assert 'grid_protection_trips_total{site="Berlin-01"} 0' in prom_text
    assert 'grid_control_cycles_total{site="Berlin-01"} 1' in prom_text


def test_metrics_collector_trip_and_breaker_open():
    """Verify metrics collector increments trip counters and maps breaker state codes."""
    collector = GridMetricsCollector(site_id="Berlin-02")

    collector.record_cycle(
        voltage_pu=0.88,
        trafo_loading_pct=104.2,
        latency_us=12.5,
        circuit_breaker_state="OPEN",
        tripped=True,
    )

    prom_text = collector.generate_prometheus_text()
    assert 'grid_circuit_breaker_state{site="Berlin-02"} 2' in prom_text
    assert 'grid_protection_trips_total{site="Berlin-02"} 1' in prom_text

