"""Real-Time Performance Benchmarking & Jitter Verification Harness."""
import time
import pytest
import numpy as np

from main import Controller, StreamingDigitalTwin, GLOBAL_TRAFO_LIMIT_MW


@pytest.mark.benchmark
def test_real_time_jitter_p99_deterministic():
    """Verify that P99 control microkernel jitter is deterministic and strictly < 10 µs."""
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
    np.random.seed(42)
    stream = np.random.uniform(20.0, 60.0, 5000)

    # Warm-up (JIT / cache warm-up)
    for _ in range(100):
        _ = ctrl.fuzzy_logic(35.0)

    latencies_us = []
    for load in stream:
        t0 = time.perf_counter()
        _ = ctrl.fuzzy_logic(load)
        dt = (time.perf_counter() - t0) * 1e6
        latencies_us.append(dt)

    p50 = np.percentile(latencies_us, 50)
    p99 = np.percentile(latencies_us, 99)
    p99_9 = np.percentile(latencies_us, 99.9)

    # Hard real-time requirements:
    # Nyquist grid cycle: 20,000 µs (50Hz)
    # README claim: P99 jitter < 10 µs
    assert p99 < 10.0, f"P99 latency ({p99:.2f} µs) exceeded 10 µs threshold"
    assert p50 < 5.0, f"P50 latency ({p50:.2f} µs) exceeded 5 µs threshold"
    assert p99_9 < 50.0, f"P99.9 latency ({p99_9:.2f} µs) exceeded 50 µs threshold"


@pytest.mark.benchmark
def test_streaming_digital_twin_harness():
    """Verify StreamingDigitalTwin harness collects latencies and calculates P99."""
    twin = StreamingDigitalTwin()
    twin.run_jitter_test(n_ticks=2000)

    assert len(twin.latencies) == 2000
    p99 = np.percentile(twin.latencies, 99)
    assert p99 < 10.0


@pytest.mark.benchmark
@pytest.mark.parametrize("n_ops", [10_000, 100_000, 1_000_000])
def test_vectorized_throughput_capacity(n_ops):
    """Verify vectorized controller maintains > 10 Million ops/sec across batch scales."""
    ctrl = Controller(GLOBAL_TRAFO_LIMIT_MW)
    data = np.random.uniform(20.0, 60.0, n_ops)

    # Warm up
    _ = ctrl.fuzzy_logic(data[:1000])

    t0 = time.perf_counter()
    _ = ctrl.fuzzy_logic(data)
    elapsed = time.perf_counter() - t0

    throughput_mops = (n_ops / elapsed) / 1e6
    assert throughput_mops > 10.0, (
        f"Throughput {throughput_mops:.2f} MOps/s fell below 10 MOps/s requirement for N={n_ops}"
    )

