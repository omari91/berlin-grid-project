"""Unit tests for Distributed Substation Coordination & Consensus (Stage B)."""
import time
import pytest
import numpy as np
from src.distribution import (
    DistributedCoordinator,
    SubstationState,
    NodeRole,
    CoordinationMessage,
)


def test_substation_state_initialization():
    """Verify SubstationState dataclass fields and default clock."""
    state = SubstationState(
        site_id="Berlin-01",
        load_mw=30.0,
        voltage_pu=0.98,
        curtailment_mw=2.5,
        capacity_mw=45.0,
    )
    assert state.site_id == "Berlin-01"
    assert state.load_mw == 30.0
    assert state.lamport_clock == 0
    assert state.timestamp > 0.0


def test_coordinator_lamport_clock_advancement():
    """Verify Lamport clock increments monotonically on local events and remote syncs."""
    coord = DistributedCoordinator("Berlin-01", ["Berlin-01", "Berlin-02"])
    assert coord.lamport_clock == 0

    c1 = coord.tick_lamport_clock()
    assert c1 == 1

    # Incoming message with higher clock advances local clock beyond sender
    c2 = coord.tick_lamport_clock(incoming_clock=10)
    assert c2 == 11
    assert coord.local_state.lamport_clock == 11


def test_coordinator_state_update():
    """Verify local state update records telemetry and timestamps."""
    coord = DistributedCoordinator("Berlin-01", ["Berlin-01", "Berlin-02"])
    state = coord.update_local_state(load_mw=42.0, voltage_pu=0.94, curtailment_mw=4.2)

    assert state.load_mw == 42.0
    assert state.voltage_pu == 0.94
    assert state.curtailment_mw == 4.2
    assert state.lamport_clock >= 1


def test_coordinator_heartbeat_and_lww_conflict_resolution():
    """Verify Last-Write-Wins (LWW) conflict resolution updates state correctly."""
    coord = DistributedCoordinator("Berlin-01", ["Berlin-01", "Berlin-02"])

    # Peer Berlin-02 sends state with clock 2
    state_v1 = SubstationState(
        site_id="Berlin-02",
        load_mw=25.0,
        voltage_pu=0.99,
        curtailment_mw=0.0,
        capacity_mw=45.0,
        lamport_clock=2,
        timestamp=100.0,
    )
    msg_v1 = CoordinationMessage(sender_id="Berlin-02", msg_type="HEARTBEAT", state=state_v1)
    coord.receive_heartbeat(msg_v1)

    assert "Berlin-02" in coord.peer_states
    assert coord.peer_states["Berlin-02"].load_mw == 25.0

    # Peer Berlin-02 sends updated state with clock 5 (higher)
    state_v2 = SubstationState(
        site_id="Berlin-02",
        load_mw=38.0,
        voltage_pu=0.96,
        curtailment_mw=3.0,
        capacity_mw=45.0,
        lamport_clock=5,
        timestamp=105.0,
    )
    msg_v2 = CoordinationMessage(sender_id="Berlin-02", msg_type="HEARTBEAT", state=state_v2)
    coord.receive_heartbeat(msg_v2)
    assert coord.peer_states["Berlin-02"].load_mw == 38.0

    # Outdated message with clock 3 must be ignored (LWW)
    state_outdated = SubstationState(
        site_id="Berlin-02",
        load_mw=10.0,
        voltage_pu=1.00,
        curtailment_mw=0.0,
        capacity_mw=45.0,
        lamport_clock=3,
        timestamp=90.0,
    )
    msg_outdated = CoordinationMessage(sender_id="Berlin-02", msg_type="HEARTBEAT", state=state_outdated)
    coord.receive_heartbeat(msg_outdated)
    assert coord.peer_states["Berlin-02"].load_mw == 38.0


def test_coordinator_distributed_redispatch():
    """Verify PRIMARY calculates proportional congestion relief across active peers."""
    nodes = ["Berlin-01", "Berlin-02", "Berlin-03"]
    b1 = DistributedCoordinator("Berlin-01", nodes, capacity_mw=60.0)
    b2 = DistributedCoordinator("Berlin-02", nodes, capacity_mw=40.0)

    # b1 is PRIMARY
    hb2 = b2.generate_heartbeat()
    b1.receive_heartbeat(hb2)

    # Non-primary returns None
    assert b2.calculate_distributed_redispatch(10.0) is None

    # Primary allocates 10 MW relief proportionally (60 MW vs 40 MW -> 6 MW vs 4 MW)
    alloc = b1.calculate_distributed_redispatch(10.0)
    assert alloc is not None
    assert np.isclose(alloc["Berlin-01"], 6.0, atol=0.1)
    assert np.isclose(alloc["Berlin-02"], 4.0, atol=0.1)
