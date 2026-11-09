"""Distributed Multi-Substation Coordination & Consensus Module.

Implements Stage B: Multi-Substation Cascaded Control
- Bounded drift synchronization
- Quorum-based decision-making (majority partition continues; minority enters local-first safety)
- Heartbeat & Partition detection (500ms heartbeat, 1.5s detection timeout)
- Anti-flapping hysteresis (5.0s hold)
- Lamport clock + timestamp Last-Write-Wins (LWW) conflict resolution
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional


class NodeRole(str, Enum):
    PRIMARY = "PRIMARY"
    STANDBY = "STANDBY"
    ISOLATED_LOCAL = "ISOLATED_LOCAL"


@dataclass
class SubstationState:
    site_id: str
    load_mw: float
    voltage_pu: float
    curtailment_mw: float
    capacity_mw: float
    lamport_clock: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoordinationMessage:
    sender_id: str
    msg_type: str  # "HEARTBEAT", "REDISPATCH", "FAILOVER_CLAIM"
    state: SubstationState
    target_primary: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)


class DistributedCoordinator:
    """Multi-substation coordinator maintaining partition tolerance and bounded drift."""

    def __init__(
        self,
        site_id: str,
        cluster_nodes: list[str],
        capacity_mw: float = 45.0,
        heartbeat_interval_s: float = 0.5,
        detection_timeout_s: float = 1.5,
        anti_flapping_window_s: float = 5.0,
    ) -> None:
        self.site_id: str = site_id
        self.cluster_nodes: list[str] = sorted(list(set(cluster_nodes)))
        self.capacity_mw: float = capacity_mw
        self.heartbeat_interval: float = heartbeat_interval_s
        self.detection_timeout: float = detection_timeout_s
        self.anti_flapping_window: float = anti_flapping_window_s

        # Quorum requirement: strictly > N/2
        self.quorum_size: int = (len(self.cluster_nodes) // 2) + 1

        self.lamport_clock: int = 0
        self.local_state: SubstationState = SubstationState(
            site_id=self.site_id,
            load_mw=0.0,
            voltage_pu=1.0,
            curtailment_mw=0.0,
            capacity_mw=self.capacity_mw,
            lamport_clock=self.lamport_clock,
        )

        # Peer tracking
        self.peer_states: dict[str, SubstationState] = {}
        self.peer_last_seen: dict[str, float] = {}

        self.role: NodeRole = NodeRole.STANDBY
        self.current_primary: Optional[str] = self.cluster_nodes[0] if self.cluster_nodes else None
        self.last_election_time: float = 0.0

        # Determine initial role
        self.evaluate_quorum_and_role()

    def tick_lamport_clock(self, incoming_clock: int = 0) -> int:
        """Advance logical Lamport clock."""
        self.lamport_clock = max(self.lamport_clock, incoming_clock) + 1
        self.local_state.lamport_clock = self.lamport_clock
        return self.lamport_clock

    def update_local_state(
        self, load_mw: float, voltage_pu: float, curtailment_mw: float
    ) -> SubstationState:
        """Update local digital twin telemetry."""
        self.tick_lamport_clock()
        self.local_state.load_mw = float(load_mw)
        self.local_state.voltage_pu = float(voltage_pu)
        self.local_state.curtailment_mw = float(curtailment_mw)
        self.local_state.timestamp = time.time()
        return self.local_state

    def generate_heartbeat(self) -> CoordinationMessage:
        """Generate outgoing heartbeat message."""
        self.tick_lamport_clock()
        return CoordinationMessage(
            sender_id=self.site_id,
            msg_type="HEARTBEAT",
            state=self.local_state,
            target_primary=self.current_primary,
        )

    def receive_heartbeat(self, msg: CoordinationMessage) -> None:
        """Process incoming heartbeat from a peer node with Last-Write-Wins resolution."""
        sender = msg.sender_id
        if sender not in self.cluster_nodes or sender == self.site_id:
            return

        now = time.time()
        self.peer_last_seen[sender] = now
        incoming_state = msg.state
        self.tick_lamport_clock(incoming_state.lamport_clock)

        existing_state = self.peer_states.get(sender)
        if existing_state is None:
            self.peer_states[sender] = incoming_state
        else:
            # LWW conflict resolution: Lamport clock first, then wall-clock timestamp
            if (incoming_state.lamport_clock > existing_state.lamport_clock) or (
                incoming_state.lamport_clock == existing_state.lamport_clock
                and incoming_state.timestamp >= existing_state.timestamp
            ):
                self.peer_states[sender] = incoming_state

        self.evaluate_quorum_and_role()

    def get_active_peer_count(self) -> int:
        """Count reachable peers within detection timeout (1.5s)."""
        now = time.time()
        active_count = 0
        for node in self.cluster_nodes:
            if node == self.site_id:
                active_count += 1
            elif (now - self.peer_last_seen.get(node, 0.0)) <= self.detection_timeout:
                active_count += 1
        return active_count

    def evaluate_quorum_and_role(self) -> NodeRole:
        """Evaluate network partition status and leader election."""
        now = time.time()
        active_nodes = self.get_active_peer_count()

        # Check Quorum
        if active_nodes < self.quorum_size:
            # Network partition minority -> Fall back to local autonomous safety
            self.role = NodeRole.ISOLATED_LOCAL
            self.current_primary = None
            return self.role

        # Quorum satisfied (majority active)
        active_candidates = [
            node for node in self.cluster_nodes
            if node == self.site_id or (now - self.peer_last_seen.get(node, 0.0)) <= self.detection_timeout
        ]
        active_candidates.sort()

        target_leader = active_candidates[0]

        # Apply anti-flapping check if leader is changing
        if target_leader != self.current_primary:
            if (now - self.last_election_time) >= self.anti_flapping_window:
                self.current_primary = target_leader
                self.last_election_time = now

        if self.current_primary == self.site_id:
            self.role = NodeRole.PRIMARY
        else:
            self.role = NodeRole.STANDBY

        return self.role

    def calculate_distributed_redispatch(
        self, total_congestion_relief_mw: float
    ) -> Optional[dict[str, float]]:
        """Compute proportional redispatch allocation across active quorum members."""
        if self.role != NodeRole.PRIMARY:
            return None

        active_peers = [
            node for node in self.cluster_nodes
            if node == self.site_id or (time.time() - self.peer_last_seen.get(node, 0.0)) <= self.detection_timeout
        ]

        total_capacity = sum(
            self.local_state.capacity_mw if n == self.site_id else self.peer_states[n].capacity_mw
            for n in active_peers if n == self.site_id or n in self.peer_states
        )

        if total_capacity <= 0:
            return None

        allocations: dict[str, float] = {}
        for n in active_peers:
            node_cap = self.local_state.capacity_mw if n == self.site_id else self.peer_states[n].capacity_mw
            # Proportional curtailment allocation
            share = (node_cap / total_capacity) * total_congestion_relief_mw
            allocations[n] = round(share, 3)

        return allocations

