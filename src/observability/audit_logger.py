"""Regulatory Audit Logger for Berlin Grid Digital Twin.

Provides cryptographically tamper-evident structured logs compliant with:
- VDE-AR-N 4110 Technical Requirements for the Connection and Operation of Customer Installations
- §14a EnWG (German Energy Industry Act) congestion management auditing
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Optional


@dataclass
class AuditRecord:
    timestamp: str
    event_type: str
    action: str
    reason: str
    load_mw: float
    curtailed_mw: float
    voltage_pu: float
    trafo_loading_pct: float
    solver_confidence: float
    cryptographic_signature: str = ""

    def compute_signature(self, secret_salt: str = "berlin-grid-audit-salt") -> str:
        """Compute SHA-256 HMAC/hash of record fields for integrity verification."""
        canonical_str = (
            f"{self.timestamp}|{self.event_type}|{self.action}|{self.reason}|"
            f"{self.load_mw:.4f}|{self.curtailed_mw:.4f}|{self.voltage_pu:.4f}|"
            f"{self.trafo_loading_pct:.2f}|{self.solver_confidence:.3f}|{secret_salt}"
        )
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class AuditLogger:
    """Manages tamper-evident regulatory control event auditing."""

    def __init__(self, log_filepath: Optional[str] = None, max_in_memory: int = 1000) -> None:
        self.log_filepath = log_filepath
        self.max_in_memory = max_in_memory
        self.records: list[AuditRecord] = []

        if self.log_filepath:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_filepath)), exist_ok=True)

    def log_control_action(
        self,
        event_type: str,
        action: str,
        reason: str,
        load_mw: float,
        curtailed_mw: float,
        voltage_pu: float,
        trafo_loading_pct: float,
        solver_confidence: float = 1.0,
    ) -> AuditRecord:
        """Log a control action with timestamp and SHA-256 signature."""
        now_iso = datetime.now(timezone.utc).isoformat()
        record = AuditRecord(
            timestamp=now_iso,
            event_type=event_type,
            action=action,
            reason=reason,
            load_mw=float(load_mw),
            curtailed_mw=float(curtailed_mw),
            voltage_pu=float(voltage_pu),
            trafo_loading_pct=float(trafo_loading_pct),
            solver_confidence=float(solver_confidence),
        )
        record.cryptographic_signature = record.compute_signature()

        self.records.append(record)
        if len(self.records) > self.max_in_memory:
            self.records.pop(0)

        if self.log_filepath:
            with open(self.log_filepath, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

        return record

    @staticmethod
    def verify_record(record: AuditRecord, secret_salt: str = "berlin-grid-audit-salt") -> bool:
        """Verify the cryptographic signature of an audit record."""
        expected_sig = record.compute_signature(secret_salt=secret_salt)
        return expected_sig == record.cryptographic_signature

    def get_recent_records(self, n: int = 10) -> list[AuditRecord]:
        """Retrieve the N most recent audit records."""
        return self.records[-n:]

