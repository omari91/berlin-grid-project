"""IEC 61850 Substation Integration Protocol Test Suite (MMS, GOOSE, Logical Nodes)."""
import pytest
from src.iec61850_layer import LogicalNode, MMXU, XCBR, PDIS, IEC61850Station


def test_logical_node_base():
    """Verify LogicalNode base class initialization, update, and get methods."""
    node = LogicalNode("LN_Test")
    assert node.name == "LN_Test"
    assert node.data_objects == {}

    node.update(custom_attr="valid", num_attr=42)
    assert node.get("custom_attr") == "valid"
    assert node.get("num_attr") == 42
    assert node.get("non_existent") is None


def test_mmxu_scada_measurements():
    """Verify MMXU Logical Node records active power, reactive power, and phase voltages."""
    mmxu = MMXU("MMXU_Feeder1")
    assert mmxu.data_objects["TotW"] == 0.0
    assert mmxu.data_objects["PPV.phsA"] == 1.0

    meas = mmxu.read_from_scada(active_mw=32.5, reactive_mvar=9.75, voltage_pu=0.985)
    assert meas["TotW"] == 32.5
    assert meas["TotVAr"] == 9.75
    assert meas["PPV.phsA"] == 0.985
    assert mmxu.get("TotW") == 32.5


def test_xcbr_circuit_breaker_switching():
    """Verify XCBR Circuit Breaker opens on trip and closes on close command."""
    xcbr = XCBR("XCBR_Main")
    # Initial state: Breaker CLOSED (Pos.stVal = True)
    assert xcbr.data_objects["Pos.stVal"] is True

    # Send GOOSE trip command
    xcbr.send_trip_command()
    assert xcbr.data_objects["Pos.stVal"] is False

    # Send close command
    xcbr.send_close_command()
    assert xcbr.data_objects["Pos.stVal"] is True


def test_pdis_undervoltage_protection_threshold():
    """Verify PDIS Distance/Undervoltage protection triggers trip at threshold < 0.90 pu."""
    pdis = PDIS("PDIS_Substation")
    assert pdis.data_objects["Str"] is False
    assert pdis.data_objects["Op"] is False

    # Safe voltage (0.95 pu) -> No trip
    assert pdis.evaluate_protection(0.95, threshold=0.90) is False
    assert pdis.data_objects["Op"] is False

    # Exact threshold boundary (0.90 pu) -> No trip
    assert pdis.evaluate_protection(0.90, threshold=0.90) is False
    assert pdis.data_objects["Op"] is False

    # Violation (0.89 pu) -> Protection operates
    assert pdis.evaluate_protection(0.89, threshold=0.90) is True
    assert pdis.data_objects["Op"] is True
    assert pdis.data_objects["Str"] is True

    # Voltage recovery (0.98 pu) -> Clears operation
    assert pdis.evaluate_protection(0.98, threshold=0.90) is False
    assert pdis.data_objects["Op"] is False
    assert pdis.data_objects["Str"] is False


def test_iec61850_station_normal_scada_cycle():
    """Verify normal SCADA cycle reads MMS telemetry without tripping breaker."""
    station = IEC61850Station()
    measurements, tripped = station.process_scada_cycle(load_mw=25.0, voltage_pu=0.97)

    assert tripped is False
    assert station.xcbr.data_objects["Pos.stVal"] is True
    assert measurements["TotW"] == 25.0
    assert measurements["TotVAr"] == 7.5  # 0.3 * load_mw
    assert measurements["PPV.phsA"] == 0.97
    assert len(station.protection_log) == 0


def test_iec61850_station_protection_trip_cycle():
    """Verify undervoltage condition triggers GOOSE trip and breaker state change."""
    station = IEC61850Station()
    # Breaker initially closed
    assert station.xcbr.data_objects["Pos.stVal"] is True

    measurements, tripped = station.process_scada_cycle(load_mw=50.0, voltage_pu=0.87)

    assert tripped is True
    # Breaker must now be OPEN
    assert station.xcbr.data_objects["Pos.stVal"] is False
    assert len(station.protection_log) == 1
    log_entry = station.protection_log[0]
    assert log_entry["voltage"] == 0.87
    assert log_entry["load"] == 50.0
    assert log_entry["action"] == "TRIP"


def test_iec61850_subsequent_cycles_when_open():
    """Verify breaker does not duplicate trip commands if already open."""
    station = IEC61850Station()
    # Trip breaker
    station.process_scada_cycle(load_mw=55.0, voltage_pu=0.85)
    assert len(station.protection_log) == 1

    # Second cycle under depression: breaker is already open
    station.process_scada_cycle(load_mw=55.0, voltage_pu=0.84)
    # Protection operated, but no second trip command needed
    assert len(station.protection_log) == 1

