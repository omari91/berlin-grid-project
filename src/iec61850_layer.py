"""
IEC 61850 Substation Integration Layer
Demonstrates how the digital twin interfaces with real substation equipment

Author: Clifford Ondieki
Reference: IEC 61850-7-4 (Logical Nodes), IEC 61850-8-1 (MMS), IEC 61850-9-2 (Sampled Values)
"""

import numpy as np


class LogicalNode:
    """Base class for IEC 61850 Logical Nodes (function blocks)"""
    def __init__(self, name):
        self.name = name
        self.data_objects = {}
    
    def update(self, **kwargs):
        self.data_objects.update(kwargs)
    
    def get(self, attribute):
        return self.data_objects.get(attribute, None)


class MMXU(LogicalNode):
    """
    MMXU = Measurement (function block for metering)
    Used for: Real-time voltage, current, power measurements
    Communication: MMS (Manufacturing Message Specification)
    """
    def __init__(self, name="MMXU1"):
        super().__init__(name)
        # IEC 61850 Data Objects
        self.data_objects = {
            'TotW': 0.0,      # Total Active Power (MW)
            'TotVAr': 0.0,    # Total Reactive Power (MVAr)
            'PPV.phsA': 1.0,  # Phase A Voltage (p.u.)
            'A.phsA': 0.0,    # Phase A Current (A)
        }
    
    def read_from_scada(self, active_mw, reactive_mvar, voltage_pu):
        """MMS Client-Server: Request measurements from bay controller"""
        self.update(
            TotW=active_mw,
            TotVAr=reactive_mvar,
            **{'PPV.phsA': voltage_pu}
        )
        return self.data_objects


class XCBR(LogicalNode):
    """
    XCBR = Circuit Breaker (function block for switching device)
    Controls: Breaker position (open/close)
    Communication: GOOSE (Generic Object Oriented Substation Event)
    """
    def __init__(self, name="XCBR1"):
        super().__init__(name)
        self.data_objects = {
            'Pos.stVal': True,    # Position: True=Closed, False=Open
            'BlkOpn': False,       # Block Opening command
        }
    
    def send_trip_command(self):
        """GOOSE: Fast peer-to-peer signal (<4ms latency)"""
        self.update(**{'Pos.stVal': False})
        print(f"🔴 [{self.name}] GOOSE Trip Signal Sent - Breaker OPEN")
    
    def send_close_command(self):
        self.update(**{'Pos.stVal': True})
        print(f"🟢 [{self.name}] GOOSE Close Signal - Breaker CLOSED")


class PDIS(LogicalNode):
    """
    PDIS = Distance Protection (function block for protection relay)
    Monitors: Voltage violations, triggers protection schemes
    Communication: GOOSE for trip signals
    """
    def __init__(self, name="PDIS1"):
        super().__init__(name)
        self.data_objects = {
            'Str': False,       # Protection Started
            'Op': False,        # Protection Operated (trip)
        }
    
    def evaluate_protection(self, voltage_pu, threshold=0.90):
        """Protection logic: Trip if voltage drops below threshold"""
        if voltage_pu < threshold:
            self.update(Str=True, Op=True)
            return True  # Trigger GOOSE trip
        else:
            self.update(Str=False, Op=False)
            return False


class IEC61850Station:
    """
    Complete IEC 61850 Station Bus
    Integrates: Measurement (MMXU) + Protection (PDIS) + Switching (XCBR)
    
    Protocols:
    - MMS: Client-server for SCADA monitoring (100-1000ms cycle)
    - GOOSE: Peer-to-peer multicast for protection (<4ms)
    - Sampled Values: Process bus measurements (not implemented here)
    """
    def __init__(self):
        self.mmxu = MMXU("MMXU_Transformer")
        self.xcbr = XCBR("XCBR_Feeder")
        self.pdis = PDIS("PDIS_Undervoltage")
        self.protection_log = []
    
    def process_scada_cycle(self, load_mw, voltage_pu):
        """
        MMS Protocol: Client-server communication for monitoring
        Typical cycle time: 100-1000ms
        
        Args:
            load_mw: Active power load (MW)
            voltage_pu: Bus voltage (per-unit)
            
        Returns:
            measurements: Dictionary of MMXU data objects
            trip_required: Boolean indicating if protection operated
        """
        # 1. Read measurements via MMS
        measurements = self.mmxu.read_from_scada(
            active_mw=load_mw,
            reactive_mvar=load_mw * 0.3,
            voltage_pu=voltage_pu
        )
        
        # 2. Evaluate protection
        trip_required = self.pdis.evaluate_protection(voltage_pu)
        
        # 3. If protection operated, send GOOSE trip (<4ms)
        if trip_required and self.xcbr.data_objects['Pos.stVal']:
            self.xcbr.send_trip_command()
            self.protection_log.append({
                'voltage': voltage_pu,
                'load': load_mw,
                'action': 'TRIP'
            })
        
        return measurements, trip_required
