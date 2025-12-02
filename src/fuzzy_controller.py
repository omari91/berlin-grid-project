import numpy as np


def sigmoid_dimming(
    load_mw: float,
    limit_mw: float,
    k: float = 15.0,
    sref: float = 0.95,
) -> float:
    stress = load_mw / limit_mw if limit_mw > 0 else 1.0
    alpha = 1.0 / (1.0 + np.exp(-k * (stress - sref)))
    final_load = min(load_mw * (1 - alpha * 0.3), limit_mw * 1.02)
    return float(final_load)
