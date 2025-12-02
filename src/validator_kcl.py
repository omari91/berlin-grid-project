import numpy as np  # noqa: F401
import pandas as pd
from typing import Tuple


def validate_kcl(
    df: pd.DataFrame,
    gen_col: str,
    load_col: str,
    import_col: str,
) -> Tuple[float, pd.Series]:
    net_load = df[load_col] - df[gen_col]
    error = (df[import_col] - net_load).abs()
    mean_error = float(error.mean())
    return mean_error, error
