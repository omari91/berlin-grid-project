import pandas as pd

from src.validator_kcl import validate_kcl


def test_kcl_computation():
    df = pd.DataFrame(
        {
            "Gen_MS_kW": [1000, 2000, 1500],
            "Total_Load_MS_kW": [5000, 5200, 4800],
            "Grid_Import_MS_kW": [4000, 3200, 3300],
        }
    )
    mean_err, err_series = validate_kcl(
        df,
        gen_col="Gen_MS_kW",
        load_col="Total_Load_MS_kW",
        import_col="Grid_Import_MS_kW",
    )
    assert mean_err >= 0
    assert len(err_series) == 3
