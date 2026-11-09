"""Unit tests for DataLayer data parsing and cleaning."""
import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from src.core import DataLayer
from main import DataLayer as MainDataLayer, FILES_MAP


def test_clean_german_float_valid_strings():
    """Verify German comma decimal and dot thousand separators are correctly parsed."""
    assert DataLayer.clean_german_float("1.234,56") == 1234.56
    assert DataLayer.clean_german_float("0,42") == 0.42
    assert DataLayer.clean_german_float("-1.000,50") == -1000.50
    assert DataLayer.clean_german_float("100") == 100.0


def test_clean_german_float_numeric_and_null():
    """Verify numeric floats, integers, and null/NA values are handled safely."""
    assert DataLayer.clean_german_float(42.5) == 42.5
    assert DataLayer.clean_german_float(100) == 100.0
    assert DataLayer.clean_german_float(None) == 0.0
    assert DataLayer.clean_german_float(np.nan) == 0.0
    assert DataLayer.clean_german_float(pd.NA) == 0.0


def test_clean_german_float_malformed():
    """Verify non-numeric strings safely return 0.0 without throwing exceptions."""
    assert DataLayer.clean_german_float("N/A") == 0.0
    assert DataLayer.clean_german_float("invalid_text") == 0.0
    assert DataLayer.clean_german_float("") == 0.0


def test_main_datalayer_synthetic_generation():
    """Verify synthetic German grid profile generation in main.py."""
    df = MainDataLayer.load_and_clean(FILES_MAP)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "Net_Load_MW" in df.columns
    assert df["Net_Load_MW"].notna().all()
    # Check that synthetic values oscillate around ~35 MW
    assert 10.0 <= df["Net_Load_MW"].mean() <= 60.0
    # Check 15-minute frequency index
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df.index) == 1000


def test_datalayer_load_berlin_data_mock():
    """Verify DataLayer.load_berlin_data parses multi-CSV files and converts kW to MW."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock CSV files
        dates = ["01.01.2024 00:00", "01.01.2024 00:15", "01.01.2024 00:30"]
        gen_data = "Datum;Erzeugung\n" + "\n".join([f"{d};1.000,5" for d in dates])
        load_data = "Datum;Verbrauch\n" + "\n".join([f"{d};2.500,0" for d in dates])
        import_data = "Datum;Import\n" + "\n".join([f"{d};1.499,5" for d in dates])

        with open(os.path.join(tmpdir, "MS_Erzeugung.csv"), "w") as f:
            f.write(gen_data)
        with open(os.path.join(tmpdir, "MS_Verbrauch.csv"), "w") as f:
            f.write(load_data)
        with open(os.path.join(tmpdir, "MS_Import.csv"), "w") as f:
            f.write(import_data)

        df = DataLayer.load_berlin_data(data_dir=tmpdir)
        assert len(df) == 3
        assert "Gen_MS_MW" in df.columns
        assert "Total_Load_MS_MW" in df.columns
        assert "Grid_Import_MS_MW" in df.columns
        assert np.isclose(df["Gen_MS_MW"].iloc[0], 1.0005)
        assert np.isclose(df["Total_Load_MS_MW"].iloc[0], 2.5)
        assert np.isclose(df["Grid_Import_MS_MW"].iloc[0], 1.4995)

