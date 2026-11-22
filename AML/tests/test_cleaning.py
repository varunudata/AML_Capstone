"""
Unit tests for Stage 2 Data Cleaning Rules.
Verifies duplicate removal, impossible value NaN conversion, and scope segregation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    load_config,
    CleaningLog,
    step1_drop_duplicates,
    step2_flag_impossible,
    step6_waterbody_scope,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_test_df():
    """Generates synthetic test DataFrame covering edge cases and violation rules."""
    return pd.DataFrame({
        "Country": ["England", "England", "USA", "Ireland", "England"],
        "Area": ["SiteA", "SiteA", "SiteB", "SiteC", "SiteD"],
        "Waterbody Type": ["River", "River", "Effluent", "Sewage", "Lake"],
        "Date": pd.to_datetime(["2005-01-01", "2005-01-01", "2006-05-10", "2010-08-15", "2015-03-20"]),
        "Ammonia (mg/l)": [0.05, 0.05, -1.0, 0.2, 5.0],  # Row 2 is negative (impossible)
        "Biochemical Oxygen Demand (mg/l)": [2.5, 2.5, 3.0, -0.5, 4.0],  # Row 3 is negative
        "Dissolved Oxygen (mg/l)": [9.5, 9.5, 0.0, 8.0, 10.5],  # Row 2 is 0.0 (impossible)
        "Orthophosphate (mg/l)": [0.02, 0.02, 0.1, 0.05, -0.01],  # Row 4 is negative
        "pH (ph units)": [7.5, 7.5, 15.2, 6.8, -0.5],  # Row 2 > 14, Row 4 < 0
        "Temperature (cel)": [12.0, 12.0, 48.0, 15.0, -5.0],  # Row 2 > 40, Row 4 < -2
        "Nitrogen (mg/l)": [4.0, 4.0, 5.0, 3.0, 2.0],
        "Nitrate (mg/l)": [3.5, 3.5, 4.5, 2.5, 1.8],
        "CCME_Values": [95.0, 95.0, 50.0, 40.0, 85.0],
        "CCME_WQI": ["Excellent", "Excellent", "Marginal", "Poor", "Good"],
    })


def test_step1_duplicate_removal(sample_test_df):
    """Assertion: Exactly 1 duplicate row is detected and dropped."""
    log = CleaningLog(len(sample_test_df))
    cleaned = step1_drop_duplicates(sample_test_df, log)
    assert len(cleaned) == 4, f"Expected 4 rows after duplicate removal, got {len(cleaned)}"
    assert log.entries[0]["rows_affected"] == 1


def test_step2_impossible_values_to_nan(sample_test_df):
    """Assertion: Physically impossible values are converted to NaN."""
    config = load_config()
    log = CleaningLog(len(sample_test_df))
    cleaned = step2_flag_impossible(sample_test_df, config, log)

    # Check Ammonia negative -> NaN
    assert np.isnan(cleaned.loc[2, "Ammonia (mg/l)"])
    # Check BOD negative -> NaN
    assert np.isnan(cleaned.loc[3, "Biochemical Oxygen Demand (mg/l)"])
    # Check DO == 0 -> NaN
    assert np.isnan(cleaned.loc[2, "Dissolved Oxygen (mg/l)"])
    # Check Orthophosphate negative -> NaN
    assert np.isnan(cleaned.loc[4, "Orthophosphate (mg/l)"])
    # Check pH > 14 -> NaN and pH < 0 -> NaN
    assert np.isnan(cleaned.loc[2, "pH (ph units)"])
    assert np.isnan(cleaned.loc[4, "pH (ph units)"])
    # Check Temp > 40 -> NaN and Temp < -2 -> NaN
    assert np.isnan(cleaned.loc[2, "Temperature (cel)"])
    assert np.isnan(cleaned.loc[4, "Temperature (cel)"])


def test_step6_waterbody_scope_exclusion(sample_test_df):
    """Assertion: Effluent and Sewage waterbody types are segregated from ambient data."""
    config = load_config()
    log = CleaningLog(len(sample_test_df))
    ambient, excluded = step6_waterbody_scope(sample_test_df, config, log)

    excluded_types = set(excluded["Waterbody Type"])
    ambient_types = set(ambient["Waterbody Type"])

    assert "Effluent" in excluded_types
    assert "Sewage" in excluded_types
    assert "Effluent" not in ambient_types
    assert "Sewage" not in ambient_types
    assert len(ambient) == 3
    assert len(excluded) == 2

