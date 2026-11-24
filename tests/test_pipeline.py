"""Unit tests for MP162 cleaning and params loading (synthetic data only, no raw CSVs)."""

import numpy as np
import pandas as pd
import yaml

from src.mp162.cleaning import clean_dataframe


def test_params_yaml_loads_cleanly():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    assert params["mlflow"]["tracking_uri"] == "sqlite:///mlflow.db"
    assert len(params["features"]["numeric"]) == 6
    assert params["features"]["target"] == "wait_min"


def test_clean_dataframe_fixes_all_issues():
    df = pd.DataFrame(
        [
            {
                "record_id": "R1",
                "date": "2026-01-01",
                "patients_ahead": -1,
                "doctors_on_duty": 4,
                "arrival_hour": 10,
                "avg_consult_min": 12.0,
                "lab_tests_pending": 5,
                "distance_km": 10.5,
                "department": "Gen.",
                "wing": "Old Wing",
                "wait_min": 850,
            },
            {
                "record_id": "R1",
                "date": "2026-01-01",
                "patients_ahead": -1,
                "doctors_on_duty": 4,
                "arrival_hour": 10,
                "avg_consult_min": 12.0,
                "lab_tests_pending": 5,
                "distance_km": 10.5,
                "department": "Gen.",
                "wing": "Old Wing",
                "wait_min": 850,
            },
            {
                "record_id": "R2",
                "date": "2026-01-02",
                "patients_ahead": 20,
                "doctors_on_duty": 5,
                "arrival_hour": 11,
                "avg_consult_min": 10.0,
                "lab_tests_pending": 2,
                "distance_km": 4.0,
                "department": " Pae. ",
                "wing": "New Wing",
                "wait_min": 65,
            },
        ]
    )
    cleaned = clean_dataframe(df, is_train=True)
    # Duplicate dropped
    assert len(cleaned) == 2
    # Sentinel -1 replaced with NaN
    assert np.isnan(cleaned.loc[0, "patients_ahead"])
    # 10x target typo fixed (850 -> 85.0)
    assert cleaned.loc[0, "wait_min"] == 85.0
    # Spelling variants normalized
    assert list(cleaned["department"]) == ["General", "Paediatrics"]
