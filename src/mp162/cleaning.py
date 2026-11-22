"""Data cleaning utilities for MP162 OPD Waiting Time dataset."""

import numpy as np
import pandas as pd
import yaml


DEPARTMENT_MAP = {
    "general": "General",
    "gen.": "General",
    "gen": "General",
    "paediatrics": "Paediatrics",
    "pae.": "Paediatrics",
    "pae": "Paediatrics",
    "orthopaedics": "Orthopaedics",
    "ort.": "Orthopaedics",
    "ort": "Orthopaedics",
}


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_department(series: pd.Series) -> pd.Series:
    """Standardize whitespace, case, and abbreviations in department names."""
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(DEPARTMENT_MAP)
    return mapped.fillna(series.astype(str).str.strip().str.title())


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean raw MP162 DataFrame without modifying the source CSV on disk."""
    out = df.copy()

    # 1. Remove exact duplicate rows on training data
    if is_train:
        out = out.drop_duplicates().reset_index(drop=True)

    # 2. Replace sentinel value (-1) in patients_ahead with NaN
    if "patients_ahead" in out.columns:
        out["patients_ahead"] = pd.to_numeric(out["patients_ahead"], errors="coerce")
        out.loc[out["patients_ahead"] < 0, "patients_ahead"] = np.nan

    # 3. Normalize department spelling variants
    if "department" in out.columns:
        out["department"] = normalize_department(out["department"])

    # 4. Strip wing whitespace if any
    if "wing" in out.columns:
        out["wing"] = out["wing"].astype(str).str.strip()

    # 5. Fix 10x target typos in wait_min (only during training)
    if is_train and "wait_min" in out.columns:
        out["wait_min"] = pd.to_numeric(out["wait_min"], errors="coerce")
        typo_mask = out["wait_min"] > 300
        out.loc[typo_mask, "wait_min"] = out.loc[typo_mask, "wait_min"] / 10.0

    return out
