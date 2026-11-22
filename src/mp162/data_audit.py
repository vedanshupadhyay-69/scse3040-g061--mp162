"""Programmatic audit of the 7 possible data problems in train.csv (MP162)."""

import yaml
import numpy as np
import pandas as pd


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_data(params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    train = pd.read_csv(params["data"]["raw_train"])
    later = pd.read_csv(params["data"]["raw_later"])
    holdout = pd.read_csv(params["data"]["holdout_features"])
    target = params["features"]["target"]
    num_cols = params["features"]["numeric"]

    # 1. Leaking column: present in train (not target) but missing from holdout
    extra_cols = sorted(list(set(train.columns) - set(holdout.columns) - {target}))
    leaking_present = len(extra_cols) > 0
    leaking_col = extra_cols[0] if leaking_present else None

    # 2. Unit mixup: check if any wing has mean >= 8x or <= 1/8x another wing
    unit_present = False
    unit_col = None
    unit_group = None
    unit_factor = None
    for col in num_cols:
        means = train.groupby("wing")[col].mean()
        ratio = means.max() / max(abs(means.min()), 1e-9)
        if ratio >= 8.0:
            unit_present = True
            unit_col = col
            unit_group = str(means.idxmax())
            unit_factor = float(round(ratio))
            break

    # 3. Sentinel value: impossible negative value in count/measurement columns
    sentinel_present = False
    sentinel_col = None
    sentinel_val = None
    sentinel_count = None
    for col in num_cols:
        neg_mask = train[col] < 0
        if neg_mask.sum() >= 5:
            val = train.loc[neg_mask, col].mode().iloc[0]
            sentinel_present = True
            sentinel_col = col
            sentinel_val = int(val) if float(val).is_integer() else float(val)
            sentinel_count = int((train[col] == val).sum())
            break

    # 4. Duplicate rows: number of extra copies in raw train.csv
    dup_count = int(train.duplicated().sum())
    dup_present = dup_count > 0

    # 5. Stuck sensor: long consecutive run (>= 15 rows) of identical value by date
    stuck_present = False
    stuck_col = None
    stuck_start = None
    stuck_end = None
    t_sorted = train.sort_values("date").reset_index(drop=True)
    for col in num_cols:
        s = t_sorted[col]
        runs = (s != s.shift()).cumsum()
        grp = t_sorted.groupby(runs).agg(
            count=(col, "size"),
            start=("date", "min"),
            end=("date", "max"),
        )
        top = grp.sort_values("count", ascending=False).iloc[0]
        if int(top["count"]) >= 15:
            stuck_present = True
            stuck_col = col
            stuck_start = str(top["start"])
            stuck_end = str(top["end"])
            break

    # 6. Target typos: 10x inflated target values separated by a large gap
    median_y = train[target].median()
    typo_mask = train[target] > (3.5 * median_y)
    typo_count = int(typo_mask.sum())
    typo_present = typo_count > 0

    # 7. Spelling variants: text column having more unique values in train than later
    spelling_present = False
    spelling_col = None
    true_levels = None
    for col in params["features"]["categorical"]:
        train_unique = set(train[col].dropna().unique())
        later_unique = set(later[col].dropna().unique())
        if len(train_unique) > len(later_unique):
            spelling_present = True
            spelling_col = col
            true_levels = int(len(later_unique))
            break

    issues = {
        "leaking_column": {
            "present": leaking_present,
            "column": leaking_col,
        },
        "unit_mixup": {
            "present": unit_present,
            "column": unit_col,
            "group_value": unit_group,
            "factor": unit_factor,
        },
        "sentinel_value": {
            "present": sentinel_present,
            "column": sentinel_col,
            "value": sentinel_val,
            "count": sentinel_count,
        },
        "duplicate_rows": {
            "present": dup_present,
            "count": dup_count if dup_present else None,
        },
        "stuck_sensor": {
            "present": stuck_present,
            "column": stuck_col,
            "start_date": stuck_start,
            "end_date": stuck_end,
        },
        "target_typos": {
            "present": typo_present,
            "count": typo_count if typo_present else None,
        },
        "spelling_variants": {
            "present": spelling_present,
            "column": spelling_col,
            "true_levels": true_levels,
        },
    }
    return issues


if __name__ == "__main__":
    results = audit_data()
    print(yaml.safe_dump({"data_issues": results}, sort_keys=False))
