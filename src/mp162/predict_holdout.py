"""Generate final holdout predictions using the @champion model."""

import os
import joblib
import pandas as pd

from src.mp162.cleaning import clean_dataframe, load_params


def predict_holdout(params_path: str = "params.yaml") -> str:
    params = load_params(params_path)
    holdout_df = pd.read_csv(params["data"]["holdout_features"])
    holdout_clean = clean_dataframe(holdout_df, is_train=False)

    feature_cols = params["features"]["numeric"] + params["features"]["categorical"]
    champion_model = joblib.load(params["model"]["output_path"])
    preds = champion_model.predict(holdout_clean[feature_cols])

    out_df = pd.DataFrame(
        {
            "record_id": holdout_df["record_id"],
            "prediction": [round(float(p), 2) for p in preds],
        }
    )

    out_path = params["data"]["predictions_out"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} holdout predictions to {out_path}")
    return out_path


if __name__ == "__main__":
    predict_holdout()
