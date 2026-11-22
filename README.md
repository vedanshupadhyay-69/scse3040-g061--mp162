# SCSE3040 MLOps Mini Project — Group G061 (MP162)

- **Group ID:** G061
- **Problem ID:** MP162 — OPD Waiting Time (Jaipur)
- **Client:** Jaipur District Hospital
- **Faculty:** Dr. Avantika Saklani (Batch 90)

## Group Members
- `S24CSEU2648` — VedanshUpadhyay
- `S24CSEU1566` — AnantSiwach
- `S24CSEU0880` — ChaitanyaSharma
- `S24CSEU1394` — AbhimanyuNarang

## Pipeline Overview
This repository implements an end-to-end MLOps pipeline for predicting hospital OPD waiting time (`wait_min`, regression evaluated by MAE):
1. **Data Versioning & Audit:** DVC tracking for raw data (`train.csv`, `later.csv`) and automated 7-issue audit (`src/mp162/data_audit.py`).
2. **Data Cleaning & Training:** Reusable cleaning (`src/mp162/cleaning.py`) and MLflow experiment tracking + Model Registry (`src/mp162/train.py`, `sqlite:///mlflow.db`).
3. **Prediction Service & Deployment:** FastAPI service (`app/main.py`) with Pydantic validation rules, structured JSON stdout logging, `Dockerfile`, GitHub Actions CI, and Kubernetes manifests (`k8s/`).
4. **Monitoring, Retraining & Holdout Inference:** 10-bin PSI drift detection, wing-level MAE analysis, 70/30 v2 retraining with 12% promotion gate (`src/mp162/monitor_and_retrain.py`), and holdout scoring (`src/mp162/predict_holdout.py`).

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.mp162.run_all
pytest -v
```
