# Pakistan House Price ML Pipeline

End-to-end reproducible workflow for training and serving a house price regression model on the Pakistan House Price dataset using DVC + Git and a Flask API.

## Project Structure
- `data/zameen-updated.csv` – raw dataset (tracked by DVC)
- `params.yaml` – configurable training params
- `src/train.py` – training pipeline that saves `models/model.pkl`, `metrics.json`, and `artifacts/columns.json`
- `dvc.yaml` / `dvc.lock` – DVC pipeline definition and lockfile
- `app.py` – Flask service exposing `/predict`

## Quickstart
1) Install deps: `python -m pip install -r requirements.txt`
2) (First time) Initialize VCS + DVC:
   ```bash
   git init
   dvc init
   dvc remote add -d localremote ./dvc_store
   ```
3) Track data with DVC:
   ```bash
   dvc add data/zameen-updated.csv
   git add data/zameen-updated.csv.dvc .gitignore .dvc/config
   ```
4) Train (creates `models/model.pkl`, `metrics.json`, and `artifacts/columns.json`):
   ```bash
   dvc repro   # or: python src/train.py
   ```
5) Serve the model:
   ```bash
   flask --app app run
   # or: python app.py
   ```
   Example request:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"location_id":3325,"property_type":"House","location":"Bahria Town","city":"Lahore","province_name":"Punjab","latitude":31.4,"longitude":74.2,"baths":4,"area":1,"purpose":"For Sale","bedrooms":3,"Area Type":"Marla","Area Size":5.0,"Area Category":"0-5 Marla"}'
   ```

## DVC Pipeline
`dvc.yaml` defines a single `train` stage:
- **cmd**: `python src/train.py`
- **deps**: `data/zameen-updated.csv`, `src/train.py`, `params.yaml`
- **outs**: `models/model.pkl` (cached), `artifacts/columns.json`
- **metrics**: `metrics.json` (no cache)

## Notes
- Update `params.yaml` to tune model/test split without changing code.
- Push Git commits plus `dvc push` to sync data/models to the configured remote (e.g., `./dvc_store`, S3, GDrive).
