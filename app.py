import json
import pathlib
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = pathlib.Path("models/model.pkl")
COLUMNS_PATH = pathlib.Path("artifacts/columns.json")

app = Flask(__name__)


def load_columns() -> List[str]:
    if not COLUMNS_PATH.exists():
        return []
    with COLUMNS_PATH.open() as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` or `dvc repro` first."
        )
    return joblib.load(MODEL_PATH)


MODEL = load_model()
TRAIN_COLUMNS = load_columns()


def normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError("Payload must be a JSON object or list of objects.")

    if TRAIN_COLUMNS:
        normalized = []
        for row in rows:
            row_data = {col: row.get(col) for col in TRAIN_COLUMNS}
            normalized.append(row_data)
        return normalized
    return rows


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.post("/predict")
def predict() -> Any:
    payload = request.get_json()
    if payload is None:
        return jsonify({"error": "Request must contain JSON body"}), 400

    try:
        rows = normalize_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    df = pd.DataFrame(rows)

    # Ensure columns align with training data
    if TRAIN_COLUMNS:
        for col in TRAIN_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[TRAIN_COLUMNS]

    predictions = MODEL.predict(df)
    return jsonify({"predictions": [float(p) for p in predictions]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
