import json
import pathlib
from typing import Dict, List

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_params(path: pathlib.Path) -> Dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_preprocessor(df: pd.DataFrame, drop_cols: List[str]) -> ColumnTransformer:
    feature_df = df.drop(columns=drop_cols, errors="ignore")
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )


def train(config: Dict) -> Dict:
    data_path = pathlib.Path(config["data"]["path"])
    target_col = config["data"]["target"]
    drop_cols = config["data"].get("drop_columns", [])

    df = pd.read_csv(data_path)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    preprocessor = build_preprocessor(X, drop_cols)

    model_cfg = config["model"]
    if model_cfg["type"] != "RandomForestRegressor":
        raise ValueError("Only RandomForestRegressor is supported in this template.")

    model = RandomForestRegressor(**model_cfg["params"])
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        shuffle=config["training"].get("shuffle", True),
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    rmse = (mean_squared_error(y_test, preds) ** 0.5)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": rmse,
        "r2": r2_score(y_test, preds),
        "test_size": len(y_test),
    }

    model_dir = pathlib.Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(clf, model_path)

    artifacts_dir = pathlib.Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    columns_path = artifacts_dir / "columns.json"
    with columns_path.open("w") as f:
        json.dump(list(X.columns), f, indent=2)

    metrics_path = pathlib.Path("metrics.json")
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "columns_path": str(columns_path),
        "metrics": metrics,
    }


def main() -> None:
    params = load_params(pathlib.Path("params.yaml"))
    result = train(params)
    print("Model saved to:", result["model_path"])
    print("Metrics saved to:", result["metrics_path"])
    print("Feature columns saved to:", result["columns_path"])
    print("Metrics:", json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
