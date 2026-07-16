from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from industrial_ml_projects.common import ProjectResult, ensure_dir, regression_metrics, save_json, save_model, seed_everything


def make_eta_data(n_rows: int = 8000) -> pd.DataFrame:
    rng = seed_everything(101)
    city = rng.integers(0, 8, n_rows)
    hour = rng.integers(0, 24, n_rows)
    day = rng.integers(0, 7, n_rows)
    trip_miles = rng.gamma(2.2, 2.4, n_rows).clip(0.2, 35)
    traffic_index = rng.beta(2 + (hour // 8), 4, n_rows)
    rain = rng.binomial(1, 0.08 + 0.05 * (city % 3 == 0), n_rows)
    airport = rng.binomial(1, 0.09, n_rows)
    driver_supply = rng.normal(0.0, 1.0, n_rows)
    sparse_event = rng.binomial(1, 0.025, n_rows)

    eta = (
        4.5
        + trip_miles * (2.1 + traffic_index * 3.4)
        + rain * 5.0
        + airport * 7.5
        + sparse_event * 11.0
        + np.sin(hour / 24 * 2 * np.pi) * 2.5
        - driver_supply * 1.3
        + city * 0.35
        + rng.normal(0, 2.2, n_rows)
    ).clip(2, None)
    return pd.DataFrame(
        {
            "city_id": city,
            "hour": hour,
            "day_of_week": day,
            "trip_miles": trip_miles,
            "traffic_index": traffic_index,
            "rain": rain,
            "airport_trip": airport,
            "driver_supply_z": driver_supply,
            "sparse_event": sparse_event,
            "eta_minutes": eta,
        }
    )


def run(output_dir: Union[str, Path], n_rows: int = 8000) -> ProjectResult:
    artifact_dir = ensure_dir(Path(output_dir) / "01_uber_eta_prediction")
    df = make_eta_data(n_rows)
    data_path = artifact_dir / "synthetic_eta_training.csv"
    df.to_csv(data_path, index=False)

    X = pd.get_dummies(df.drop(columns=["eta_minutes"]), columns=["city_id", "hour", "day_of_week"], dtype=float)
    y = df["eta_minutes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.25)

    model = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.02, random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = regression_metrics(y_test, preds)

    golden = X_test.head(50)
    golden_preds = model.predict(golden)
    serving_mae_shift = float(np.abs(golden_preds - model.predict(golden.copy())).mean())
    importances = permutation_importance(model, X_test, y_test, n_repeats=4, random_state=101)
    top_features = sorted(
        zip(X.columns, importances.importances_mean),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    metrics.update(
        {
            "golden_serving_prediction_delta": serving_mae_shift,
            "top_features": [{"feature": f, "importance": float(v)} for f, v in top_features],
            "training_rows": int(len(df)),
        }
    )
    save_model({"model": model, "columns": list(X.columns)}, artifact_dir / "model.joblib")
    save_json(metrics, artifact_dir / "metrics.json")
    return ProjectResult("uber_eta_prediction", artifact_dir, metrics)
