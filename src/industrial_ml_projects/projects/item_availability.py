from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from industrial_ml_projects.common import ProjectResult, classification_metrics, ensure_dir, save_json, save_model, seed_everything


def make_availability_data(n_rows: int = 8000) -> pd.DataFrame:
    rng = seed_everything(303)
    category = rng.integers(0, 9, n_rows)
    retailer = rng.integers(0, 18, n_rows)
    region = rng.integers(0, 6, n_rows)
    user_bucket = rng.integers(0, 8, n_rows)
    day = rng.integers(0, 7, n_rows)
    hour = rng.integers(7, 23, n_rows)
    recent_found_rate = rng.beta(8 - 0.35 * (category == 2), 2.5, n_rows)
    inventory_age_hours = rng.gamma(2.0, 8.0, n_rows)
    basket_qty = rng.poisson(2.4, n_rows) + 1
    promo = rng.binomial(1, 0.18, n_rows)
    shortage = ((category == 2) & (region <= 2)) | ((retailer % 7 == 0) & (day >= 5))

    logit = (
        -0.2
        + recent_found_rate * 4.0
        - np.log1p(inventory_age_hours) * 0.65
        - basket_qty * 0.08
        - promo * 0.35
        - shortage.astype(float) * 1.1
        + (hour < 12) * 0.3
        + rng.normal(0, 0.45, n_rows)
    )
    prob = 1 / (1 + np.exp(-logit))
    found = rng.binomial(1, prob)
    return pd.DataFrame(
        {
            "category": category,
            "retailer": retailer,
            "region": region,
            "user_bucket": user_bucket,
            "day": day,
            "hour": hour,
            "recent_found_rate": recent_found_rate,
            "inventory_age_hours": inventory_age_hours,
            "basket_qty": basket_qty,
            "promo": promo,
            "found": found,
        }
    )


def threshold_for(row: pd.Series, base_thresholds: dict[int, float]) -> float:
    threshold = base_thresholds[int(row.category)]
    threshold += -0.04 if row.user_bucket in {0, 1} else 0.0
    threshold += 0.06 if row.category == 2 and row.region <= 2 else 0.0
    threshold += 0.03 if row.promo == 1 else 0.0
    return float(np.clip(threshold, 0.2, 0.9))


def run(output_dir: Union[str, Path], n_rows: int = 8000) -> ProjectResult:
    artifact_dir = ensure_dir(Path(output_dir) / "03_item_availability_rta")
    df = make_availability_data(n_rows)
    df.to_csv(artifact_dir / "synthetic_item_availability.csv", index=False)

    X = pd.get_dummies(df.drop(columns=["found"]), columns=["category", "retailer", "region", "user_bucket", "day", "hour"], dtype=float)
    y = df["found"]
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(X, y, df, test_size=0.25, random_state=303, stratify=y)
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, max_leaf_nodes=25, random_state=303)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    base_metrics = classification_metrics(y_test.to_numpy(), scores, threshold=0.55)

    base_thresholds = {int(c): float(0.48 + 0.015 * c) for c in sorted(df.category.unique())}
    thresholds = raw_test.apply(lambda row: threshold_for(row, base_thresholds), axis=1)
    displayed = scores >= thresholds.to_numpy()
    selection_rate = float(displayed.mean())
    found_rate_when_displayed = float(y_test.to_numpy()[displayed].mean()) if displayed.any() else 0.0
    metrics = {
        **base_metrics,
        "selection_rate": selection_rate,
        "found_rate_when_displayed": found_rate_when_displayed,
        "threshold_strategy": "base threshold per category plus stacked deltas for user bucket, shortage region, and promotion",
    }

    save_model({"model": model, "columns": list(X.columns), "base_thresholds": base_thresholds}, artifact_dir / "model.joblib")
    save_json(metrics, artifact_dir / "metrics.json")
    save_json({"base_thresholds": base_thresholds, "example_deltas": {"new_user_bucket": -0.04, "shortage_category_region": 0.06, "promo": 0.03}}, artifact_dir / "threshold_config.json")
    return ProjectResult("item_availability_rta", artifact_dir, metrics)
