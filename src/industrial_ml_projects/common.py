from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_model(model: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def split_xy(df: pd.DataFrame, target: str, test_size: float = 0.25):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y if y.nunique() == 2 else None)


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    positives = np.maximum(1, np.sum(y_true == 1))
    predicted = np.maximum(1, np.sum(y_pred == 1))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    precision = tp / predicted
    recall = tp / positives
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y_true, y_pred))}


def ranking_ndcg(labels: pd.Series, scores: pd.Series, query_ids: pd.Series, k: int = 10) -> float:
    values = []
    for _, idx in query_ids.groupby(query_ids).groups.items():
        if len(idx) > 1:
            values.append(ndcg_score([labels.iloc[list(idx)].to_numpy()], [scores.iloc[list(idx)].to_numpy()], k=k))
    return float(np.mean(values)) if values else 0.0


@dataclass(frozen=True)
class ProjectResult:
    name: str
    artifact_dir: Path
    metrics: dict[str, Any]
