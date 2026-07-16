from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from industrial_ml_projects.common import ProjectResult, classification_metrics, ensure_dir, save_json, save_model, seed_everything


DOMAINS = ["gmail.com", "outlook.com", "proton.me", "merchant.io", "shop.co", "fastmail.com"]
MCCS = ["apparel", "electronics", "digital", "travel", "grocery", "services"]


def make_accounts(n_accounts: int = 1800) -> pd.DataFrame:
    rng = seed_everything(202)
    n_rings = max(8, n_accounts // 120)
    rows = []
    for account_id in range(n_accounts):
        ring = account_id % n_rings if account_id < n_rings * 14 else -1
        base = ring if ring >= 0 else account_id
        fraud = int(ring >= 0 and rng.random() < 0.85)
        rows.append(
            {
                "account_id": account_id,
                "fraud_ring": ring,
                "is_fraud": fraud,
                "email_domain": DOMAINS[base % len(DOMAINS)] if ring >= 0 else rng.choice(DOMAINS),
                "bank_hash": f"bank_{base % 37}" if ring >= 0 else f"bank_{rng.integers(0, 800)}",
                "device_hash": f"dev_{base % 49}" if ring >= 0 else f"dev_{rng.integers(0, 1200)}",
                "ip_block": f"10.{base % 60}.{rng.integers(0, 255)}" if ring >= 0 else f"10.{rng.integers(0, 220)}.{rng.integers(0, 255)}",
                "merchant_category": MCCS[base % len(MCCS)] if ring >= 0 else rng.choice(MCCS),
                "avg_ticket": float(rng.lognormal(3.2 + fraud * 0.25, 0.45)),
                "chargeback_rate": float(rng.beta(2 + fraud * 7, 18)),
            }
        )
    return pd.DataFrame(rows)


def pair_features(a: pd.Series, b: pd.Series) -> dict[str, float]:
    return {
        "same_email_domain": float(a.email_domain == b.email_domain),
        "same_bank": float(a.bank_hash == b.bank_hash),
        "same_device": float(a.device_hash == b.device_hash),
        "same_ip_prefix": float(a.ip_block.split(".")[1] == b.ip_block.split(".")[1]),
        "same_category": float(a.merchant_category == b.merchant_category),
        "ticket_ratio": float(min(a.avg_ticket, b.avg_ticket) / max(a.avg_ticket, b.avg_ticket)),
        "chargeback_gap": float(abs(a.chargeback_rate - b.chargeback_rate)),
        "label": float(a.fraud_ring >= 0 and a.fraud_ring == b.fraud_ring),
        "left_id": int(a.account_id),
        "right_id": int(b.account_id),
    }


def make_pairs(accounts: pd.DataFrame, max_negative: int = 5000) -> pd.DataFrame:
    rng = seed_everything(203)
    rows = []
    for _, group in accounts[accounts.fraud_ring >= 0].groupby("fraud_ring"):
        for i, j in combinations(group.index[:10], 2):
            rows.append(pair_features(accounts.loc[i], accounts.loc[j]))
    for _ in range(max_negative):
        i, j = rng.choice(accounts.index, 2, replace=False)
        rows.append(pair_features(accounts.loc[i], accounts.loc[j]))
    return pd.DataFrame(rows)


def run(output_dir: Union[str, Path], n_rows: int = 8000) -> ProjectResult:
    artifact_dir = ensure_dir(Path(output_dir) / "02_similarity_fraud_rings")
    accounts = make_accounts(max(1200, n_rows // 4))
    pairs = make_pairs(accounts)
    accounts.to_csv(artifact_dir / "synthetic_accounts.csv", index=False)
    pairs.to_csv(artifact_dir / "synthetic_pair_training.csv", index=False)

    feature_cols = [c for c in pairs.columns if c not in {"label", "left_id", "right_id"}]
    X_train, X_test, y_train, y_test = train_test_split(
        pairs[feature_cols], pairs["label"].astype(int), test_size=0.25, random_state=203, stratify=pairs["label"].astype(int)
    )
    model = GradientBoostingClassifier(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=203)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test.to_numpy(), scores, threshold=0.55)

    candidate_edges = pairs.loc[pairs[["same_bank", "same_device", "same_ip_prefix"]].sum(axis=1) > 0].copy()
    candidate_edges["score"] = model.predict_proba(candidate_edges[feature_cols])[:, 1]
    graph = nx.Graph()
    graph.add_nodes_from(accounts.account_id.tolist())
    graph.add_edges_from(candidate_edges.loc[candidate_edges.score >= 0.55, ["left_id", "right_id"]].itertuples(index=False, name=None))
    clusters = [sorted(c) for c in nx.connected_components(graph) if len(c) >= 3]
    metrics.update({"candidate_edges": int(len(candidate_edges)), "clusters_found": int(len(clusters)), "largest_cluster_size": int(max(map(len, clusters), default=0))})

    save_model({"model": model, "feature_cols": feature_cols}, artifact_dir / "model.joblib")
    save_json(metrics, artifact_dir / "metrics.json")
    save_json({"clusters": clusters[:20]}, artifact_dir / "detected_clusters_sample.json")
    return ProjectResult("similarity_fraud_rings", artifact_dir, metrics)
