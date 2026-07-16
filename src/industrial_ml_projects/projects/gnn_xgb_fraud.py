from __future__ import annotations

from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from industrial_ml_projects.common import ProjectResult, classification_metrics, ensure_dir, save_json, save_model, seed_everything


def make_transaction_graph(n_transactions: int = 8000) -> tuple[pd.DataFrame, nx.Graph]:
    rng = seed_everything(404)
    n_accounts, n_devices, n_cards = 1600, 1100, 1800
    risky_accounts = set(rng.choice(n_accounts, size=90, replace=False))
    rows = []
    graph = nx.Graph()
    for i in range(n_transactions):
        account = int(rng.integers(0, n_accounts))
        is_risky = account in risky_accounts
        device = int(account % 80 if is_risky and rng.random() < 0.65 else rng.integers(0, n_devices))
        card = int(account % 120 if is_risky and rng.random() < 0.55 else rng.integers(0, n_cards))
        amount = float(rng.lognormal(3.1 + 0.45 * is_risky, 0.7))
        velocity_1h = int(rng.poisson(2.0 + 5.0 * is_risky))
        foreign_ip = int(rng.binomial(1, 0.09 + 0.25 * is_risky))
        prob = 1 / (1 + np.exp(-(-4.0 + 0.025 * amount + 0.42 * velocity_1h + 1.1 * foreign_ip + 1.4 * is_risky)))
        fraud = int(rng.binomial(1, min(prob, 0.95)))
        rows.append({"transaction_id": i, "account": account, "device": device, "card": card, "amount": amount, "velocity_1h": velocity_1h, "foreign_ip": foreign_ip, "fraud": fraud})
        graph.add_edge(f"acct_{account}", f"dev_{device}")
        graph.add_edge(f"acct_{account}", f"card_{card}")
    return pd.DataFrame(rows), graph


def graph_embeddings(df: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    degree = dict(graph.degree())
    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=60)
    clustering = nx.clustering(graph)
    features = []
    for row in df.itertuples(index=False):
        account_node = f"acct_{row.account}"
        device_node = f"dev_{row.device}"
        card_node = f"card_{row.card}"
        features.append(
            {
                "account_degree": degree.get(account_node, 0),
                "device_degree": degree.get(device_node, 0),
                "card_degree": degree.get(card_node, 0),
                "account_pagerank": pagerank.get(account_node, 0),
                "device_pagerank": pagerank.get(device_node, 0),
                "card_pagerank": pagerank.get(card_node, 0),
                "account_clustering": clustering.get(account_node, 0),
            }
        )
    return pd.DataFrame(features)


def run(output_dir: Union[str, Path], n_rows: int = 8000) -> ProjectResult:
    artifact_dir = ensure_dir(Path(output_dir) / "04_gnn_xgb_fraud")
    tx, graph = make_transaction_graph(n_rows)
    gfeat = graph_embeddings(tx, graph)
    data = pd.concat([tx[["amount", "velocity_1h", "foreign_ip", "fraud"]], gfeat], axis=1)
    data.to_csv(artifact_dir / "synthetic_graph_fraud_features.csv", index=False)

    X = data.drop(columns=["fraud"])
    y = data["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=404, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = GradientBoostingClassifier(n_estimators=160, learning_rate=0.05, max_depth=3, random_state=404)
    model.fit(X_train_scaled, y_train)
    scores = model.predict_proba(X_test_scaled)[:, 1]
    metrics = classification_metrics(y_test.to_numpy(), scores, threshold=0.45)
    metrics.update({"nodes": int(graph.number_of_nodes()), "edges": int(graph.number_of_edges()), "graph_features": list(gfeat.columns)})

    save_model({"model": model, "scaler": scaler, "columns": list(X.columns)}, artifact_dir / "model.joblib")
    save_json(metrics, artifact_dir / "metrics.json")
    return ProjectResult("gnn_xgb_fraud", artifact_dir, metrics)
