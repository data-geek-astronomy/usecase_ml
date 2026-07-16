from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from industrial_ml_projects.common import ProjectResult, ensure_dir, ranking_ndcg, regression_metrics, save_json, save_model, seed_everything


TOPICS = {
    "security": ["access", "sso", "policy", "incident", "audit", "permission"],
    "sales": ["pipeline", "account", "renewal", "forecast", "lead", "quota"],
    "engineering": ["deploy", "latency", "api", "migration", "bug", "release"],
    "hr": ["benefits", "onboarding", "review", "manager", "career", "training"],
    "finance": ["invoice", "budget", "expense", "contract", "revenue", "tax"],
}


def make_docs_and_queries(n_queries: int = 400, docs_per_query: int = 16) -> pd.DataFrame:
    rng = seed_everything(505)
    topics = list(TOPICS)
    docs = []
    for doc_id in range(900):
        topic = topics[doc_id % len(topics)]
        words = rng.choice(TOPICS[topic], size=5, replace=True).tolist()
        noise_topic = rng.choice([t for t in topics if t != topic])
        words += rng.choice(TOPICS[noise_topic], size=2, replace=False).tolist()
        docs.append({"doc_id": doc_id, "doc_topic": topic, "doc_text": f"{topic} " + " ".join(words)})

    rows = []
    for qid in range(n_queries):
        topic = topics[qid % len(topics)]
        intent_words = rng.choice(TOPICS[topic], size=3, replace=False).tolist()
        query = f"{topic} " + " ".join(intent_words)
        candidates = rng.choice(len(docs), size=docs_per_query, replace=False)
        for rank, doc_idx in enumerate(candidates):
            doc = docs[int(doc_idx)]
            overlap = len(set(query.split()) & set(doc["doc_text"].split()))
            same_topic = int(doc["doc_topic"] == topic)
            click = int((same_topic and rank < 8 and rng.random() < 0.55) or (overlap >= 3 and rng.random() < 0.35))
            human_label = int(np.clip(1 + overlap + 2 * same_topic + rng.normal(0, 0.35), 1, 5))
            rows.append({"query_id": qid, "query": query, "doc_id": doc["doc_id"], "doc_text": doc["doc_text"], "rank_position": rank + 1, "click": click, "human_label": human_label})
    return pd.DataFrame(rows)


def teacher_label(row: pd.Series) -> int:
    q_terms = set(row.query.split())
    d_terms = set(row.doc_text.split())
    overlap = len(q_terms & d_terms)
    click_boost = 1 if row.click and row.rank_position <= 8 else 0
    return int(np.clip(1 + overlap + click_boost, 1, 5))


def run(output_dir: Union[str, Path], n_rows: int = 8000) -> ProjectResult:
    artifact_dir = ensure_dir(Path(output_dir) / "05_dropbox_search_relevance")
    df = make_docs_and_queries(max(250, n_rows // 20))
    df["teacher_label"] = df.apply(teacher_label, axis=1)
    df.to_csv(artifact_dir / "synthetic_query_doc_labels.csv", index=False)

    human_seed = df.sample(frac=0.08, random_state=505)
    teacher_mse = float(np.mean((human_seed.teacher_label - human_seed.human_label) ** 2))

    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    corpus = pd.concat([df["query"], df["doc_text"]], ignore_index=True)
    vectorizer.fit(corpus)
    q_vec = vectorizer.transform(df["query"])
    d_vec = vectorizer.transform(df["doc_text"])
    lexical_similarity = np.asarray([cosine_similarity(q_vec[i], d_vec[i])[0, 0] for i in range(len(df))])

    features = pd.DataFrame(
        {
            "lexical_similarity": lexical_similarity,
            "rank_position": df["rank_position"],
            "click": df["click"],
            "teacher_label": df["teacher_label"],
        }
    )
    y = df["human_label"]
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(features, y, df, test_size=0.25, random_state=505)
    model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06, max_leaf_nodes=15, random_state=505)
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 1, 5)
    metrics = regression_metrics(y_test, preds)
    metrics.update(
        {
            "teacher_vs_human_mse_on_seed": teacher_mse,
            "ranker_ndcg_at_10": ranking_ndcg(y_test.reset_index(drop=True), pd.Series(preds), meta_test.reset_index(drop=True)["query_id"], k=10),
            "human_seed_rows": int(len(human_seed)),
            "amplified_teacher_rows": int(len(df)),
        }
    )

    save_model({"model": model, "vectorizer": vectorizer, "feature_columns": list(features.columns)}, artifact_dir / "model.joblib")
    save_json(metrics, artifact_dir / "metrics.json")
    save_json({"prompt_contract": "Use the query, document text, click disagreement, and organizational context to assign a 1-5 relevance label; calibrate against human seed labels before scaling."}, artifact_dir / "labeling_policy.json")
    return ProjectResult("dropbox_search_relevance", artifact_dir, metrics)
