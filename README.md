# Industrial ML Projects: Five End-to-End Synthetic Systems

This repository contains five portfolio-grade machine learning projects inspired by production engineering writeups from Uber, Stripe, Instacart, NVIDIA, and Dropbox. Every project uses synthetic data generated in code, so the repo is safe to publish and easy to reproduce.

The goal is to show the kind of implementation expected from an ML developer with roughly four years of experience: clear problem framing, realistic feature generation, training and evaluation code, saved artifacts, deployment-aware metrics, and documentation that explains the engineering tradeoffs.

## Source Inspiration

| Project | Inspired by | Production idea translated into this repo |
| --- | --- | --- |
| `01_distributed_eta_xgboost_style` | Uber, distributed XGBoost for large data training | Scalable tabular ETA regression, feature consistency checks, golden serving validation, model importance |
| `02_similarity_fraud_rings` | Stripe, similarity clustering for fraud rings | Supervised pairwise similarity model, candidate edge pruning, connected components for ring discovery |
| `03_item_availability_rta` | Instacart, real-time item availability architecture | Availability classifier, low-latency bulk scoring, threshold resolver, stackable deltas for segment tuning |
| `04_gnn_xgb_fraud` | NVIDIA, GNN embeddings plus XGBoost fraud detection | Synthetic transaction graph, graph-derived embeddings, fraud classifier, graph metadata in metrics |
| `05_llm_assisted_relevance` | Dropbox, LLM-assisted human labeling for search relevance | Human seed labels, teacher-generated labels, relevance ranker, NDCG evaluation |

## Repository Layout

```text
industrial_ml_projects/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── src/industrial_ml_projects/
│   ├── common.py
│   ├── run_all.py
│   └── projects/
│       ├── distributed_eta.py
│       ├── similarity_clustering.py
│       ├── item_availability.py
│       ├── gnn_xgb_fraud.py
│       └── llm_relevance.py
└── tests/
    └── test_smoke.py
```

Generated outputs go to `artifacts/` and are intentionally ignored by Git.

## Quickstart

```bash
cd "industrial_ml_projects"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m industrial_ml_projects.run_all --output-dir artifacts --rows 8000
```

Run smoke tests:

```bash
pip install pytest
python -m pytest tests -q
```

## Project Details

### 1. Distributed ETA XGBoost-Style Regression

Problem statement: predict ride ETA from trip distance, traffic, weather, city, time, and supply features.

What it demonstrates:

- Production-style tabular regression workflow inspired by distributed GBDT systems.
- Consistent feature schema saved with the model.
- Golden prediction check to catch training-serving drift.
- Permutation feature importance for post-training analysis.

Artifacts:

- `synthetic_eta_training.csv`
- `model.joblib`
- `metrics.json`

### 2. Similarity Fraud Ring Clustering

Problem statement: detect groups of fraudulent merchant accounts that reuse attributes such as bank accounts, devices, IP ranges, and merchant categories.

What it demonstrates:

- Synthetic account graph with known fraud rings.
- Pairwise feature generation and supervised similarity learning.
- Candidate edge pruning to avoid all-pairs scoring.
- Connected components to produce analyst-reviewable fraud clusters.

Artifacts:

- `synthetic_accounts.csv`
- `synthetic_pair_training.csv`
- `detected_clusters_sample.json`
- `model.joblib`
- `metrics.json`

### 3. Real-Time Item Availability

Problem statement: predict whether a grocery item will be found by a shopper and choose which items to show or suppress in customer-facing retrieval.

What it demonstrates:

- Availability model trained from synthetic catalog, retailer, region, basket, and inventory freshness signals.
- Segment-aware threshold resolver.
- Delta framework for product/category/region/user-bucket adjustments without retraining.
- Selection rate and found rate metrics, which are closer to product operations than raw AUC alone.

Artifacts:

- `synthetic_item_availability.csv`
- `threshold_config.json`
- `model.joblib`
- `metrics.json`

### 4. Graph-Enhanced Financial Fraud Detection

Problem statement: detect fraudulent transactions using both tabular transaction signals and graph structure from shared accounts, cards, and devices.

What it demonstrates:

- Synthetic heterogeneous transaction graph.
- Graph-derived embedding features using degree, PageRank, and clustering signals.
- Fraud classifier trained on graph plus transaction features.
- Metrics that include graph size, graph feature list, ROC AUC, and average precision.

Artifacts:

- `synthetic_graph_fraud_features.csv`
- `model.joblib`
- `metrics.json`

### 5. LLM-Assisted Search Relevance Labeling

Problem statement: train a search relevance model using a small human-labeled seed set and a larger teacher-labeled synthetic dataset.

What it demonstrates:

- Query-document candidate generation.
- Human label simulation on a 1-5 relevance scale.
- Teacher labeling policy that acts like an offline LLM evaluator.
- Relevance model trained from lexical similarity, click signals, rank position, and teacher labels.
- NDCG@10 ranking evaluation.

Artifacts:

- `synthetic_query_doc_labels.csv`
- `labeling_policy.json`
- `model.joblib`
- `metrics.json`

## Upload to GitHub

Create a new empty GitHub repository first, then run:

```bash
cd "industrial_ml_projects"
git init
git add .
git commit -m "Build five synthetic industrial ML projects"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

If you already have a remote:

```bash
git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

## Upload Artifacts to Hugging Face

Install and log in:

```bash
pip install huggingface_hub
huggingface-cli login
```

Run the pipelines so artifacts exist:

```bash
python -m industrial_ml_projects.run_all --output-dir artifacts --rows 8000
```

Create a model repository and upload:

```bash
huggingface-cli repo create industrial-ml-projects --type model --private false
huggingface-cli upload <YOUR_HF_USERNAME>/industrial-ml-projects artifacts . --repo-type model
```

To upload the source code as well:

```bash
huggingface-cli upload <YOUR_HF_USERNAME>/industrial-ml-projects . . --repo-type model \
  --exclude ".venv/*" \
  --exclude "__pycache__/*" \
  --exclude ".git/*"
```

## Notes on Synthetic Data

The data is not copied from any company. Each generator creates synthetic distributions that mimic the shape of the production problem:

- ETA data includes traffic, supply, sparse event, city, and weather effects.
- Fraud-ring data includes reused high-effort identifiers such as bank and device hashes.
- Availability data includes inventory freshness, regional shortages, promotions, and segment thresholds.
- Graph fraud data includes shared account-card-device structure.
- Relevance data includes query-document overlap, click disagreement, human seed labels, and amplified teacher labels.

This makes the projects publishable while still showing production-relevant ML judgment.

## References

- Uber Engineering: [Productionizing Distributed XGBoost to Train Deep Tree Models with Large Data Sets at Uber](https://www.uber.com/us/en/blog/productionizing-distributed-xgboost/)
- Stripe: [Similarity clustering to catch fraud rings](https://stripe.com/blog/similarity-clustering)
- Instacart: [Instacart’s Item Availability Architecture: Solving for scale and consistency](https://company.instacart.com/tech-innovation/instacarts-item-availability-architecture-solving-for-scale-and-consistency)
- NVIDIA Developer: [Supercharging Fraud Detection in Financial Services with Graph Neural Networks](https://developer.nvidia.com/blog/supercharging-fraud-detection-in-financial-services-with-graph-neural-networks/)
- Dropbox Tech: [Using LLMs to amplify human labeling and improve Dash search relevance](https://dropbox.tech/machine-learning/llm-human-labeling-improving-search-relevance-dropbox-dash)
