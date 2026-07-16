---
title: "Usecase ML"
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Usecase ML

This repository contains five machine learning projects inspired by engineering work from Uber, Stripe, Instacart, NVIDIA, and Dropbox.

The data is synthetic. Nothing here is copied from any company system. Each project creates its own data, trains a model, saves metrics, and writes model artifacts. The goal is to make the work easy to run, easy to review, and easy to explain in interviews.

This project is written for a developer with about four years of machine learning experience. It shows practical thinking around data generation, feature design, model training, evaluation, and deployment readiness.

## Projects

### Uber ETA Prediction

![Uber project image](assets/company_logos/uber.jpeg)

File: `src/industrial_ml_projects/projects/uber.py`

This project predicts ride arrival time using trip distance, city, traffic, weather, airport trips, driver supply, and time based patterns.

Why it matters:

People care about arrival time being accurate. A small error can change pickup planning, pricing, dispatch, and customer trust.

What the project shows:

* Synthetic ride data generation
* Tabular regression model training
* Feature schema saved with the model
* Golden prediction check for serving consistency
* Feature importance for model review

Main outputs:

* `synthetic_eta_training.csv`
* `model.joblib`
* `metrics.json`

### Stripe Fraud Ring Detection

![Stripe project image](assets/company_logos/stripe.jpeg)

File: `src/industrial_ml_projects/projects/stripe.py`

This project finds connected groups of suspicious merchant accounts that reuse signals like bank accounts, devices, IP ranges, email domains, and merchant categories.

Why it matters:

Fraud often does not happen through one account. It happens through groups of accounts that look separate at first glance but share hidden patterns.

What the project shows:

* Synthetic merchant account data
* Pair based similarity features
* Supervised model for account similarity
* Graph construction from high confidence account pairs
* Connected component discovery for analyst review

Main outputs:

* `synthetic_accounts.csv`
* `synthetic_pair_training.csv`
* `detected_clusters_sample.json`
* `model.joblib`
* `metrics.json`

### Instacart Item Availability

![Instacart project image](assets/company_logos/instacart.png)

File: `src/industrial_ml_projects/projects/instacart.py`

This project predicts whether an item will actually be found by a shopper. It uses signals such as category, retailer, region, inventory age, basket size, promotion status, and recent found rate.

Why it matters:

If a grocery app shows items that are often unavailable, customers lose trust. If it hides too many items, customers lose choice. The model helps balance both sides.

What the project shows:

* Synthetic grocery availability data
* Availability classifier training
* Segment level threshold logic
* Product decision metrics such as selection rate and found rate
* A threshold config that can be adjusted without retraining the model

Main outputs:

* `synthetic_item_availability.csv`
* `threshold_config.json`
* `model.joblib`
* `metrics.json`

### NVIDIA Graph Fraud Detection

![NVIDIA project image](assets/company_logos/nvidia.jpg)

File: `src/industrial_ml_projects/projects/nvidia.py`

This project detects fraud in financial transactions by combining normal transaction features with graph features from shared accounts, cards, and devices.

Why it matters:

Fraud patterns are often easier to spot in a network than in a single row of data. Shared devices and cards can reveal risk that a plain table may miss.

What the project shows:

* Synthetic transaction graph creation
* Graph features such as degree, PageRank, and clustering
* Fraud model trained on transaction and graph signals
* Fraud metrics such as ROC AUC and average precision
* Graph metadata saved with model results

Main outputs:

* `synthetic_graph_fraud_features.csv`
* `model.joblib`
* `metrics.json`

### Dropbox Search Relevance

![Dropbox project image](assets/company_logos/dropbox.jpeg)

File: `src/industrial_ml_projects/projects/dropbox.py`

This project trains a search relevance model using a small set of human style labels and a larger set of teacher labels. It simulates how human review and model assisted labeling can improve search quality.

Why it matters:

Search is only useful when the best documents rise to the top. Human labels are valuable but expensive, so a careful teacher label workflow can help scale training data.

What the project shows:

* Synthetic query and document generation
* Human style labels on a relevance scale
* Teacher label generation
* Ranking model training
* NDCG evaluation for search quality

Main outputs:

* `synthetic_query_doc_labels.csv`
* `labeling_policy.json`
* `model.joblib`
* `metrics.json`

## Folder Structure

```text
industrial_ml_projects/
├── README.md
├── app.py
├── assets/company_logos/
├── requirements.txt
├── pyproject.toml
├── Makefile
├── src/industrial_ml_projects/
│   ├── common.py
│   ├── run_all.py
│   └── projects/
│       ├── uber.py
│       ├── stripe.py
│       ├── instacart.py
│       ├── nvidia.py
│       └── dropbox.py
└── tests/
    └── test_smoke.py
```

Generated model files and data files go into `artifacts/`. That folder is ignored by Git because it can be recreated at any time.

## Run Locally

```bash
cd "industrial_ml_projects"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m industrial_ml_projects.run_all --output-dir artifacts --rows 8000
```

Run tests:

```bash
pip install pytest
python -m pytest tests -q
```

## Push To GitHub

Use this repository:

```text
https://github.com/data-geek-astronomy/usecase_ml
```

Commands:

```bash
cd "/Users/aravindkumar/Research agent/industrial_ml_projects"
git add .
git commit -m "Rename projects by company and improve README"
git remote set-url origin https://github.com/data-geek-astronomy/usecase_ml.git
git push origin main
```

If `origin` does not exist yet:

```bash
git remote add origin https://github.com/data-geek-astronomy/usecase_ml.git
git push -u origin main
```

## Push To Hugging Face Space

Use this Space:

```text
https://huggingface.co/spaces/Darkweb007/Usecase_ML
```

Commands:

```bash
cd "/Users/aravindkumar/Research agent/industrial_ml_projects"
git add .
git commit -m "Update Hugging Face Space files"
git remote set-url hf https://huggingface.co/spaces/Darkweb007/Usecase_ML
git push hf main
```

If `hf` does not exist yet:

```bash
git remote add hf https://huggingface.co/spaces/Darkweb007/Usecase_ML
git push -u hf main
```

## References

Uber Engineering: [Productionizing Distributed XGBoost to Train Deep Tree Models with Large Data Sets at Uber](https://www.uber.com/us/en/blog/productionizing-distributed-xgboost/)

Stripe: [Similarity clustering to catch fraud rings](https://stripe.com/blog/similarity-clustering)

Instacart: [Instacart item availability architecture](https://company.instacart.com/tech-innovation/instacarts-item-availability-architecture-solving-for-scale-and-consistency)

NVIDIA Developer: [Fraud detection with graph neural networks](https://developer.nvidia.com/blog/supercharging-fraud-detection-in-financial-services-with-graph-neural-networks/)

Dropbox Tech: [Improving search relevance with LLM assisted human labeling](https://dropbox.tech/machine-learning/llm-human-labeling-improving-search-relevance-dropbox-dash)
