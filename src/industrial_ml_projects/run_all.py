from __future__ import annotations

import argparse
from pathlib import Path

from industrial_ml_projects.common import save_json
from industrial_ml_projects.projects import (
    distributed_eta,
    gnn_xgb_fraud,
    item_availability,
    llm_relevance,
    similarity_clustering,
)


PROJECTS = [
    distributed_eta.run,
    similarity_clustering.run,
    item_availability.run,
    gnn_xgb_fraud.run,
    llm_relevance.run,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--rows", type=int, default=8000, help="Approximate rows per synthetic tabular project.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = {}
    for run_project in PROJECTS:
        result = run_project(output_dir=output_dir, n_rows=args.rows)
        summary[result.name] = result.metrics
        print(f"[ok] {result.name}: {result.metrics}")

    save_json(summary, output_dir / "summary_metrics.json")
    print(f"\nWrote summary to {output_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()

