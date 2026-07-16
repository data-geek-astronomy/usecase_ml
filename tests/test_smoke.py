from pathlib import Path

from industrial_ml_projects.projects import distributed_eta, item_availability, llm_relevance, similarity_clustering


def test_core_projects_smoke(tmp_path: Path):
    results = [
        distributed_eta.run(tmp_path, n_rows=600),
        similarity_clustering.run(tmp_path, n_rows=600),
        item_availability.run(tmp_path, n_rows=600),
        llm_relevance.run(tmp_path, n_rows=600),
    ]
    for result in results:
        assert result.artifact_dir.exists()
        assert result.metrics

