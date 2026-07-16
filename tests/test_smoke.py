from pathlib import Path

from industrial_ml_projects.projects import dropbox, instacart, stripe, uber


def test_core_projects_smoke(tmp_path: Path):
    results = [
        uber.run(tmp_path, n_rows=600),
        stripe.run(tmp_path, n_rows=600),
        instacart.run(tmp_path, n_rows=600),
        dropbox.run(tmp_path, n_rows=600),
    ]
    for result in results:
        assert result.artifact_dir.exists()
        assert result.metrics
