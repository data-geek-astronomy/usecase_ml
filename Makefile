.PHONY: setup run test clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	python -m industrial_ml_projects.run_all --output-dir artifacts

test:
	python -m pytest tests -q

clean:
	rm -rf artifacts data .pytest_cache

