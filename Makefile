.PHONY: setup data eda train evaluate interpret pipeline test clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	python -m src.load_data

eda:
	python -m src.eda

train:
	python -m src.train --retrain

evaluate:
	python -m src.evaluate

interpret:
	python -m src.interpret

pipeline:
	python -m src.run_pipeline

test:
	python -m pytest tests/ -v

clean:
	rm -rf models/*.joblib models/*.json reports/eda_summary.json
	find reports/figures -name '*.png' ! -name '.gitkeep' -delete
