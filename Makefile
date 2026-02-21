.PHONY: setup install test lint run-api run-dashboard docker-build

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	pip install -e .

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	black src/ tests/
	flake8 src/ tests/ --max-line-length=100

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run src/dashboard/app.py

docker-build:
	docker-compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
