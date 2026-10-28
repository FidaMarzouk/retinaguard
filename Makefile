.PHONY: setup lint format test pipeline serve app up down

UV ?= uv

setup:
	$(UV) sync --all-extras
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest

pipeline:
	$(UV) run python -m retinaguard.pipeline

serve:
	$(UV) run uvicorn retinaguard.serving.api:app --reload --port 8000

app:
	$(UV) run streamlit run src/retinaguard/app/main.py

up:
	docker compose up -d --build

down:
	docker compose down
