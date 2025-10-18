.PHONY: help install up down run evaluate test lint format clean

help:
	@echo "IoMT Swarm Range - Available Commands:"
	@echo ""
	@echo "  make install    - Install dependencies with Poetry"
	@echo "  make up         - Build and start Docker services"
	@echo "  make down       - Stop and remove containers"
	@echo "  make run        - Run default scenario"
	@echo "  make evaluate   - Compute metrics and generate figures"
	@echo "  make test       - Run unit tests"
	@echo "  make test-int   - Run integration tests"
	@echo "  make lint       - Run linters and type checks"
	@echo "  make format     - Format code with Ruff"
	@echo "  make clean      - Clean generated files"
	@echo ""

install:
	poetry install
	@echo "✓ Dependencies installed"

up:
	docker compose -f deploy/docker/docker-compose.yml up -d --build
	@echo "✓ Services started"

down:
	docker compose -f deploy/docker/docker-compose.yml down -v
	@echo "✓ Services stopped"

run:
	poetry run python scripts/run_scenario.py --config conf/experiment.yaml

evaluate:
	poetry run python scripts/export_results.py
	@echo "✓ Results exported"

test:
	poetry run pytest tests/unit -v --cov --cov-report=term-missing

test-int:
	poetry run pytest tests/integration -v --log-cli-level=INFO

lint:
	poetry run ruff check .
	@echo "✓ Linting complete"

format:
	poetry run ruff format .
	poetry run ruff check . --fix
	@echo "✓ Code formatted"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✓ Cleaned temporary files"
