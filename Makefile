.PHONY: install up down ingest lint test

install:        ## install deps into a uv-managed venv
	uv sync

up:             ## start Neo4j in Docker
	docker compose up -d

down:           ## stop Neo4j
	docker compose down

ingest:         ## run the Phase 1 ingestion pipeline
	uv run python -m graphrag_assistant.ingest

lint:
	uv run ruff check .

test:
	uv run pytest -q
