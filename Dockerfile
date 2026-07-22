# Build a self-contained image for the GraphRAG API.
FROM python:3.12-slim

# uv: fast, reproducible dependency install (same tool you use locally)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (this layer is cached unless pyproject changes,
# so rebuilds after a code edit are fast).
COPY pyproject.toml ./
RUN uv pip install --system --no-cache \
    fastapi "uvicorn[standard]" neo4j "neo4j-graphrag>=1.17" openai \
    pydantic pydantic-settings python-dotenv tqdm

# Now copy the application code.
COPY src/ ./src/

ENV PYTHONPATH=/app/src
EXPOSE 8000

# Launch the API. host 0.0.0.0 = listen on all interfaces so the port is
# reachable from outside the container (127.0.0.1 would only serve itself).
CMD ["uvicorn", "graphrag_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
