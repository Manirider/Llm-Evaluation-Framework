FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.2

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root --no-interaction

# ---------------------------------------------------------------------------
# Final slim runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy only the virtual environment and source
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/

# Install the project in the venv so the entry-point is available
RUN pip install --no-deps -e .

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["python", "-c", "import llm_eval; print(llm_eval.__version__)"]

ENTRYPOINT ["llm-eval"]
CMD ["--help"]
