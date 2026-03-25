# Dockerfile.app — Multi-stage build for the Streamlit frontend

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies (cached unless pyproject.toml / uv.lock change)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS deps

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev group).
# --frozen ensures uv.lock is not modified during the build.
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Stage 2: Final runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual-env created by uv in the deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application source
COPY shared/ ./shared/
COPY app/     ./app/
COPY ingestion/ ./ingestion/

# Put the venv on PATH so `streamlit` is found directly
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
