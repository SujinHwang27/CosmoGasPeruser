FROM python:3.12-slim

# System dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY dvc.yaml dvc.lock ./

# Set PYTHONPATH so src.core imports work
ENV PYTHONPATH=/app

# Default: show pipeline status
CMD ["uv", "run", "dvc", "status"]
