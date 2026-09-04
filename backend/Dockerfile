# Kavach — Backend Dockerfile
# ===================================
# Multi-stage build:
#   builder  → installs Python deps with wheels
#   production → lean runtime image (~2.1GB with BERT weights)

# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ── Production stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS production

LABEL maintainer="Kavach Research Team"
LABEL version="1.0.0"
LABEL description="Kavach Email Security API"

# Non-root user for security
RUN groupadd -r kavach && useradd -r -g kavach kavach

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from builder
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create directories
RUN mkdir -p /app/models /app/logs && \
    chown -R kavach:kavach /app

# Switch to non-root
USER kavach

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKERS=4 \
    MODEL_PATH=/app/models/bert_v1 \
    LOG_LEVEL=INFO

# Run migrations then start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS} --loop uvloop --http httptools --log-config /app/logging.json"]
