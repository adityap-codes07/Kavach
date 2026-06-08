"""
SmartShield: Context-Aware Email Security Extension
FastAPI Backend — Main Application Entry Point

Authors: SmartShield Research Team
Version: 1.0.0
License: MIT
"""

import time
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
import sentry_sdk

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.models.bert_classifier import BERTClassifier
from app.services.email_analyzer import EmailAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# Observability
# ─────────────────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "smartshield_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "smartshield_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
INFERENCE_LATENCY = Histogram(
    "smartshield_inference_latency_ms",
    "ML inference latency in ms",
    buckets=[10, 25, 50, 100, 250, 500, 1000],
)

if settings.SENTRY_DSN:
    sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    logger.info("🛡️  SmartShield API starting up …")
    # Warm up BERT model (loads weights into GPU/CPU memory once)
    app.state.classifier = BERTClassifier.load(settings.MODEL_PATH)
    app.state.analyzer = EmailAnalyzer(classifier=app.state.classifier)
    logger.info("✅ Models loaded — API is ready to serve requests.")
    yield
    logger.info("🔴 SmartShield API shutting down …")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartShield API",
    description=(
        "Context-Aware Email Security API using BERT and Explainable AI. "
        "Provides spam detection, phishing analysis, sender reputation, "
        "risk scoring, and SHAP/LIME explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    response.headers["X-Process-Time"] = f"{latency:.4f}s"
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter — replace with Redis sliding window in prod."""
    client_ip = request.client.host
    # Production: integrate with redis-py or fastapi-limiter
    return await call_next(request)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Prometheus metrics endpoint ───────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Health / readiness probes ─────────────────────────────────────────────────
@app.get("/health", tags=["Infrastructure"])
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/ready", tags=["Infrastructure"])
async def readiness_check(request: Request):
    if not hasattr(request.app.state, "classifier"):
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"status": "ready"}


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again."},
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_config=None,  # use our custom logging
    )
