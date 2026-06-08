"""
SmartShield — Application Configuration
=========================================
All settings loaded from environment variables with sane defaults.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ────────────────────────────────────────────────────────────────
    DEBUG:    bool = False
    WORKERS:  int  = Field(default=4, ge=1, le=32)
    HOST:     str  = "0.0.0.0"
    PORT:     int  = 8000
    LOG_LEVEL: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://smartshield:password@localhost:5432/smartshield"
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # ── ML Model ──────────────────────────────────────────────────────────────
    MODEL_PATH: str   = Field(default="checkpoints/bert_v1")
    MODEL_DEVICE: str = "auto"    # "auto" | "cpu" | "cuda" | "mps"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32

    # ── Security ──────────────────────────────────────────────────────────────
    VIRUSTOTAL_API_KEY:    Optional[str] = None
    SAFE_BROWSING_API_KEY: Optional[str] = None
    WHOIS_TIMEOUT_SECONDS: float = 3.0
    DNS_TIMEOUT_SECONDS:   float = 3.0

    # ── API ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "chrome-extension://*"]
    )
    API_RATE_LIMIT_PER_MINUTE: int = 60
    MAX_EMAIL_LENGTH: int = 50_000
    MAX_BATCH_SIZE:   int = 50
    MAX_FILE_SIZE_MB: float = 5.0

    # ── Cache ─────────────────────────────────────────────────────────────────
    ANALYSIS_CACHE_TTL_SECONDS: int = 300    # 5 minutes
    URL_CACHE_TTL_SECONDS:      int = 3600   # 1 hour

    # ── Observability ─────────────────────────────────────────────────────────
    SENTRY_DSN:              Optional[str] = None
    PROMETHEUS_ENABLED: bool = True
    TRACE_SAMPLE_RATE: float = 0.1

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.MAX_FILE_SIZE_MB * 1_000_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
