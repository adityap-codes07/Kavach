"""
SmartShield — Database Schema (SQLAlchemy 2.0)
================================================
Tables:
  analyses          — one row per email analyzed
  url_results       — one row per URL found in an email
  feedback          — user feedback (correct / incorrect classification)
  model_versions    — track deployed model checkpoints
  rate_limits       — per-IP rate limit tracking (Redis alternative)

Migration:
  alembic upgrade head
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Analysis — one row per email scanned
# ─────────────────────────────────────────────────────────────────────────────
class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email_hash: Mapped[str]      = mapped_column(String(64), index=True, nullable=False)
    subject:    Mapped[str]      = mapped_column(Text, nullable=False, default="")
    sender:     Mapped[str]      = mapped_column(String(320), nullable=False, default="")
    from_domain:Mapped[str]      = mapped_column(String(255), nullable=False, default="", index=True)

    # Classification outputs
    classification: Mapped[str]  = mapped_column(String(20), nullable=False, index=True)
    risk_score:     Mapped[int]  = mapped_column(Integer, nullable=False, index=True)
    risk_level:     Mapped[str]  = mapped_column(String(20), nullable=False)
    confidence:     Mapped[float]= mapped_column(Float, nullable=False)

    # BERT probabilities
    prob_legitimate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prob_spam:       Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prob_phishing:   Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Risk breakdown
    bert_contribution:   Mapped[float] = mapped_column(Float, default=0.0)
    url_contribution:    Mapped[float] = mapped_column(Float, default=0.0)
    header_contribution: Mapped[float] = mapped_column(Float, default=0.0)
    sender_contribution: Mapped[float] = mapped_column(Float, default=0.0)
    keyword_contribution:Mapped[float] = mapped_column(Float, default=0.0)

    # Header authentication
    spf_pass:  Mapped[bool] = mapped_column(Boolean, default=False)
    dkim_pass: Mapped[bool] = mapped_column(Boolean, default=False)
    dmarc_pass:Mapped[bool] = mapped_column(Boolean, default=False)

    # URL summary
    urls_found:           Mapped[int] = mapped_column(Integer, default=0)
    malicious_url_count:  Mapped[int] = mapped_column(Integer, default=0)
    newly_reg_domain_count:Mapped[int]= mapped_column(Integer, default=0)

    # Flagged keywords (stored as comma-separated)
    flagged_keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # XAI explanation summary
    explain_method:  Mapped[str] = mapped_column(String(100), default="attention")
    explain_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Performance
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bert_latency_ms:  Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    model_version:    Mapped[str]   = mapped_column(String(50), default="1.0.0")

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    source: Mapped[str] = mapped_column(
        String(50), default="extension"
    )  # extension | api | batch

    # Relationships
    url_results: Mapped[List["URLResult"]] = relationship(
        "URLResult", back_populates="analysis", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="analysis"
    )

    def __repr__(self) -> str:
        return (
            f"<Analysis id={self.id} email_hash={self.email_hash} "
            f"classification={self.classification} risk={self.risk_score}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# URL Result — one row per URL found
# ─────────────────────────────────────────────────────────────────────────────
class URLResult(Base):
    __tablename__ = "url_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    url:    Mapped[str]  = mapped_column(Text, nullable=False)
    domain: Mapped[str]  = mapped_column(String(255), nullable=False, index=True)

    risk_score:            Mapped[float] = mapped_column(Float, default=0.0)
    is_malicious:          Mapped[bool]  = mapped_column(Boolean, default=False)
    is_newly_registered:   Mapped[bool]  = mapped_column(Boolean, default=False)
    domain_age_days:       Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    uses_ip_address:       Mapped[bool]  = mapped_column(Boolean, default=False)
    has_suspicious_tld:    Mapped[bool]  = mapped_column(Boolean, default=False)
    is_typosquat:          Mapped[bool]  = mapped_column(Boolean, default=False)
    typosquat_target:      Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    redirect_count:        Mapped[int]   = mapped_column(Integer, default=0)
    virustotal_hits:       Mapped[int]   = mapped_column(Integer, default=0)
    google_safebrowsing_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    flags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="url_results")


# ─────────────────────────────────────────────────────────────────────────────
# Feedback — user correction of model decisions
# ─────────────────────────────────────────────────────────────────────────────
class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id"), index=True
    )
    correct_label: Mapped[str]   = mapped_column(String(20), nullable=False)
    model_was_wrong: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime]  = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="feedbacks")


# ─────────────────────────────────────────────────────────────────────────────
# ModelVersion — track checkpoint deployments
# ─────────────────────────────────────────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version:        Mapped[str]   = mapped_column(String(50), unique=True, nullable=False)
    model_type:     Mapped[str]   = mapped_column(String(50), nullable=False)
    accuracy:       Mapped[float] = mapped_column(Float, nullable=False)
    macro_f1:       Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc:        Mapped[float] = mapped_column(Float, nullable=False)
    p50_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    train_samples:  Mapped[int]   = mapped_column(Integer, nullable=False)
    is_active:      Mapped[bool]  = mapped_column(Boolean, default=False, index=True)
    deployed_at:    Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Database session factory (async)
# ─────────────────────────────────────────────────────────────────────────────
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_engine_and_session(database_url: str):
    """
    Create async engine and session factory.

    Example database_url:
      postgresql+asyncpg://user:password@localhost:5432/smartshield
    """
    engine = create_async_engine(
        database_url,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        echo=False,
    )
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return engine, session_factory


async def init_db(engine):
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
