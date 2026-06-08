"""
SmartShield — Email Analyzer Service
=====================================
Orchestrates: BERT inference, URL analysis, header inspection,
              sender reputation, XGBoost fallback, risk scoring.

Risk Score Formula (0–100):
  score = 0.35 * bert_prob + 0.25 * url_risk + 0.20 * header_risk
        + 0.10 * sender_rep + 0.10 * keyword_risk
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.models.bert_classifier import BERTClassifier, ClassificationResult
from app.services.explainer import ExplainabilityService, ExplainResult
from app.services.header_analyzer import HeaderAnalyzer, HeaderReport
from app.services.url_analyzer import URLAnalyzer, URLReport

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Phishing / Spam keyword lexicon (curated from Nazario + SpamAssassin)
# ─────────────────────────────────────────────────────────────────────────────
PHISHING_KEYWORDS = frozenset(
    [
        "verify your account", "confirm your identity", "click here immediately",
        "suspended account", "unusual activity", "update your billing",
        "password expired", "security alert", "account will be closed",
        "winner", "congratulations", "claim your prize", "you have been selected",
        "free gift", "limited time offer", "act now", "urgent action required",
        "dear customer", "dear account holder", "noreply", "do-not-reply",
        "ssn", "social security", "bank account number", "credit card number",
        "wire transfer", "western union", "bitcoin payment",
    ]
)

SPAM_PATTERNS = [
    re.compile(r"\b(buy|get)\s+now\b", re.I),
    re.compile(r"\bfree\s+(trial|access|money|offer)\b", re.I),
    re.compile(r"\b\d+%\s+off\b", re.I),
    re.compile(r"\bunsubscribe\b", re.I),
    re.compile(r"[A-Z]{5,}"),                              # EXCESSIVE CAPS
    re.compile(r"(!{2,}|\?{2,})"),                         # !! ??
    re.compile(r"\$[\d,]+(?:\.\d{2})?"),                   # money amounts
]


# ─────────────────────────────────────────────────────────────────────────────
# Result structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskBreakdown:
    bert_contribution: float      # 0–35
    url_contribution: float       # 0–25
    header_contribution: float    # 0–20
    sender_contribution: float    # 0–10
    keyword_contribution: float   # 0–10


@dataclass
class SecurityRecommendation:
    severity: str      # "critical" | "high" | "medium" | "low" | "info"
    category: str
    message: str
    action: str


@dataclass
class EmailAnalysisResult:
    # ── Identity
    email_hash: str
    subject: str
    sender: str

    # ── Classification
    classification: str        # LEGITIMATE | SPAM | PHISHING
    risk_score: int            # 0–100
    confidence: float          # 0.0–1.0
    risk_level: str            # "safe" | "low" | "medium" | "high" | "critical"

    # ── Sub-reports
    url_report: URLReport
    header_report: HeaderReport
    bert_result: ClassificationResult
    explain_result: ExplainResult

    # ── Detail
    risk_breakdown: RiskBreakdown
    flagged_keywords: List[str]
    spam_patterns_found: List[str]
    recommendations: List[SecurityRecommendation]

    # ── Performance
    total_latency_ms: float
    analysis_timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────
class EmailAnalyzer:
    """
    Async-ready orchestrator that runs all analysis modules in parallel
    and fuses their outputs into a single risk score.
    """

    def __init__(self, classifier: BERTClassifier):
        self.classifier = classifier
        self.url_analyzer = URLAnalyzer()
        self.header_analyzer = HeaderAnalyzer()
        self.explainer = ExplainabilityService(classifier)

    # ── Public API ────────────────────────────────────────────────────────────
    async def analyze(
        self,
        raw_text: str,
        headers: Optional[Dict[str, str]] = None,
        subject: str = "",
        sender: str = "",
    ) -> EmailAnalysisResult:
        t0 = time.perf_counter()

        # 1. Run IO-bound tasks concurrently
        bert_task = asyncio.to_thread(self._run_bert, raw_text)
        url_task = asyncio.to_thread(self._run_url_analysis, raw_text)
        header_task = asyncio.to_thread(self._run_header_analysis, headers or {}, sender)

        bert_result, url_report, header_report = await asyncio.gather(
            bert_task, url_task, header_task
        )

        # 2. Keyword / pattern scan
        flagged_kw, spam_pats = self._scan_keywords(raw_text + " " + subject)

        # 3. Explainability (uses cached BERT outputs)
        explain_result = await asyncio.to_thread(
            self.explainer.explain, raw_text, bert_result
        )

        # 4. Risk fusion
        risk_score, breakdown = self._fuse_risk(
            bert_result, url_report, header_report, flagged_kw, spam_pats
        )
        risk_level = self._risk_level(risk_score)
        classification = self._final_classification(bert_result, risk_score)

        # 5. Recommendations
        recommendations = self._generate_recommendations(
            risk_score, url_report, header_report, flagged_kw, bert_result
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return EmailAnalysisResult(
            email_hash=hashlib.sha256(raw_text.encode()).hexdigest()[:16],
            subject=subject,
            sender=sender,
            classification=classification,
            risk_score=risk_score,
            confidence=bert_result.confidence,
            risk_level=risk_level,
            url_report=url_report,
            header_report=header_report,
            bert_result=bert_result,
            explain_result=explain_result,
            risk_breakdown=breakdown,
            flagged_keywords=flagged_kw,
            spam_patterns_found=spam_pats,
            recommendations=recommendations,
            total_latency_ms=round(elapsed_ms, 2),
        )

    # ── Sub-tasks ─────────────────────────────────────────────────────────────
    def _run_bert(self, text: str) -> ClassificationResult:
        return self.classifier.predict(text[:1024])   # guard token limit

    def _run_url_analysis(self, text: str) -> URLReport:
        urls = self.url_analyzer.extract_urls(text)
        return self.url_analyzer.analyze_urls(urls)

    def _run_header_analysis(
        self, headers: Dict[str, str], sender: str
    ) -> HeaderReport:
        return self.header_analyzer.analyze(headers, sender)

    def _scan_keywords(
        self, text: str
    ) -> Tuple[List[str], List[str]]:
        lower = text.lower()
        flagged = [kw for kw in PHISHING_KEYWORDS if kw in lower]
        patterns_found = []
        for pat in SPAM_PATTERNS:
            matches = pat.findall(text)
            if matches:
                patterns_found.extend(str(m) for m in matches[:3])
        return flagged, patterns_found

    # ── Risk fusion ───────────────────────────────────────────────────────────
    def _fuse_risk(
        self,
        bert: ClassificationResult,
        urls: URLReport,
        headers: HeaderReport,
        keywords: List[str],
        patterns: List[str],
    ) -> Tuple[int, RiskBreakdown]:
        # BERT contribution (0–35)
        if bert.label == "LEGITIMATE":
            bert_contrib = bert.confidence * 5          # low contribution if clean
        else:
            bert_contrib = bert.confidence * 35

        # URL risk (0–25)
        url_contrib = min(urls.aggregate_risk * 25, 25.0)

        # Header risk (0–20)
        header_contrib = min(headers.risk_score * 20, 20.0)

        # Sender reputation (0–10)
        sender_contrib = max(0.0, 10.0 - headers.sender_trust_score * 10)

        # Keyword risk (0–10)
        kw_hits = len(keywords) + len(patterns)
        kw_contrib = min(kw_hits * 2.0, 10.0)

        raw = bert_contrib + url_contrib + header_contrib + sender_contrib + kw_contrib
        risk_score = int(min(round(raw), 100))

        return risk_score, RiskBreakdown(
            bert_contribution=round(bert_contrib, 2),
            url_contribution=round(url_contrib, 2),
            header_contribution=round(header_contrib, 2),
            sender_contribution=round(sender_contrib, 2),
            keyword_contribution=round(kw_contrib, 2),
        )

    @staticmethod
    def _risk_level(score: int) -> str:
        if score < 20:  return "safe"
        if score < 40:  return "low"
        if score < 60:  return "medium"
        if score < 80:  return "high"
        return "critical"

    @staticmethod
    def _final_classification(bert: ClassificationResult, risk: int) -> str:
        if risk < 30 and bert.label == "LEGITIMATE":
            return "LEGITIMATE"
        if bert.label == "PHISHING" or risk >= 70:
            return "PHISHING"
        if bert.label == "SPAM" or risk >= 40:
            return "SPAM"
        return "LEGITIMATE"

    # ── Recommendations ───────────────────────────────────────────────────────
    def _generate_recommendations(
        self,
        score: int,
        urls: URLReport,
        headers: HeaderReport,
        keywords: List[str],
        bert: ClassificationResult,
    ) -> List[SecurityRecommendation]:
        recs: List[SecurityRecommendation] = []

        if urls.malicious_url_count > 0:
            recs.append(SecurityRecommendation(
                severity="critical",
                category="URL Safety",
                message=f"{urls.malicious_url_count} URL(s) flagged as malicious by threat intel.",
                action="Do NOT click any links in this email.",
            ))

        if not headers.spf_pass:
            recs.append(SecurityRecommendation(
                severity="high",
                category="Email Authentication",
                message="SPF record check failed — sender domain could not be verified.",
                action="Treat this email as suspicious; verify sender via other channels.",
            ))

        if not headers.dkim_pass:
            recs.append(SecurityRecommendation(
                severity="high",
                category="Email Authentication",
                message="DKIM signature is missing or invalid.",
                action="The email may have been tampered with in transit.",
            ))

        if keywords:
            sample = ", ".join(f'"{k}"' for k in keywords[:4])
            recs.append(SecurityRecommendation(
                severity="medium",
                category="Content Analysis",
                message=f"Phishing language detected: {sample}.",
                action="Do not provide personal information or credentials.",
            ))

        if urls.newly_registered_domain_count > 0:
            recs.append(SecurityRecommendation(
                severity="medium",
                category="Domain Intelligence",
                message=f"{urls.newly_registered_domain_count} domain(s) registered within the past 30 days.",
                action="Newly registered domains are a common phishing indicator.",
            ))

        if bert.label == "PHISHING":
            recs.append(SecurityRecommendation(
                severity="critical",
                category="AI Classification",
                message=f"BERT model classified this as PHISHING with {bert.confidence:.0%} confidence.",
                action="Do not click links, reply, or open attachments.",
            ))

        if score < 20:
            recs.append(SecurityRecommendation(
                severity="info",
                category="Assessment",
                message="No significant threats detected.",
                action="Standard caution still applies when clicking external links.",
            ))

        return sorted(recs, key=lambda r: ["critical","high","medium","low","info"].index(r.severity))
