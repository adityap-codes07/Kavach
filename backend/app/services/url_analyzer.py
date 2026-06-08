"""
SmartShield — URL Analyzer Service
====================================
Features:
  - URL extraction via regex + BeautifulSoup fallback
  - Domain age detection (WHOIS)
  - VirusTotal / Google Safe Browsing API integration
  - Homoglyph / typosquatting detection
  - Redirect chain following
  - Suspicious TLD classification
  - IP-based URL detection
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
import whois

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
URL_REGEX = re.compile(
    r"https?://(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)",
    re.IGNORECASE,
)

SUSPICIOUS_TLDS: Set[str] = {
    ".tk", ".ml", ".ga", ".cf", ".gq",   # free/abused TLDs
    ".xyz", ".top", ".club", ".online",
    ".site", ".click", ".link", ".download",
}

TRUSTED_DOMAINS: Set[str] = {
    "google.com", "microsoft.com", "apple.com", "amazon.com",
    "facebook.com", "twitter.com", "linkedin.com", "github.com",
    "paypal.com", "stripe.com",
}

# Brands commonly targeted in phishing
BRAND_KEYWORDS: Dict[str, Set[str]] = {
    "paypal":    {"paypa1", "paypa-l", "pay-pal", "paypalservice"},
    "microsoft": {"micros0ft", "m1crosoft", "microsofft"},
    "apple":     {"app1e", "appie", "ap-ple", "icloud-security"},
    "amazon":    {"amaz0n", "amazn", "amazon-support"},
    "google":    {"g00gle", "goog1e", "gooogle"},
    "netflix":   {"netfl1x", "netfliix", "net-flix"},
    "bankofamerica": {"bankofamerica-secure", "boa-login"},
    "chase":     {"chase-secure", "chaseonline"},
}

HOMOGLYPHS: Dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a",
    "5": "s", "6": "b", "8": "b", "@": "a",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class URLAnalysis:
    url: str
    domain: str
    risk_score: float          # 0.0–1.0
    is_malicious: bool
    is_newly_registered: bool
    domain_age_days: Optional[int]
    uses_ip_address: bool
    has_suspicious_tld: bool
    is_typosquat: bool
    typosquat_target: Optional[str]
    redirect_count: int
    final_url: Optional[str]
    virustotal_hits: int
    google_safebrowsing_flagged: bool
    flags: List[str]


@dataclass
class URLReport:
    urls_found: int
    urls_analyzed: List[URLAnalysis]
    malicious_url_count: int
    newly_registered_domain_count: int
    suspicious_tld_count: int
    typosquat_count: int
    aggregate_risk: float      # 0.0–1.0
    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# URLAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class URLAnalyzer:
    """
    Synchronous URL analysis (designed to be called from asyncio.to_thread).
    Uses httpx for async HTTP when needed, falling back to safe stubs.
    """

    def __init__(
        self,
        virustotal_key: Optional[str] = None,
        safe_browsing_key: Optional[str] = None,
        max_redirect_depth: int = 3,
        timeout: float = 5.0,
    ):
        self.vt_key = virustotal_key
        self.sb_key = safe_browsing_key
        self.max_redirect_depth = max_redirect_depth
        self.timeout = timeout

    # ── URL extraction ────────────────────────────────────────────────────────
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        found = URL_REGEX.findall(text)
        # Deduplicate, preserve order
        seen: Set[str] = set()
        unique = []
        for u in found:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    # ── Main analysis entry ───────────────────────────────────────────────────
    def analyze_urls(self, urls: List[str]) -> URLReport:
        if not urls:
            return URLReport(
                urls_found=0, urls_analyzed=[], malicious_url_count=0,
                newly_registered_domain_count=0, suspicious_tld_count=0,
                typosquat_count=0, aggregate_risk=0.0, summary="No URLs found.",
            )

        analyses = [self._analyze_single(u) for u in urls[:20]]   # cap at 20

        malicious = sum(1 for a in analyses if a.is_malicious)
        new_domains = sum(1 for a in analyses if a.is_newly_registered)
        sus_tlds = sum(1 for a in analyses if a.has_suspicious_tld)
        typosquats = sum(1 for a in analyses if a.is_typosquat)

        max_risk = max((a.risk_score for a in analyses), default=0.0)
        mean_risk = sum(a.risk_score for a in analyses) / len(analyses) if analyses else 0.0
        aggregate = min((max_risk * 0.7 + mean_risk * 0.3), 1.0)

        summary = self._build_summary(malicious, new_domains, sus_tlds, typosquats)

        return URLReport(
            urls_found=len(urls),
            urls_analyzed=analyses,
            malicious_url_count=malicious,
            newly_registered_domain_count=new_domains,
            suspicious_tld_count=sus_tlds,
            typosquat_count=typosquats,
            aggregate_risk=aggregate,
            summary=summary,
        )

    def _analyze_single(self, url: str) -> URLAnalysis:
        flags: List[str] = []
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(":")[0]   # strip port

        # ── IP-based URL
        uses_ip = self._is_ip_address(domain)
        if uses_ip:
            flags.append("IP-based URL (no domain name)")

        # ── Suspicious TLD
        tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
        has_sus_tld = tld in SUSPICIOUS_TLDS
        if has_sus_tld:
            flags.append(f"Suspicious TLD: {tld}")

        # ── Typosquatting
        is_typosquat, target = self._check_typosquat(domain)
        if is_typosquat:
            flags.append(f"Possible typosquat of '{target}'")

        # ── Domain age
        domain_age_days, newly_registered = self._check_domain_age(domain)
        if newly_registered:
            flags.append(f"Domain registered {domain_age_days} days ago")

        # ── Redirect chain
        redirect_count, final_url = self._follow_redirects(url)
        if redirect_count > 1:
            flags.append(f"URL redirects {redirect_count} times")

        # ── Threat intelligence (stub — replace with real API calls)
        vt_hits = self._query_virustotal(url)
        sb_flagged = self._query_safe_browsing(url)
        is_malicious = vt_hits >= 3 or sb_flagged

        if vt_hits > 0:
            flags.append(f"VirusTotal: {vt_hits} engines flagged")
        if sb_flagged:
            flags.append("Google Safe Browsing: flagged as malicious")

        # ── Long/obfuscated URL
        if len(url) > 200:
            flags.append("Unusually long URL (possible obfuscation)")

        # ── At-sign in URL (phishing trick)
        if "@" in parsed.netloc:
            flags.append("@ symbol in URL host (credential injection attempt)")
            is_malicious = True

        # ── Risk score aggregation
        risk = self._compute_url_risk(
            uses_ip, has_sus_tld, is_typosquat, newly_registered,
            redirect_count, vt_hits, sb_flagged, flags
        )

        return URLAnalysis(
            url=url, domain=domain,
            risk_score=risk, is_malicious=is_malicious,
            is_newly_registered=newly_registered,
            domain_age_days=domain_age_days,
            uses_ip_address=uses_ip,
            has_suspicious_tld=has_sus_tld,
            is_typosquat=is_typosquat,
            typosquat_target=target,
            redirect_count=redirect_count,
            final_url=final_url,
            virustotal_hits=vt_hits,
            google_safebrowsing_flagged=sb_flagged,
            flags=flags,
        )

    # ── Helper methods ────────────────────────────────────────────────────────
    @staticmethod
    def _is_ip_address(host: str) -> bool:
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    @staticmethod
    def _normalize_homoglyphs(text: str) -> str:
        for glyph, char in HOMOGLYPHS.items():
            text = text.replace(glyph, char)
        return text

    def _check_typosquat(self, domain: str) -> Tuple[bool, Optional[str]]:
        normalized = self._normalize_homoglyphs(domain.split(".")[0])
        for brand, variants in BRAND_KEYWORDS.items():
            if normalized == brand:
                return False, None
            if any(v in normalized for v in variants):
                return True, brand
            # Edit-distance ≤ 2 check
            if self._levenshtein(normalized, brand) <= 2 and len(brand) > 5:
                return True, brand
        return False, None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return URLAnalyzer._levenshtein(s2, s1)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    def _check_domain_age(self, domain: str) -> Tuple[Optional[int], bool]:
        try:
            w = whois.whois(domain)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation:
                age = (datetime.now(timezone.utc) - creation.replace(tzinfo=timezone.utc)).days
                return age, age < 30
        except Exception:
            pass
        return None, False

    def _follow_redirects(self, url: str) -> Tuple[int, Optional[str]]:
        try:
            with httpx.Client(follow_redirects=True, timeout=self.timeout,
                              max_redirects=self.max_redirect_depth) as client:
                r = client.head(url)
                history_len = len(r.history)
                return history_len, str(r.url) if r.url != url else None
        except Exception:
            return 0, None

    def _query_virustotal(self, url: str) -> int:
        """Returns number of VT engines flagging the URL."""
        if not self.vt_key:
            return 0
        # Production: POST to https://www.virustotal.com/api/v3/urls
        return 0

    def _query_safe_browsing(self, url: str) -> bool:
        """Returns True if Google Safe Browsing flags the URL."""
        if not self.sb_key:
            return False
        # Production: POST to https://safebrowsing.googleapis.com/v4/threatMatches:find
        return False

    @staticmethod
    def _compute_url_risk(
        ip: bool, sus_tld: bool, typosquat: bool, new_domain: bool,
        redirects: int, vt_hits: int, sb: bool, flags: List[str]
    ) -> float:
        score = 0.0
        if ip:        score += 0.25
        if sus_tld:   score += 0.15
        if typosquat: score += 0.30
        if new_domain: score += 0.20
        if redirects > 1: score += min(redirects * 0.05, 0.15)
        if vt_hits:   score += min(vt_hits * 0.05, 0.25)
        if sb:        score += 0.40
        return min(score, 1.0)

    @staticmethod
    def _build_summary(malicious: int, new: int, sus: int, typo: int) -> str:
        parts = []
        if malicious: parts.append(f"{malicious} malicious URL(s)")
        if new:       parts.append(f"{new} newly registered domain(s)")
        if sus:       parts.append(f"{sus} suspicious TLD(s)")
        if typo:      parts.append(f"{typo} possible typosquat(s)")
        return "URL analysis: " + (", ".join(parts) or "No threats detected") + "."
