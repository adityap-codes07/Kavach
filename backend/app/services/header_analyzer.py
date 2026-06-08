"""
SmartShield — Header Analyzer Service
=======================================
Analyzes:
  - SPF (Sender Policy Framework)
  - DKIM (DomainKeys Identified Mail)
  - DMARC (Domain-based Message Authentication)
  - Return-Path / From mismatch
  - X-Mailer fingerprinting (spam tool detection)
  - Received chain hop analysis
  - Sender domain reputation (DNSBL lookup)
"""

from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import dns.resolver
import dns.exception

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Known spam tool X-Mailer fingerprints
# ─────────────────────────────────────────────────────────────────────────────
SUSPICIOUS_MAILERS = frozenset([
    "the bat!", "MailMate", "Mass Mailer", "Turbo Mailer",
    "AtMail", "GroupMail", "Dark Mailer", "Advanced Mass Sender",
    "SilverSender", "Extractor Pro",
])

# DNSBLs to query
DNSBL_SERVERS = [
    "zen.spamhaus.org",
    "b.barracudacentral.org",
    "bl.spamcop.net",
    "dnsbl.sorbs.net",
]

# RFC 5322 email address regex
EMAIL_REGEX = re.compile(r"[<\s]([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})[>\s,]?")
DOMAIN_REGEX = re.compile(r"@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HeaderReport:
    spf_pass: bool
    spf_result: str                 # "pass" | "fail" | "softfail" | "none" | "neutral"
    dkim_pass: bool
    dkim_result: str
    dmarc_pass: bool
    dmarc_policy: str               # "none" | "quarantine" | "reject"
    sender_trust_score: float       # 0.0 (untrusted) – 1.0 (fully trusted)
    risk_score: float               # 0.0 – 1.0
    from_domain: str
    return_path_domain: str
    from_return_path_mismatch: bool
    suspicious_mailer: Optional[str]
    hop_count: int
    dnsbl_listed: bool
    dnsbl_servers_hit: List[str]
    flags: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Header Analyzer
# ─────────────────────────────────────────────────────────────────────────────
class HeaderAnalyzer:
    """
    Parses raw email headers and performs authentication + reputation checks.
    All DNS calls are timeout-guarded to maintain low latency.
    """

    DNS_TIMEOUT = 3.0  # seconds

    def analyze(
        self,
        headers: Dict[str, str],
        sender: str = "",
    ) -> HeaderReport:
        flags: List[str] = []

        # ── Normalize headers (case-insensitive lookup)
        h = {k.lower(): v for k, v in headers.items()}

        # ── Extract domains
        from_domain = self._extract_domain(h.get("from", sender))
        return_path_domain = self._extract_domain(h.get("return-path", ""))

        # ── SPF
        spf_result, spf_pass = self._parse_spf(h)
        if not spf_pass:
            flags.append(f"SPF {spf_result.upper()} — sender domain unverified")

        # ── DKIM
        dkim_result, dkim_pass = self._parse_dkim(h)
        if not dkim_pass:
            flags.append(f"DKIM {dkim_result.upper()} — signature invalid or missing")

        # ── DMARC
        dmarc_pass, dmarc_policy = self._check_dmarc(from_domain)
        if not dmarc_pass:
            flags.append(f"DMARC {dmarc_policy.upper()} — policy not passing")

        # ── From / Return-Path mismatch
        mismatch = bool(
            from_domain and return_path_domain and from_domain != return_path_domain
        )
        if mismatch:
            flags.append(
                f"From/Return-Path mismatch: {from_domain} ≠ {return_path_domain}"
            )

        # ── X-Mailer fingerprinting
        mailer = h.get("x-mailer", h.get("user-agent", ""))
        sus_mailer = self._fingerprint_mailer(mailer)
        if sus_mailer:
            flags.append(f"Suspicious email client: {sus_mailer}")

        # ── Hop count analysis
        received_headers = [v for k, v in headers.items() if k.lower() == "received"]
        hop_count = len(received_headers)
        if hop_count > 10:
            flags.append(f"Unusually long routing chain: {hop_count} hops")

        # ── DNSBL lookup
        dnsbl_hits = self._query_dnsbl(from_domain)
        if dnsbl_hits:
            flags.append(f"Domain listed in {len(dnsbl_hits)} DNSBL(s): {', '.join(dnsbl_hits)}")

        # ── Authentication-Results header cross-check
        auth_results = h.get("authentication-results", "")
        if auth_results:
            if "spf=fail" in auth_results.lower():
                flags.append("Authentication-Results header confirms SPF failure")
            if "dkim=fail" in auth_results.lower():
                flags.append("Authentication-Results header confirms DKIM failure")

        # ── Compute scores
        trust_score = self._compute_trust_score(
            spf_pass, dkim_pass, dmarc_pass, mismatch, bool(dnsbl_hits), bool(sus_mailer)
        )
        risk_score = 1.0 - trust_score

        return HeaderReport(
            spf_pass=spf_pass,
            spf_result=spf_result,
            dkim_pass=dkim_pass,
            dkim_result=dkim_result,
            dmarc_pass=dmarc_pass,
            dmarc_policy=dmarc_policy,
            sender_trust_score=round(trust_score, 3),
            risk_score=round(risk_score, 3),
            from_domain=from_domain,
            return_path_domain=return_path_domain,
            from_return_path_mismatch=mismatch,
            suspicious_mailer=sus_mailer,
            hop_count=hop_count,
            dnsbl_listed=bool(dnsbl_hits),
            dnsbl_servers_hit=dnsbl_hits,
            flags=flags,
        )

    # ── SPF parsing ───────────────────────────────────────────────────────────
    @staticmethod
    def _parse_spf(headers: Dict[str, str]) -> Tuple[str, bool]:
        """Parse Received-SPF and Authentication-Results for SPF verdict."""
        spf_header = headers.get("received-spf", "")
        auth_header = headers.get("authentication-results", "")

        combined = (spf_header + " " + auth_header).lower()
        if "spf=pass" in combined or combined.startswith("pass"):
            return "pass", True
        if "spf=fail" in combined or combined.startswith("fail"):
            return "fail", False
        if "softfail" in combined:
            return "softfail", False
        if "neutral" in combined:
            return "neutral", True   # neutral is not a failure per RFC 7208
        if not spf_header and not ("spf=" in auth_header.lower()):
            return "none", False     # no SPF record = unverified
        return "unknown", False

    # ── DKIM parsing ──────────────────────────────────────────────────────────
    @staticmethod
    def _parse_dkim(headers: Dict[str, str]) -> Tuple[str, bool]:
        """Parse DKIM-Signature header and Authentication-Results."""
        sig = headers.get("dkim-signature", "")
        auth = headers.get("authentication-results", "").lower()

        if "dkim=pass" in auth:
            return "pass", True
        if "dkim=fail" in auth:
            return "fail", False

        if not sig:
            return "none", False

        # Basic structural validation of DKIM-Signature
        required_tags = ["v=1", "a=rsa-sha", "d=", "s=", "b="]
        if all(tag.lower() in sig.lower() for tag in required_tags):
            return "present", True

        return "malformed", False

    # ── DMARC DNS lookup ──────────────────────────────────────────────────────
    def _check_dmarc(self, domain: str) -> Tuple[bool, str]:
        """Query _dmarc.{domain} TXT record and parse policy."""
        if not domain:
            return False, "none"
        try:
            resolver = dns.resolver.Resolver()
            resolver.lifetime = self.DNS_TIMEOUT
            answers = resolver.resolve(f"_dmarc.{domain}", "TXT")
            for rdata in answers:
                txt = b"".join(rdata.strings).decode("utf-8", errors="replace").lower()
                if "v=dmarc1" in txt:
                    if "p=reject" in txt:
                        return True, "reject"
                    if "p=quarantine" in txt:
                        return True, "quarantine"
                    if "p=none" in txt:
                        return False, "none"   # DMARC present but policy=none
        except (dns.exception.DNSException, Exception):
            pass
        return False, "none"

    # ── DNSBL lookup ──────────────────────────────────────────────────────────
    def _query_dnsbl(self, domain: str) -> List[str]:
        """Check if sending domain's IP is listed in any DNSBL."""
        if not domain:
            return []
        try:
            ip = socket.gethostbyname(domain)
        except socket.gaierror:
            return []

        # Reverse the IP for DNSBL query format
        reversed_ip = ".".join(reversed(ip.split(".")))
        hits: List[str] = []

        for dnsbl in DNSBL_SERVERS:
            try:
                resolver = dns.resolver.Resolver()
                resolver.lifetime = self.DNS_TIMEOUT
                query = f"{reversed_ip}.{dnsbl}"
                resolver.resolve(query, "A")
                hits.append(dnsbl)
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                pass  # Not listed
            except Exception:
                pass  # DNS error — skip silently

        return hits

    # ── X-Mailer fingerprinting ───────────────────────────────────────────────
    @staticmethod
    def _fingerprint_mailer(mailer_str: str) -> Optional[str]:
        if not mailer_str:
            return None
        for sus in SUSPICIOUS_MAILERS:
            if sus.lower() in mailer_str.lower():
                return sus
        return None

    # ── Domain extraction ─────────────────────────────────────────────────────
    @staticmethod
    def _extract_domain(header_value: str) -> str:
        if not header_value:
            return ""
        match = DOMAIN_REGEX.search(header_value)
        return match.group(1).lower() if match else ""

    # ── Trust score ───────────────────────────────────────────────────────────
    @staticmethod
    def _compute_trust_score(
        spf: bool,
        dkim: bool,
        dmarc: bool,
        mismatch: bool,
        dnsbl: bool,
        sus_mailer: bool,
    ) -> float:
        score = 1.0
        if not spf:     score -= 0.25
        if not dkim:    score -= 0.25
        if not dmarc:   score -= 0.15
        if mismatch:    score -= 0.20
        if dnsbl:       score -= 0.30
        if sus_mailer:  score -= 0.10
        return max(round(score, 3), 0.0)
