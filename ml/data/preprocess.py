"""
SmartShield — Data Preprocessing Pipeline
==========================================
Processes four public email security datasets into a unified JSONL format
ready for BERT fine-tuning.

Datasets supported:
  1. Enron Email Corpus          → data/raw/enron/
  2. SpamAssassin Public Corpus  → data/raw/spamassassin/
  3. CEAS 2008 Challenge         → data/raw/ceas2008/
  4. Nazario Phishing Corpus     → data/raw/nazario/

Output:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
  data/processed/stats.json

Label schema:
  {"text": "...", "label": "SPAM" | "LEGITIMATE" | "PHISHING"}

Usage:
  python ml/data/preprocess.py --raw_dir data/raw --output_dir data/processed
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import logging
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_TEXT_CHARS = 4096    # cap for training (BERT context window)
MIN_TEXT_CHARS = 20      # discard very short emails
HTML_TAG_RE   = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE        = re.compile(r"https?://\S+")
BASE64_RE     = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
NON_ASCII_RE  = re.compile(r"[^\x00-\x7F]+")


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(raw: str, mask_urls: bool = True) -> str:
    """
    Clean raw email body text for BERT tokenization.
    Steps:
      1. Decode Unicode escape sequences
      2. Remove HTML tags
      3. Normalize whitespace
      4. Optionally mask URLs with [URL] token
      5. Remove base64 blobs
      6. Truncate to MAX_TEXT_CHARS
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", raw)

    # Remove HTML
    text = HTML_TAG_RE.sub(" ", text)

    # Remove base64 payloads
    text = BASE64_RE.sub(" [ATTACHMENT] ", text)

    # Mask URLs
    if mask_urls:
        text = URL_RE.sub(" [URL] ", text)

    # Collapse whitespace
    text = WHITESPACE_RE.sub(" ", text).strip()

    # Truncate
    return text[:MAX_TEXT_CHARS]


def extract_email_text(
    msg: email.message.Message,
    prefer_html: bool = False,
) -> str:
    """Extract text payload from a parsed email.message.Message."""
    parts_text: List[str] = []
    parts_html: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                decoded = payload.decode(charset, errors="replace")
                if ct == "text/plain":
                    parts_text.append(decoded)
                elif ct == "text/html":
                    parts_html.append(decoded)
            except Exception:
                continue
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                parts_text.append(payload.decode(charset, errors="replace"))
        except Exception:
            pass

    preferred = parts_text if not prefer_html else parts_html
    fallback  = parts_html if not prefer_html else parts_text

    combined = " ".join(preferred or fallback)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_enron(raw_dir: Path) -> Iterator[Dict]:
    """
    Enron dataset layout:
      enron/ham/*.txt   → LEGITIMATE
      enron/spam/*.txt  → SPAM
    """
    label_map = {"ham": "LEGITIMATE", "spam": "SPAM"}
    for split_name, label in label_map.items():
        split_dir = raw_dir / "enron" / split_name
        if not split_dir.exists():
            logger.warning("Enron %s directory not found: %s", split_name, split_dir)
            continue
        for fpath in tqdm(list(split_dir.glob("*.txt")), desc=f"Enron {split_name}"):
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                # Enron files contain raw RFC 2822 messages
                try:
                    msg = email.message_from_string(raw)
                    body = extract_email_text(msg)
                    subject = msg.get("Subject", "")
                except Exception:
                    body = raw
                    subject = ""
                text = clean_text(subject + " " + body)
                if len(text) >= MIN_TEXT_CHARS:
                    yield {"text": text, "label": label, "source": "enron"}
            except Exception as e:
                logger.debug("Failed to load %s: %s", fpath, e)


def load_spamassassin(raw_dir: Path) -> Iterator[Dict]:
    """
    SpamAssassin layout:
      spamassassin/easy_ham/      → LEGITIMATE
      spamassassin/easy_ham_2/    → LEGITIMATE
      spamassassin/hard_ham/      → LEGITIMATE
      spamassassin/spam/          → SPAM
      spamassassin/spam_2/        → SPAM
    """
    legitimate_dirs = ["easy_ham", "easy_ham_2", "hard_ham"]
    spam_dirs       = ["spam", "spam_2"]

    def _load_dir(d: Path, label: str):
        if not d.exists():
            logger.warning("SpamAssassin dir not found: %s", d)
            return
        for fpath in tqdm(list(d.glob("*")), desc=f"SA {d.name}"):
            if fpath.is_file():
                try:
                    raw = fpath.read_bytes()
                    msg = email.message_from_bytes(raw)
                    body = extract_email_text(msg)
                    subject = msg.get("Subject", "")
                    text = clean_text(subject + " " + body)
                    if len(text) >= MIN_TEXT_CHARS:
                        yield {"text": text, "label": label, "source": "spamassassin"}
                except Exception:
                    pass

    for d in legitimate_dirs:
        yield from _load_dir(raw_dir / "spamassassin" / d, "LEGITIMATE")
    for d in spam_dirs:
        yield from _load_dir(raw_dir / "spamassassin" / d, "SPAM")


def load_ceas2008(raw_dir: Path) -> Iterator[Dict]:
    """
    CEAS 2008 challenge format:
      ceas2008/ceas2008.csv with columns: label,subject,body
      label: 0=ham, 1=spam
    """
    csv_path = raw_dir / "ceas2008" / "ceas2008.csv"
    if not csv_path.exists():
        logger.warning("CEAS 2008 CSV not found: %s", csv_path)
        return

    import csv
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="CEAS 2008"):
            try:
                label_raw = int(row.get("label", row.get("spam", 0)))
                label = "SPAM" if label_raw == 1 else "LEGITIMATE"
                subject = row.get("subject", "")
                body = row.get("body", row.get("text", ""))
                text = clean_text(subject + " " + body)
                if len(text) >= MIN_TEXT_CHARS:
                    yield {"text": text, "label": label, "source": "ceas2008"}
            except Exception:
                continue


def load_nazario(raw_dir: Path) -> Iterator[Dict]:
    """
    Nazario Phishing Corpus layout:
      nazario/phishing/*.eml  OR  nazario/*.mbox
    All files are phishing emails → PHISHING label.
    """
    nazario_dir = raw_dir / "nazario"
    if not nazario_dir.exists():
        logger.warning("Nazario dir not found: %s", nazario_dir)
        return

    # Support both .eml files and flat mbox-style files
    candidates = list(nazario_dir.rglob("*.eml")) + list(nazario_dir.rglob("*.txt"))
    for fpath in tqdm(candidates, desc="Nazario phishing"):
        try:
            raw = fpath.read_bytes()
            msg = email.message_from_bytes(raw)
            body = extract_email_text(msg)
            subject = msg.get("Subject", "")
            text = clean_text(subject + " " + body)
            if len(text) >= MIN_TEXT_CHARS:
                yield {"text": text, "label": "PHISHING", "source": "nazario"}
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def deduplicate(samples: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for s in samples:
        h = hashlib.md5(s["text"][:256].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(s)
    logger.info("Deduplication: %d → %d samples", len(samples), len(unique))
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (for minority phishing class)
# ─────────────────────────────────────────────────────────────────────────────
def augment_minority(
    samples: List[Dict],
    target_label: str = "PHISHING",
    target_ratio: float = 0.12,
) -> List[Dict]:
    """
    Simple synonym augmentation to boost phishing class to target_ratio.
    In production, use nlpaug or back-translation.
    """
    from random import Random
    rng = Random(42)
    total = len(samples)
    n_target = sum(1 for s in samples if s["label"] == target_label)
    desired = int(total * target_ratio)

    if n_target >= desired:
        return samples

    minority = [s for s in samples if s["label"] == target_label]
    needed = desired - n_target
    augmented = []
    for _ in range(needed):
        original = rng.choice(minority)
        words = original["text"].split()
        # Simple dropout augmentation
        new_words = [w for w in words if rng.random() > 0.05]
        if len(new_words) < 15:
            new_words = words
        augmented.append({
            "text": " ".join(new_words),
            "label": target_label,
            "source": original["source"] + "_aug",
        })
    logger.info("Augmented %s: %d → %d samples", target_label, n_target, n_target + needed)
    return samples + augmented


# ─────────────────────────────────────────────────────────────────────────────
# Split and save
# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(
    samples: List[Dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    labels = [s["label"] for s in samples]
    indices = np.arange(len(samples))

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    train_val_idx, test_idx = next(sss1.split(indices, labels))

    # Second split: val from train_val
    train_val_labels = [labels[i] for i in train_val_idx]
    val_frac = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=random_state
    )
    train_rel_idx, val_rel_idx = next(sss2.split(train_val_idx, train_val_labels))

    train = [samples[train_val_idx[i]] for i in train_rel_idx]
    val   = [samples[train_val_idx[i]] for i in val_rel_idx]
    test  = [samples[i] for i in test_idx]
    return train, val, test


def save_jsonl(samples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s["text"], "label": s["label"]}) + "\n")
    logger.info("Saved %d samples → %s", len(samples), path)


def save_stats(train, val, test, output_dir: Path):
    def dist(split): return dict(Counter(s["label"] for s in split))
    stats = {
        "total": len(train) + len(val) + len(test),
        "train": {"count": len(train), "distribution": dist(train)},
        "val":   {"count": len(val),   "distribution": dist(val)},
        "test":  {"count": len(test),  "distribution": dist(test)},
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Dataset stats:\n%s", json.dumps(stats, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(raw_dir: Path, output_dir: Path):
    logger.info("Starting SmartShield data preprocessing pipeline…")

    # Load all datasets
    all_samples: List[Dict] = []
    for loader in [load_enron, load_spamassassin, load_ceas2008, load_nazario]:
        try:
            batch = list(loader(raw_dir))
            logger.info("%s loaded: %d samples", loader.__name__, len(batch))
            all_samples.extend(batch)
        except Exception as e:
            logger.error("Loader %s failed: %s", loader.__name__, e)

    logger.info("Total raw samples: %d", len(all_samples))

    # Deduplicate
    all_samples = deduplicate(all_samples)

    # Augment minority phishing class
    all_samples = augment_minority(all_samples, "PHISHING", target_ratio=0.12)

    # Shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(all_samples)  # type: ignore

    # Stratified split
    train, val, test = stratified_split(all_samples)

    # Save
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val,   output_dir / "val.jsonl")
    save_jsonl(test,  output_dir / "test.jsonl")
    save_stats(train, val, test, output_dir)

    logger.info("✅ Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartShield data preprocessor")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    run_pipeline(Path(args.raw_dir), Path(args.output_dir))
