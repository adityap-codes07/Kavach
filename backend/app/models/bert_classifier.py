"""
Kavach — BERT-based Email Classifier
==========================================
Supports: bert-base-uncased, distilbert-base-uncased, roberta-base
Fine-tuned on: Enron + SpamAssassin + CEAS 2008 + Nazario Phishing Corpus

Architecture:
  [CLS] token → Linear(768, 256) → GELU → Dropout(0.3) → Linear(256, 3)
  Classes: 0=LEGITIMATE, 1=SPAM, 2=PHISHING
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────────────────────────────────────
LABEL2ID: Dict[str, int] = {"LEGITIMATE": 0, "SPAM": 1, "PHISHING": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

SUPPORTED_MODELS = {
    "bert":       "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta":    "roberta-base",
}


# ─────────────────────────────────────────────────────────────────────────────
# Custom classification head with attention pooling
# ─────────────────────────────────────────────────────────────────────────────
class AttentionPoolingHead(nn.Module):
    """
    Weighted mean-pool over all token representations using a learned
    attention vector.  This outperforms [CLS]-only pooling on short texts.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T, H)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        # Compute attention weights
        scores = self.attention(hidden_states).squeeze(-1)   # (B, T)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)    # (B, T, 1)
        pooled = (hidden_states * weights).sum(dim=1)        # (B, H)
        return self.classifier(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    token_count: int
    attention_weights: Optional[List[float]] = None    # for visualization
    top_tokens: Optional[List[Tuple[str, float]]] = field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────────────────
class BERTClassifier:
    """
    Wrapper around a fine-tuned transformer for 3-class email classification.

    Usage:
        clf = BERTClassifier(model_name="bert")
        clf.train(train_dataset, val_dataset)
        result = clf.predict("You won a prize! Click here now.")
    """

    def __init__(
        self,
        model_name: str = "bert",
        num_labels: int = 3,
        max_length: int = 512,
        device: Optional[str] = None,
        use_attention_pooling: bool = True,
    ):
        self.model_variant = model_name
        self.base_model_name = SUPPORTED_MODELS.get(model_name, model_name)
        self.num_labels = num_labels
        self.max_length = max_length
        self.use_attention_pooling = use_attention_pooling

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("BERTClassifier using device: %s", self.device)

        self._load_tokenizer()
        self._load_model()

    # ── Private helpers ───────────────────────────────────────────────────────
    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            use_fast=True,
        )

    def _load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=self.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            output_attentions=True,
            output_hidden_states=True,
        )
        if self.use_attention_pooling:
            hidden_size = self.model.config.hidden_size
            self._custom_head = AttentionPoolingHead(
                hidden_size, self.num_labels
            ).to(self.device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _tokenize(self, text: Union[str, List[str]]) -> dict:
        if isinstance(text, (list, tuple, np.ndarray)):
            text = [str(t) for t in text]
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """Single-sample inference with attention extraction."""
        t0 = time.perf_counter()

        encoding = self._tokenize(text)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if getattr(self, "use_attention_pooling", False) and hasattr(self, "_custom_head"):
            hidden = outputs.hidden_states[-1]          # last layer (B, T, H)
            logits = self._custom_head(hidden, attention_mask)
        else:
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        label_id = int(np.argmax(probs))

        # Extract [CLS]-averaged attention for token highlighting
        attn = outputs.attentions[-1]                   # (B, heads, T, T)
        cls_attn = attn[0, :, 0, :].mean(dim=0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist()
        )
        token_attn_pairs = list(zip(tokens, cls_attn.tolist()))
        top_tokens = sorted(token_attn_pairs, key=lambda x: x[1], reverse=True)[
            :10
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ClassificationResult(
            label=ID2LABEL[label_id],
            label_id=label_id,
            confidence=float(probs[label_id]),
            probabilities={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            inference_time_ms=round(elapsed_ms, 2),
            token_count=int(attention_mask.sum().item()),
            attention_weights=cls_attn.tolist(),
            top_tokens=top_tokens,
        )

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[ClassificationResult]:
        """Batched inference for bulk analysis."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoding = self._tokenize(batch)
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if not self.use_attention_pooling else \
                self._custom_head(outputs.hidden_states[-1], attention_mask)
            probs_batch = F.softmax(logits, dim=-1).cpu().numpy()

            for probs in probs_batch:
                label_id = int(np.argmax(probs))
                results.append(
                    ClassificationResult(
                        label=ID2LABEL[label_id],
                        label_id=label_id,
                        confidence=float(probs[label_id]),
                        probabilities={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
                        inference_time_ms=0.0,
                        token_count=0,
                    )
                )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, save_dir: Union[str, Path]):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        if self.use_attention_pooling:
            torch.save(
                self._custom_head.state_dict(),
                save_dir / "attention_head.pt",
            )
        logger.info("Model saved to %s", save_dir)

    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> "BERTClassifier":
        load_dir = Path(load_dir)
        instance = cls.__new__(cls)
        instance.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        instance.max_length = 512
        instance.use_attention_pooling = True

        instance.tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            str(load_dir),
            output_attentions=True,
            output_hidden_states=True,
        ).to(instance.device)
        instance.model.eval()

        head_path = load_dir / "attention_head.pt"
        if head_path.exists():
            hidden_size = instance.model.config.hidden_size
            instance._custom_head = AttentionPoolingHead(hidden_size, 3).to(
                instance.device
            )
            instance._custom_head.load_state_dict(
                torch.load(head_path, map_location=instance.device)
            )
            instance._custom_head.eval()
            instance.use_attention_pooling = True
        else:
            instance.use_attention_pooling = False

        logger.info("Model loaded from %s", load_dir)
        return instance
