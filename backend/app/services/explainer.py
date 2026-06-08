"""
SmartShield — Explainability Service
======================================
Methods:
  1. SHAP (KernelExplainer on BERT pipeline)
  2. LIME (TextExplainer)
  3. Attention rollout visualization
  4. Integrated Gradients (future)

Produces human-readable explanations for any BERT classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenImportance:
    token: str
    importance: float     # negative = against flag; positive = toward flag
    layer: str            # "shap" | "lime" | "attention"


@dataclass
class ExplainResult:
    method: str
    predicted_label: str
    confidence: float
    token_importances: List[TokenImportance]
    top_positive_tokens: List[Tuple[str, float]]   # words pushing toward flag
    top_negative_tokens: List[Tuple[str, float]]   # words suggesting clean
    natural_language_summary: str
    lime_text_segments: Optional[List[Tuple[str, float]]] = None
    shap_base_value: Optional[float] = None


class ExplainabilityService:
    """
    Provides post-hoc explanations for BERT email classifications.
    Falls back gracefully if optional SHAP/LIME libraries are unavailable.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self._shap_available = self._try_import_shap()
        self._lime_available = self._try_import_lime()

    @staticmethod
    def _try_import_shap() -> bool:
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            logger.warning("SHAP not available; falling back to attention-only explanations.")
            return False

    @staticmethod
    def _try_import_lime() -> bool:
        try:
            from lime.lime_text import LimeTextExplainer  # noqa: F401
            return True
        except ImportError:
            logger.warning("LIME not available; falling back to attention-only explanations.")
            return False

    # ── Public entry point ────────────────────────────────────────────────────
    def explain(self, text: str, bert_result) -> ExplainResult:
        """
        Selects best available explanation method:
        SHAP > LIME > Attention rollout.
        """
        if self._shap_available:
            return self._explain_shap(text, bert_result)
        elif self._lime_available:
            return self._explain_lime(text, bert_result)
        else:
            return self._explain_attention(text, bert_result)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    def _explain_shap(self, text: str, bert_result) -> ExplainResult:
        import shap

        try:
            pipeline_fn = self._make_pipeline_fn()
            masker = shap.maskers.Text(r"\W+")
            explainer = shap.Explainer(pipeline_fn, masker)
            shap_values = explainer([text])

            # shap_values.values shape: (1, num_tokens, num_classes)
            label_id = bert_result.label_id
            values = shap_values.values[0, :, label_id]
            tokens = shap_values.data[0]

            importances = [
                TokenImportance(token=t, importance=float(v), layer="shap")
                for t, v in zip(tokens, values)
                if t.strip()
            ]

            base_value = float(shap_values.base_values[0, label_id])
            summary = self._summarise(importances, bert_result)

            return ExplainResult(
                method="SHAP (KernelExplainer)",
                predicted_label=bert_result.label,
                confidence=bert_result.confidence,
                token_importances=importances,
                top_positive_tokens=self._top_n(importances, positive=True),
                top_negative_tokens=self._top_n(importances, positive=False),
                natural_language_summary=summary,
                shap_base_value=base_value,
            )
        except Exception as e:
            logger.exception("SHAP explanation failed: %s", e)
            return self._explain_attention(text, bert_result)

    # ── LIME ──────────────────────────────────────────────────────────────────
    def _explain_lime(self, text: str, bert_result) -> ExplainResult:
        from lime.lime_text import LimeTextExplainer

        try:
            explainer = LimeTextExplainer(
                class_names=["LEGITIMATE", "SPAM", "PHISHING"],
                random_state=42,
            )
            pipeline_fn = self._make_pipeline_fn()

            exp = explainer.explain_instance(
                text,
                pipeline_fn,
                num_features=15,
                num_samples=500,
                labels=[bert_result.label_id],
            )
            raw = exp.as_list(label=bert_result.label_id)
            importances = [
                TokenImportance(token=tok, importance=float(val), layer="lime")
                for tok, val in raw
            ]
            segments = [(tok, float(val)) for tok, val in raw]
            summary = self._summarise(importances, bert_result)

            return ExplainResult(
                method="LIME (TextExplainer)",
                predicted_label=bert_result.label,
                confidence=bert_result.confidence,
                token_importances=importances,
                top_positive_tokens=self._top_n(importances, positive=True),
                top_negative_tokens=self._top_n(importances, positive=False),
                natural_language_summary=summary,
                lime_text_segments=segments,
            )
        except Exception as e:
            logger.exception("LIME explanation failed: %s", e)
            return self._explain_attention(text, bert_result)

    # ── Attention rollout ─────────────────────────────────────────────────────
    def _explain_attention(self, text: str, bert_result) -> ExplainResult:
        """
        Uses [CLS] token attention weights from the last transformer layer
        as a lightweight proxy for token importance.
        Reference: Abnar & Zuidema (2020) — Quantifying Attention Flow.
        """
        if bert_result.top_tokens:
            importances = [
                TokenImportance(token=tok, importance=float(attn), layer="attention")
                for tok, attn in bert_result.top_tokens
                if not tok.startswith("[") and tok != "##"
            ]
        else:
            importances = []

        summary = self._summarise(importances, bert_result)

        return ExplainResult(
            method="Attention Rollout (CLS-weighted)",
            predicted_label=bert_result.label,
            confidence=bert_result.confidence,
            token_importances=importances,
            top_positive_tokens=self._top_n(importances, positive=True),
            top_negative_tokens=self._top_n(importances, positive=False),
            natural_language_summary=summary,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_pipeline_fn(self):
        """Returns a function (texts: List[str]) -> np.ndarray of probabilities."""
        def _predict_proba(texts: List[str]) -> np.ndarray:
            results = self.classifier.predict_batch(texts)
            probs = np.array([
                [r.probabilities.get("LEGITIMATE", 0),
                 r.probabilities.get("SPAM", 0),
                 r.probabilities.get("PHISHING", 0)]
                for r in results
            ])
            return probs
        return _predict_proba

    @staticmethod
    def _top_n(
        importances: List[TokenImportance],
        positive: bool,
        n: int = 5,
    ) -> List[Tuple[str, float]]:
        filtered = [i for i in importances
                    if (i.importance > 0) == positive and not i.token.startswith("##")]
        filtered.sort(key=lambda x: abs(x.importance), reverse=True)
        return [(i.token, round(i.importance, 4)) for i in filtered[:n]]

    @staticmethod
    def _summarise(
        importances: List[TokenImportance],
        bert_result,
    ) -> str:
        top_pos = sorted(
            [i for i in importances if i.importance > 0],
            key=lambda x: x.importance,
            reverse=True,
        )[:3]
        labels = {"LEGITIMATE": "legitimate", "SPAM": "spam", "PHISHING": "phishing"}
        label_text = labels.get(bert_result.label, bert_result.label.lower())

        if not top_pos:
            return (
                f"The model classified this email as {label_text} "
                f"(confidence: {bert_result.confidence:.0%}). "
                "No single token dominated the prediction."
            )

        token_list = ", ".join(f'"{t.token}"' for t in top_pos)
        return (
            f"The model classified this email as {label_text} "
            f"(confidence: {bert_result.confidence:.0%}). "
            f"Key influencing tokens: {token_list}. "
            "These words most strongly contributed to the classification decision."
        )
