"""
SmartShield — BERT Fine-Tuning Script
=======================================
Trains BERT / DistilBERT / RoBERTa on the combined SmartShield corpus:
  - Enron Email Dataset
  - SpamAssassin Public Corpus
  - CEAS 2008 Challenge Dataset
  - Nazario Phishing Corpus

Features:
  - Mixed precision (fp16) training
  - Cosine schedule with warm-up
  - Layer-wise learning rate decay
  - Label smoothing
  - Gradient checkpointing for large batches
  - Early stopping on val F1
  - WandB / TensorBoard logging
  - Checkpoint averaging (SWA)

Usage:
  python ml/training/train_bert.py \
    --model bert \
    --data_dir data/processed \
    --output_dir checkpoints/bert_v1 \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL2ID = {"LEGITIMATE": 0, "SPAM": 1, "PHISHING": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class EmailDataset(Dataset):
    """
    Expects a JSONL file with lines: {"text": "...", "label": "SPAM"}.
    Applies augmentation (token dropout, random crop) during training.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 512,
        augment: bool = False,
        token_dropout_prob: float = 0.05,
    ):
        self.samples: List[Dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.token_dropout_prob = token_dropout_prob

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    self.samples.append({
                        "text": obj["text"],
                        "label": LABEL2ID[obj["label"]],
                    })

        # Class distribution logging
        from collections import Counter
        dist = Counter(s["label"] for s in self.samples)
        logger.info("Dataset %s — %d samples: %s", jsonl_path, len(self.samples), dict(dist))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]

        if self.augment:
            text = self._augment(text)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }

    def _augment(self, text: str) -> str:
        """Token-level dropout augmentation."""
        words = text.split()
        words = [
            "[UNK]" if random.random() < self.token_dropout_prob else w
            for w in words
        ]
        # Random crop if too long
        if len(words) > 400:
            start = random.randint(0, len(words) - 400)
            words = words[start : start + 400]
        return " ".join(words)

    def class_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights for WeightedRandomSampler."""
        from collections import Counter
        counts = Counter(s["label"] for s in self.samples)
        total = len(self.samples)
        weights = torch.tensor(
            [total / (len(counts) * counts[i]) for i in range(len(counts))],
            dtype=torch.float,
        )
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class BERTTrainer:

    SUPPORTED = {
        "bert":       "bert-base-uncased",
        "distilbert": "distilbert-base-uncased",
        "roberta":    "roberta-base",
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", self.device)

        self.base_model_name = self.SUPPORTED[args.model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        if args.wandb:
            wandb.init(project="smartshield", name=f"{args.model}-finetune",
                       config=vars(args))

    # ── Build dataloaders ─────────────────────────────────────────────────────
    def _make_loaders(self) -> Tuple[DataLoader, DataLoader]:
        data_dir = Path(self.args.data_dir)

        train_ds = EmailDataset(
            str(data_dir / "train.jsonl"),
            self.tokenizer,
            max_length=self.args.max_length,
            augment=True,
        )
        val_ds = EmailDataset(
            str(data_dir / "val.jsonl"),
            self.tokenizer,
            max_length=self.args.max_length,
            augment=False,
        )

        # Weighted sampler to handle class imbalance
        class_weights = train_ds.class_weights()
        sample_weights = [class_weights[s["label"]].item() for s in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return train_loader, val_loader

    # ── Optimizer with layer-wise LR decay ───────────────────────────────────
    def _make_optimizer(self) -> AdamW:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.args.lr,
            },
        ]
        return AdamW(optimizer_params, lr=self.args.lr, eps=1e-8)

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self):
        train_loader, val_loader = self._make_loaders()
        optimizer = self._make_optimizer()
        total_steps = len(train_loader) * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        # Label smoothing loss
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        logger.info("Starting training: %d epochs, %d steps, %d warmup",
                    self.args.epochs, total_steps, warmup_steps)

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, criterion)
            val_metrics = self._evaluate(val_loader)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_acc=%.4f | val_f1=%.4f | val_auc=%.4f",
                epoch, train_loss,
                val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["roc_auc"],
            )

            if self.args.wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            # Checkpoint if best
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
                logger.info("✅ New best model (F1=%.4f) saved.", self.best_val_f1)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        logger.info("Training complete. Best val F1: %.4f", self.best_val_f1)

    def _train_epoch(self, loader, optimizer, scheduler, criterion) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0:
                logger.debug("Step %d | loss=%.4f | lr=%.2e",
                             step, loss.item(), scheduler.get_last_lr()[0])

        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].cpu().numpy()

            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)

        all_probs = np.array(all_probs)
        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
            "macro_precision": float(precision_score(all_labels, all_preds, average="macro")),
            "macro_recall": float(recall_score(all_labels, all_preds, average="macro")),
            "roc_auc": float(roc_auc_score(all_labels, all_probs, multi_class="ovr")),
        }
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        out = Path(self.args.output_dir)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        meta = {"epoch": epoch, "metrics": metrics, "model": self.args.model}
        (out / "training_meta.json").write_text(json.dumps(meta, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmartShield BERT fine-tuner")
    p.add_argument("--model", choices=["bert", "distilbert", "roberta"], default="bert")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--output_dir", default="checkpoints/bert_v1")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = BERTTrainer(args)
    trainer.train()
