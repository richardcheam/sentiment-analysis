"""Training utilities for transformer sentiment models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from feedback_intelligence.config import TransformerTrainingConfig
from feedback_intelligence.types import ReviewRecord
from feedback_intelligence.utils.io import write_json

LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


@dataclass(slots=True)
class TransformerTrainingResult:
    """Saved metadata for a fine-tuned transformer run."""

    model_name: str
    output_dir: str
    device: str
    train_rows: int
    validation_rows: int
    test_rows: int
    best_epoch: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "device": self.device,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "best_epoch": self.best_epoch,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "history": self.history,
            "config": self.config,
        }


class ReviewClassificationDataset(Dataset):
    """Tokenized review dataset for transformer fine-tuning."""

    def __init__(self, records: list[ReviewRecord], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL_TO_ID[record.label], dtype=torch.long),
        }


def train_transformer_model(
    train_records: list[ReviewRecord],
    test_records: list[ReviewRecord],
    config: TransformerTrainingConfig,
) -> TransformerTrainingResult:
    """Fine-tune a transformer model and save the artifacts."""
    _seed_everything(config.seed)

    train_split, validation_split = train_test_split(
        train_records,
        test_size=config.validation_fraction,
        random_state=config.seed,
        stratify=[record.label for record in train_records],
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    device = _resolve_device()
    model.to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        ReviewClassificationDataset(train_split, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        ReviewClassificationDataset(validation_split, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ReviewClassificationDataset(test_records, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_metrics: dict[str, Any] | None = None
    best_validation_score = float("-inf")

    for epoch in range(config.epochs):
        average_train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            config.epochs,
        )
        validation_metrics = _evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(average_train_loss, 4),
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if float(validation_metrics["macro_f1"]) > best_validation_score:
            best_validation_score = float(validation_metrics["macro_f1"])
            best_epoch = epoch + 1
            best_validation_metrics = validation_metrics
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    if best_validation_metrics is None:
        raise RuntimeError("Transformer training completed without recording validation metrics.")

    if best_epoch != config.epochs:
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        model.to(device)

    test_metrics = _evaluate(model, test_loader, device)

    result = TransformerTrainingResult(
        model_name=config.model_name,
        output_dir=str(output_dir),
        device=str(device),
        train_rows=len(train_split),
        validation_rows=len(validation_split),
        test_rows=len(test_records),
        best_epoch=best_epoch,
        validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
        history=history,
        config=config.to_dict(),
    )
    write_json(Path(config.metrics_output_path), result.to_dict())
    return result


def _train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
) -> float:
    model.train()
    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch_index + 1}/{total_epochs}",
        leave=False,
    )
    total_loss = 0.0
    batch_count = 0
    for batch in progress:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    if batch_count == 0:
        raise ValueError("Expected at least one training batch.")
    return total_loss / batch_count


def _evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_labels = batch["labels"]
            logits = outputs.logits
            total_loss += float(outputs.loss.item())
            batch_count += 1
            batch_predictions = torch.argmax(logits, dim=-1)
            predictions.extend(batch_predictions.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    if batch_count == 0:
        raise ValueError("Expected at least one evaluation batch.")

    report = classification_report(
        labels,
        predictions,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": round(total_loss / batch_count, 4),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
        "report": _round_nested(report),
    }


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _round_nested(payload: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            rounded[key] = _round_nested(value)
        elif isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded
