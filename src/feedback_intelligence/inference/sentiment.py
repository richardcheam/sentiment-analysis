"""Sentiment inference backends for saved models and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from feedback_intelligence.utils.io import load_joblib


@dataclass(slots=True)
class SentimentPrediction:
    """Normalized prediction output for one review."""

    predicted_label: str
    negative_probability: float
    positive_probability: float
    confidence: float
    uncertainty: float


class SklearnSentimentPredictor:
    """Inference wrapper around a fitted scikit-learn pipeline."""

    def __init__(self, pipeline: Pipeline, model_path: Path | None = None) -> None:
        self.pipeline = pipeline
        self.model_path = model_path

    @classmethod
    def from_path(cls, model_path: Path) -> SklearnSentimentPredictor:
        pipeline = load_joblib(model_path)
        return cls(pipeline=pipeline, model_path=model_path)

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        probabilities = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        positive_index = classes.index("positive")
        negative_index = classes.index("negative")
        predictions = self.pipeline.predict(texts)

        rows: list[SentimentPrediction] = []
        for label, probs in zip(predictions, probabilities, strict=False):
            confidence = float(np.max(probs))
            rows.append(
                SentimentPrediction(
                    predicted_label=str(label),
                    negative_probability=round(float(probs[negative_index]), 4),
                    positive_probability=round(float(probs[positive_index]), 4),
                    confidence=round(confidence, 4),
                    uncertainty=round(1.0 - confidence, 4),
                )
            )
        return rows

    def describe(self) -> dict[str, Any]:
        payload = {"model_name": "tfidf_logistic_regression", "backend": "scikit-learn"}
        if self.model_path is not None:
            payload["model_path"] = str(self.model_path)
        return payload


class TransformerSentimentPredictor:
    """Inference wrapper for a saved Hugging Face sequence classification model."""

    def __init__(self, model_path: Path, max_length: int = 256, device: str = "auto") -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            use_safetensors=True,
        )
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        rows: list[SentimentPrediction] = []
        for probs in probabilities:
            negative_probability = float(probs[0])
            positive_probability = float(probs[1])
            confidence = max(negative_probability, positive_probability)
            predicted_label = (
                "positive" if positive_probability >= negative_probability else "negative"
            )
            rows.append(
                SentimentPrediction(
                    predicted_label=predicted_label,
                    negative_probability=round(negative_probability, 4),
                    positive_probability=round(positive_probability, 4),
                    confidence=round(confidence, 4),
                    uncertainty=round(1.0 - confidence, 4),
                )
            )
        return rows

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": "roberta_finetuned",
            "backend": "transformers",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "max_length": self.max_length,
        }

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def load_sentiment_predictor(
    model_path: Path,
    backend: str,
    max_length: int = 256,
    device: str = "auto",
):
    """Load a sentiment predictor from a saved model artifact."""
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"scikit-learn", "sklearn", "tfidf", "tfidf_logreg"}:
        return SklearnSentimentPredictor.from_path(model_path)
    if normalized_backend in {"transformers", "huggingface"}:
        return TransformerSentimentPredictor(
            model_path=model_path,
            max_length=max_length,
            device=device,
        )
    raise ValueError(f"Unsupported sentiment backend: {backend}")
