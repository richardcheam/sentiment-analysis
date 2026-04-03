"""Sentiment inference backends for saved models and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

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
        return {"model_name": "tfidf_logistic_regression", "backend": "scikit-learn"}


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
