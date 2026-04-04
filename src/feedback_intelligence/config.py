"""Configuration helpers for repeatable experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BaselineExperimentConfig:
    """Configuration for the TF-IDF + Logistic Regression baseline."""

    sample_size: int = 4_000
    validation_fraction: float = 0.2
    seed: int = 42
    max_features: int = 20_000
    min_df: int = 2
    ngram_min: int = 1
    ngram_max: int = 2
    max_iter: int = 1_000
    c: float = 4.0
    max_error_examples: int = 8
    model_output_path: str = "artifacts/models/tfidf_logreg_imdb.joblib"

    @classmethod
    def from_json(cls, path: Path) -> BaselineExperimentConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewAnalysisConfig:
    """Configuration for review analysis, clustering, and priority scoring."""

    sentiment_backend: str = "scikit-learn"
    sentiment_model_path: str = "artifacts/models/tfidf_logreg_imdb.joblib"
    sentiment_max_length: int = 256
    analysis_sample_size: int = 600
    seed: int = 42
    theme_clusters: int = 6
    embedding_backend: str = "tfidf_svd"
    embedding_dimensions: int = 64
    embedding_max_features: int = 10_000
    top_terms_per_theme: int = 6
    max_reviews_in_report: int = 50

    @classmethod
    def from_json(cls, path: Path) -> ReviewAnalysisConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransformerTrainingConfig:
    """Configuration for fine-tuning the transformer sentiment model."""

    model_name: str = "roberta-base"
    output_dir: str = "artifacts/models/roberta_imdb"
    metrics_output_path: str = "artifacts/models/roberta_imdb_metrics.json"
    train_sample_size: int = 4_000
    test_sample_size: int = 2_000
    validation_fraction: float = 0.2
    seed: int = 42
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 2

    @classmethod
    def from_json(cls, path: Path) -> TransformerTrainingConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AmazonTransferEvaluationConfig:
    """Configuration for evaluating a saved model on Amazon polarity reviews."""

    dataset_name: str = "SetFit/amazon_polarity"
    dataset_split: str = "test"
    dataset_sample_size: int = 2_000
    include_title: bool = True
    seed: int = 42
    sentiment_backend: str = "scikit-learn"
    sentiment_model_path: str = "artifacts/models/tfidf_logreg_imdb.joblib"
    sentiment_max_length: int = 256
    max_error_examples: int = 20

    @classmethod
    def from_json(cls, path: Path) -> AmazonTransferEvaluationConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
