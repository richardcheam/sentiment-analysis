"""Baseline benchmark for review sentiment classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from feedback_intelligence.config import BaselineExperimentConfig
from feedback_intelligence.types import ReviewRecord
from feedback_intelligence.utils.io import save_joblib


@dataclass(slots=True)
class BenchmarkResult:
    """Structured output for a benchmark run."""

    model_name: str
    model_output_path: str | None
    config: dict[str, Any]
    dataset: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    slice_metrics: list[dict[str, Any]]
    error_examples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainedBaselineModel:
    """Fitted baseline pipeline with validation metadata."""

    pipeline: Pipeline
    train_rows: int
    validation_rows: int
    validation_metrics: dict[str, Any]


def run_tfidf_logreg_baseline(
    train_records: list[ReviewRecord],
    test_records: list[ReviewRecord],
    config: BaselineExperimentConfig,
    model_output_path: Path | None = None,
) -> BenchmarkResult:
    """Fit a TF-IDF + Logistic Regression benchmark and return rich metrics."""
    trained = fit_tfidf_logreg_pipeline(train_records, config)
    test_frame = _records_to_frame(test_records)
    pipeline = trained.pipeline
    resolved_model_output_path = model_output_path or Path(config.model_output_path)
    save_joblib(resolved_model_output_path, pipeline)

    test_metrics = _evaluate_frame(pipeline, test_frame)
    slice_metrics = _slice_metrics(test_frame, pipeline)
    error_examples = _error_examples(
        frame=test_frame,
        pipeline=pipeline,
        limit=config.max_error_examples,
    )

    return BenchmarkResult(
        model_name="tfidf_logistic_regression",
        model_output_path=str(resolved_model_output_path),
        config=config.to_dict(),
        dataset={
            "train_rows": trained.train_rows,
            "validation_rows": trained.validation_rows,
            "test_rows": len(test_frame),
        },
        validation_metrics=trained.validation_metrics,
        test_metrics=test_metrics,
        slice_metrics=slice_metrics,
        error_examples=error_examples,
    )


def fit_tfidf_logreg_pipeline(
    train_records: list[ReviewRecord],
    config: BaselineExperimentConfig,
) -> TrainedBaselineModel:
    """Fit the TF-IDF + Logistic Regression pipeline and keep validation metrics."""
    train_frame = _records_to_frame(train_records)

    train_split, val_split = train_test_split(
        train_frame,
        test_size=config.validation_fraction,
        random_state=config.seed,
        stratify=train_frame["label"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=config.max_features,
                    min_df=config.min_df,
                    ngram_range=(config.ngram_min, config.ngram_max),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config.max_iter,
                    C=config.c,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_split["text"], train_split["label"])

    return TrainedBaselineModel(
        pipeline=pipeline,
        train_rows=len(train_split),
        validation_rows=len(val_split),
        validation_metrics=_evaluate_frame(pipeline, val_split),
    )


def _records_to_frame(records: list[ReviewRecord]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "review_id": [record.review_id for record in records],
            "text": [record.text for record in records],
            "label": [record.label for record in records],
            "word_count": [record.word_count for record in records],
        }
    )
    if frame.empty:
        raise ValueError("Expected at least one review record.")
    return frame


def _evaluate_frame(pipeline: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    predictions = pipeline.predict(frame["text"])
    report = classification_report(
        frame["label"],
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(frame["label"], predictions)), 4),
        "macro_f1": round(float(f1_score(frame["label"], predictions, average="macro")), 4),
        "report": _round_nested(report),
    }


def _slice_metrics(frame: pd.DataFrame, pipeline: Pipeline) -> list[dict[str, Any]]:
    enriched = frame.copy()
    enriched["prediction"] = pipeline.predict(enriched["text"])
    enriched["length_bucket"] = pd.cut(
        enriched["word_count"],
        bins=[0, 100, 250, 500, float("inf")],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True,
    )

    rows: list[dict[str, Any]] = []
    for bucket, bucket_frame in enriched.groupby("length_bucket", observed=False):
        if bucket_frame.empty:
            continue
        rows.append(
            {
                "length_bucket": str(bucket),
                "rows": int(len(bucket_frame)),
                "accuracy": round(
                    float(accuracy_score(bucket_frame["label"], bucket_frame["prediction"])), 4
                ),
            }
        )
    return rows


def _error_examples(frame: pd.DataFrame, pipeline: Pipeline, limit: int) -> list[dict[str, Any]]:
    probabilities = pipeline.predict_proba(frame["text"])
    predictions = pipeline.classes_[probabilities.argmax(axis=1)]

    errors = frame.copy()
    errors["prediction"] = predictions
    errors["confidence"] = probabilities.max(axis=1)
    errors = errors[errors["label"] != errors["prediction"]]
    errors = errors.sort_values(by="confidence", ascending=False)

    examples: list[dict[str, Any]] = []
    for row in errors.head(limit).itertuples(index=False):
        examples.append(
            {
                "review_id": row.review_id,
                "label": row.label,
                "prediction": row.prediction,
                "confidence": round(float(row.confidence), 4),
                "word_count": int(row.word_count),
                "text_preview": row.text[:220].replace("\n", " "),
            }
        )
    return examples


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
