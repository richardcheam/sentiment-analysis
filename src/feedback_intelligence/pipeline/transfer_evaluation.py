"""Evaluate a saved sentiment model on an external review dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from feedback_intelligence.types import ReviewRecord


@dataclass(slots=True)
class TransferEvaluationArtifact:
    """Structured output for zero-shot or transfer evaluations."""

    dataset: dict[str, Any]
    sentiment_model: dict[str, Any]
    metrics: dict[str, Any]
    slice_metrics: list[dict[str, Any]]
    error_examples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_reviews_with_predictor(
    review_records: list[ReviewRecord],
    predictor,
    dataset_info: dict[str, Any],
    max_error_examples: int = 20,
) -> TransferEvaluationArtifact:
    """Evaluate a predictor on labeled reviews from another dataset."""
    frame = _records_to_frame(review_records)
    predictions = predictor.predict_batch(frame["text"].tolist())

    frame["predicted_label"] = [row.predicted_label for row in predictions]
    frame["negative_probability"] = [row.negative_probability for row in predictions]
    frame["positive_probability"] = [row.positive_probability for row in predictions]
    frame["confidence"] = [row.confidence for row in predictions]

    return TransferEvaluationArtifact(
        dataset={
            **dataset_info,
            "rows": len(frame),
            "label_distribution": frame["label"].value_counts().sort_index().to_dict(),
        },
        sentiment_model=predictor.describe(),
        metrics=_evaluate_frame(frame),
        slice_metrics=_slice_metrics(frame),
        error_examples=_error_examples(frame, limit=max_error_examples),
    )


def _records_to_frame(records: list[ReviewRecord]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "review_id": [record.review_id for record in records],
            "text": [record.text for record in records],
            "label": [record.label for record in records],
            "split": [record.split for record in records],
            "source": [record.source for record in records],
            "word_count": [record.word_count for record in records],
        }
    )
    if frame.empty:
        raise ValueError("Expected at least one labeled review record.")
    return frame


def _evaluate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    report = classification_report(
        frame["label"],
        frame["predicted_label"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(frame["label"], frame["predicted_label"])), 4),
        "macro_f1": round(
            float(f1_score(frame["label"], frame["predicted_label"], average="macro")),
            4,
        ),
        "report": _round_nested(report),
    }


def _slice_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    enriched = frame.copy()
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
                    float(accuracy_score(bucket_frame["label"], bucket_frame["predicted_label"])),
                    4,
                ),
            }
        )
    return rows


def _error_examples(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    errors = frame[frame["label"] != frame["predicted_label"]].copy()
    errors = errors.sort_values(by="confidence", ascending=False)

    examples: list[dict[str, Any]] = []
    for row in errors.head(limit).itertuples(index=False):
        examples.append(
            {
                "review_id": row.review_id,
                "label": row.label,
                "prediction": row.predicted_label,
                "confidence": round(float(row.confidence), 4),
                "negative_probability": round(float(row.negative_probability), 4),
                "positive_probability": round(float(row.positive_probability), 4),
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
