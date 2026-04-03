"""Review-level analysis workflow for feedback intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from feedback_intelligence.analysis.theme_discovery import discover_themes
from feedback_intelligence.benchmarks.tfidf_logreg import fit_tfidf_logreg_pipeline
from feedback_intelligence.config import BaselineExperimentConfig, ReviewAnalysisConfig
from feedback_intelligence.features.embeddings import build_embeddings
from feedback_intelligence.inference.sentiment import SklearnSentimentPredictor
from feedback_intelligence.types import ReviewRecord


@dataclass(slots=True)
class ReviewAnalysisArtifact:
    """Structured output for review intelligence workflows."""

    sentiment_model: dict[str, Any]
    config: dict[str, Any]
    theme_summary: list[dict[str, Any]]
    review_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_reviews(
    train_records: list[ReviewRecord],
    analysis_records: list[ReviewRecord],
    baseline_config: BaselineExperimentConfig,
    analysis_config: ReviewAnalysisConfig,
) -> ReviewAnalysisArtifact:
    """Generate structured insights for a set of reviews."""
    trained = fit_tfidf_logreg_pipeline(train_records, baseline_config)
    predictor = SklearnSentimentPredictor(trained.pipeline)
    return analyze_reviews_with_predictor(
        review_records=analysis_records,
        predictor=predictor,
        analysis_config=analysis_config,
        sentiment_model_info={
            **predictor.describe(),
            "train_rows": trained.train_rows,
            "validation_rows": trained.validation_rows,
            "validation_metrics": trained.validation_metrics,
        },
        extra_config={"baseline": baseline_config.to_dict()},
    )


def analyze_reviews_with_predictor(
    review_records: list[ReviewRecord],
    predictor,
    analysis_config: ReviewAnalysisConfig,
    sentiment_model_info: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> ReviewAnalysisArtifact:
    """Generate structured insights using a loaded sentiment predictor."""
    frame = pd.DataFrame(
        {
            "review_id": [record.review_id for record in review_records],
            "text": [record.text for record in review_records],
            "true_label": [record.label for record in review_records],
            "split": [record.split for record in review_records],
            "word_count": [record.word_count for record in review_records],
        }
    )
    if frame.empty:
        raise ValueError("Expected at least one analysis record.")

    predictions = predictor.predict_batch(frame["text"].tolist())
    frame["predicted_label"] = [row.predicted_label for row in predictions]
    frame["positive_probability"] = [row.positive_probability for row in predictions]
    frame["negative_probability"] = [row.negative_probability for row in predictions]
    frame["confidence"] = [row.confidence for row in predictions]
    frame["uncertainty"] = [row.uncertainty for row in predictions]

    embedding_result = build_embeddings(
        texts=frame["text"].tolist(),
        backend=analysis_config.embedding_backend,
        seed=analysis_config.seed,
        max_features=analysis_config.embedding_max_features,
        dimensions=analysis_config.embedding_dimensions,
    )
    theme_result = discover_themes(
        embeddings=embedding_result.vectors,
        tfidf_matrix=embedding_result.tfidf_matrix,
        feature_names=embedding_result.vectorizer.get_feature_names_out(),
        n_clusters=analysis_config.theme_clusters,
        top_terms_per_cluster=analysis_config.top_terms_per_theme,
        seed=analysis_config.seed,
    )

    frame["theme_id"] = theme_result.assignments
    frame["theme_terms"] = frame["theme_id"].map(
        lambda theme_id: theme_result.top_terms_by_cluster[int(theme_id)]
    )
    frame["priority_score"] = frame.apply(_priority_score, axis=1)
    frame["priority_level"] = frame["priority_score"].map(_priority_level)
    frame["text_preview"] = frame["text"].str.slice(0, 220).str.replace("\n", " ", regex=False)
    _round_output_columns(
        frame,
        ["positive_probability", "negative_probability", "confidence", "uncertainty"],
    )

    theme_summary = _build_theme_summary(frame)
    prioritized = (
        frame.sort_values(["priority_score", "negative_probability"], ascending=[False, False])
        .head(analysis_config.max_reviews_in_report)
        .copy()
    )
    prioritized["theme_terms"] = prioritized["theme_terms"].map(lambda terms: ", ".join(terms))

    return ReviewAnalysisArtifact(
        sentiment_model=sentiment_model_info or predictor.describe(),
        config={
            "analysis": analysis_config.to_dict(),
            **(extra_config or {}),
        },
        theme_summary=theme_summary,
        review_rows=prioritized[
            [
                "review_id",
                "split",
                "true_label",
                "predicted_label",
                "negative_probability",
                "positive_probability",
                "confidence",
                "uncertainty",
                "word_count",
                "theme_id",
                "theme_terms",
                "priority_score",
                "priority_level",
                "text_preview",
            ]
        ].to_dict(orient="records"),
    )


def analyze_review_records(
    review_records: list[ReviewRecord],
    predictor,
    analysis_config: ReviewAnalysisConfig,
    sentiment_model_info: dict[str, Any] | None = None,
) -> ReviewAnalysisArtifact:
    """Public wrapper for analyzing arbitrary review batches."""
    return analyze_reviews_with_predictor(
        review_records=review_records,
        predictor=predictor,
        analysis_config=analysis_config,
        sentiment_model_info=sentiment_model_info,
    )


def _priority_score(row: pd.Series) -> float:
    length_factor = min(float(row["word_count"]) / 400.0, 1.0)
    score = (
        0.65 * float(row["negative_probability"])
        + 0.20 * length_factor
        + 0.15 * float(row["uncertainty"])
    )
    return round(score * 100.0, 2)


def _priority_level(score: float) -> str:
    if score >= 75:
        return "urgent"
    if score >= 55:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _build_theme_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for theme_id, theme_frame in frame.groupby("theme_id", observed=False):
        theme_terms = theme_frame["theme_terms"].iloc[0]
        summary_rows.append(
            {
                "theme_id": int(theme_id),
                "theme_terms": theme_terms,
                "review_count": int(len(theme_frame)),
                "predicted_negative_rate": round(
                    float((theme_frame["predicted_label"] == "negative").mean()), 2
                ),
                "average_confidence": round(float(theme_frame["confidence"].mean()), 2),
                "average_priority_score": round(float(theme_frame["priority_score"].mean()), 2),
                "example_review_ids": theme_frame["review_id"].head(3).tolist(),
            }
        )
    return sorted(summary_rows, key=lambda row: row["average_priority_score"], reverse=True)


def _round_output_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = frame[column].map(lambda value: round(float(value), 2))
