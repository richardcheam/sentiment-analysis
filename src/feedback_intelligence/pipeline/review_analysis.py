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

THEME_TERM_STOPWORDS = {
    "br",
    "br br",
    "quot",
    "amp",
    "lt",
    "gt",
    "really",
    "just",
}


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
            "metadata": [record.metadata or {} for record in review_records],
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
    frame["theme_label"] = frame["theme_terms"].map(_theme_label)
    frame["requires_manual_review"] = frame.apply(
        lambda row: _requires_manual_review(row, analysis_config),
        axis=1,
    )
    frame["manual_review_reason"] = frame.apply(
        lambda row: _manual_review_reason(row, analysis_config),
        axis=1,
    )
    frame["review_status"] = frame["requires_manual_review"].map(
        lambda value: "manual_review" if bool(value) else "auto_triaged"
    )
    frame["priority_score"] = frame.apply(_priority_score, axis=1)
    frame["priority_level"] = frame["priority_score"].map(_priority_level)
    frame["text_preview"] = frame["text"].str.slice(0, 220).str.replace("\n", " ", regex=False)
    _round_output_columns(
        frame,
        ["positive_probability", "negative_probability", "confidence", "uncertainty"],
    )

    theme_summary = _build_theme_summary(frame, theme_result=theme_result)
    prioritized = frame.sort_values(
        ["priority_score", "negative_probability"],
        ascending=[False, False],
    ).copy()
    prioritized["theme_terms"] = prioritized["theme_terms"].map(
        lambda terms: ", ".join(_clean_theme_terms(terms))
    )

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
                "theme_label",
                "theme_terms",
                "requires_manual_review",
                "manual_review_reason",
                "review_status",
                "priority_score",
                "priority_level",
                "text_preview",
                "text",
                "metadata",
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


def _build_theme_summary(
    frame: pd.DataFrame,
    theme_result=None,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for theme_id, theme_frame in frame.groupby("theme_id", observed=False):
        theme_terms = theme_frame["theme_terms"].iloc[0]
        theme_label = str(theme_frame["theme_label"].iloc[0])
        representative_review = _representative_theme_row(
            frame=frame,
            theme_frame=theme_frame,
            theme_id=int(theme_id),
            theme_result=theme_result,
        )
        negative_rate = float((theme_frame["predicted_label"] == "negative").mean())
        summary_rows.append(
            {
                "theme_id": int(theme_id),
                "theme_label": theme_label,
                "theme_terms": theme_terms,
                "keyword_signature": ", ".join(_clean_theme_terms(theme_terms)),
                "theme_signal": _theme_signal(negative_rate),
                "review_count": int(len(theme_frame)),
                "predicted_negative_rate": round(negative_rate, 2),
                "average_confidence": round(float(theme_frame["confidence"].mean()), 2),
                "average_priority_score": round(float(theme_frame["priority_score"].mean()), 2),
                "manual_review_count": int(theme_frame["requires_manual_review"].sum()),
                "dominant_channel": _dominant_metadata_value(theme_frame, "channel"),
                "dominant_product": _dominant_metadata_value(theme_frame, "product"),
                "representative_review_id": str(representative_review["review_id"]),
                "representative_review_preview": str(representative_review["text_preview"]),
                "example_review_ids": theme_frame["review_id"].head(3).tolist(),
            }
        )
    return sorted(summary_rows, key=lambda row: row["average_priority_score"], reverse=True)


def _round_output_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = frame[column].map(lambda value: round(float(value), 2))


def _requires_manual_review(row: pd.Series, config: ReviewAnalysisConfig) -> bool:
    confidence = float(row["confidence"])
    uncertainty = float(row["uncertainty"])
    return bool(
        confidence < config.manual_review_confidence_threshold
        or uncertainty > config.manual_review_uncertainty_threshold
    )


def _manual_review_reason(row: pd.Series, config: ReviewAnalysisConfig) -> str:
    reasons: list[str] = []
    confidence = float(row["confidence"])
    uncertainty = float(row["uncertainty"])
    if confidence < config.manual_review_confidence_threshold:
        reasons.append("low_confidence")
    if uncertainty > config.manual_review_uncertainty_threshold:
        reasons.append("high_uncertainty")
    return ",".join(reasons) if reasons else "clear_enough"


def _theme_label(terms: list[str]) -> str:
    cleaned_terms = _clean_theme_terms(terms)
    if not cleaned_terms:
        return "General feedback"
    chosen_terms = cleaned_terms[:3]
    return " / ".join(_title_case_phrase(term) for term in chosen_terms)


def _clean_theme_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen_roots: set[str] = set()
    for term in terms:
        normalized = " ".join(
            token
            for token in str(term).strip().lower().split()
            if token and token not in THEME_TERM_STOPWORDS
        )
        if not normalized:
            continue
        root = normalized.replace(" ", "").rstrip("s")
        if root in seen_roots:
            continue
        seen_roots.add(root)
        cleaned.append(normalized)
    return cleaned


def _title_case_phrase(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _theme_signal(predicted_negative_rate: float) -> str:
    if predicted_negative_rate >= 0.7:
        return "Complaint hotspot"
    if predicted_negative_rate <= 0.3:
        return "Positive highlight"
    return "Mixed feedback"


def _dominant_metadata_value(theme_frame: pd.DataFrame, key: str) -> str:
    values = [
        str(value).strip()
        for value in theme_frame["metadata"].map(
            lambda metadata: (metadata or {}).get(key, "")
            if isinstance(metadata, dict)
            else ""
        )
        if str(value).strip()
    ]
    if not values:
        return "n/a"
    return pd.Series(values).value_counts().index[0]


def _representative_theme_row(
    frame: pd.DataFrame,
    theme_frame: pd.DataFrame,
    theme_id: int,
    theme_result,
) -> pd.Series:
    if theme_result is not None and hasattr(theme_result, "representative_row_index_by_cluster"):
        representative_index = theme_result.representative_row_index_by_cluster.get(theme_id)
        if representative_index is not None:
            return frame.iloc[int(representative_index)]
    return (
        theme_frame.sort_values(
            ["requires_manual_review", "priority_score", "negative_probability"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )
