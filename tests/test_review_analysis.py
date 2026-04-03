from feedback_intelligence.config import BaselineExperimentConfig, ReviewAnalysisConfig
from feedback_intelligence.pipeline.review_analysis import analyze_reviews
from feedback_intelligence.types import ReviewRecord


def test_review_analysis_returns_priority_and_themes() -> None:
    train_records = []
    for index in range(12):
        train_records.append(
            ReviewRecord(
                review_id=f"p-{index}",
                text="excellent acting moving story heartfelt performances",
                label="positive",
                split="train",
                source="unit-test",
            )
        )
        train_records.append(
            ReviewRecord(
                review_id=f"n-{index}",
                text="terrible pacing boring script weak direction disappointing",
                label="negative",
                split="train",
                source="unit-test",
            )
        )

    analysis_records = [
        ReviewRecord(
            review_id="a-1",
            text="terrible pacing and weak script made this disappointing",
            label="negative",
            split="test",
            source="unit-test",
        ),
        ReviewRecord(
            review_id="a-2",
            text="excellent acting and heartfelt performances made this moving",
            label="positive",
            split="test",
            source="unit-test",
        ),
        ReviewRecord(
            review_id="a-3",
            text="boring direction but a few strong scenes kept it watchable",
            label="negative",
            split="test",
            source="unit-test",
        ),
        ReviewRecord(
            review_id="a-4",
            text="moving story with excellent acting and warm energy",
            label="positive",
            split="test",
            source="unit-test",
        ),
    ]

    artifact = analyze_reviews(
        train_records=train_records,
        analysis_records=analysis_records,
        baseline_config=BaselineExperimentConfig(sample_size=24, min_df=1, max_features=128),
        analysis_config=ReviewAnalysisConfig(
            analysis_sample_size=4,
            theme_clusters=2,
            embedding_dimensions=4,
            embedding_max_features=64,
            max_reviews_in_report=4,
        ),
    )

    assert artifact.theme_summary
    assert len(artifact.review_rows) == 4
    assert "priority_score" in artifact.review_rows[0]
    assert "priority_level" in artifact.review_rows[0]
    assert "theme_terms" in artifact.review_rows[0]
