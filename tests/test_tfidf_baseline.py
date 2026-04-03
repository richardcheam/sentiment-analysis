from feedback_intelligence.benchmarks.tfidf_logreg import run_tfidf_logreg_baseline
from feedback_intelligence.config import BaselineExperimentConfig
from feedback_intelligence.types import ReviewRecord


def test_tfidf_baseline_returns_metrics() -> None:
    train_records = []
    for index in range(10):
        train_records.append(
            ReviewRecord(
                review_id=f"p-{index}",
                text=f"amazing fun excellent movie number {index}",
                label="positive",
                split="train",
                source="unit-test",
            )
        )
        train_records.append(
            ReviewRecord(
                review_id=f"n-{index}",
                text=f"terrible boring awful movie number {index}",
                label="negative",
                split="train",
                source="unit-test",
            )
        )

    test_records = [
        ReviewRecord(
            review_id="p-test",
            text="excellent uplifting and fun",
            label="positive",
            split="test",
            source="unit-test",
        ),
        ReviewRecord(
            review_id="n-test",
            text="awful dull and terrible",
            label="negative",
            split="test",
            source="unit-test",
        ),
    ]

    config = BaselineExperimentConfig(sample_size=20, min_df=1, max_features=128)
    result = run_tfidf_logreg_baseline(train_records, test_records, config)

    assert result.model_name == "tfidf_logistic_regression"
    assert "accuracy" in result.validation_metrics
    assert "accuracy" in result.test_metrics
    assert result.dataset["train_rows"] > 0
