from pathlib import Path

from feedback_intelligence.benchmarks.tfidf_logreg import run_tfidf_logreg_baseline
from feedback_intelligence.config import BaselineExperimentConfig
from feedback_intelligence.inference.sentiment import load_sentiment_predictor
from feedback_intelligence.types import ReviewRecord


def test_tfidf_baseline_returns_metrics(tmp_path: Path) -> None:
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

    model_output_path = tmp_path / "tfidf_logreg.joblib"
    config = BaselineExperimentConfig(
        sample_size=20,
        min_df=1,
        max_features=128,
        model_output_path=str(model_output_path),
    )
    result = run_tfidf_logreg_baseline(train_records, test_records, config)

    assert result.model_name == "tfidf_logistic_regression"
    assert result.model_output_path == str(model_output_path)
    assert "accuracy" in result.validation_metrics
    assert "accuracy" in result.test_metrics
    assert result.dataset["train_rows"] > 0
    assert model_output_path.exists()

    predictor = load_sentiment_predictor(model_output_path, backend="scikit-learn")
    predictions = predictor.predict_batch(["excellent and uplifting", "terrible and dull"])

    assert len(predictions) == 2
    assert {row.predicted_label for row in predictions} <= {"positive", "negative"}
