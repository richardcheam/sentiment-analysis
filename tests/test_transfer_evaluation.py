from feedback_intelligence.inference.sentiment import SentimentPrediction
from feedback_intelligence.pipeline.transfer_evaluation import evaluate_reviews_with_predictor
from feedback_intelligence.types import ReviewRecord


class FakePredictor:
    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        rows = []
        for text in texts:
            if "terrible" in text.lower():
                rows.append(
                    SentimentPrediction(
                        predicted_label="negative",
                        negative_probability=0.91,
                        positive_probability=0.09,
                        confidence=0.91,
                        uncertainty=0.09,
                    )
                )
            else:
                rows.append(
                    SentimentPrediction(
                        predicted_label="positive",
                        negative_probability=0.08,
                        positive_probability=0.92,
                        confidence=0.92,
                        uncertainty=0.08,
                    )
                )
        return rows

    def describe(self) -> dict[str, str]:
        return {"model_name": "fake-transfer-model", "backend": "test"}


def test_evaluate_reviews_with_predictor_returns_transfer_artifact() -> None:
    review_records = [
        ReviewRecord(
            review_id="a-1",
            text="Terrible battery life and flimsy build.",
            label="negative",
            split="test",
            source="amazon-test",
        ),
        ReviewRecord(
            review_id="a-2",
            text="Excellent value and great quality.",
            label="positive",
            split="test",
            source="amazon-test",
        ),
    ]

    artifact = evaluate_reviews_with_predictor(
        review_records=review_records,
        predictor=FakePredictor(),
        dataset_info={"dataset_name": "amazon-test", "split": "test"},
        max_error_examples=5,
    )

    assert artifact.dataset["rows"] == 2
    assert artifact.metrics["accuracy"] == 1.0
    assert artifact.metrics["macro_f1"] == 1.0
    assert artifact.sentiment_model["model_name"] == "fake-transfer-model"
    assert artifact.slice_metrics
    assert artifact.error_examples == []
