from pathlib import Path

from feedback_intelligence.app.gradio_app import analyze_reviews_for_demo, create_demo
from feedback_intelligence.config import ReviewAnalysisConfig
from feedback_intelligence.inference.sentiment import SentimentPrediction


class FakePredictor:
    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        rows = []
        for text in texts:
            if "Terrible" in text or "terrible" in text:
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
                        negative_probability=0.12,
                        positive_probability=0.88,
                        confidence=0.88,
                        uncertainty=0.12,
                    )
                )
        return rows

    def describe(self) -> dict[str, str]:
        return {"model_name": "fake-transformer", "backend": "test"}


def test_analyze_reviews_for_demo_returns_tables() -> None:
    summary, theme_frame, review_frame = analyze_reviews_for_demo(
        text=(
            "Terrible pacing and weak script ruined the experience.\n\n"
            "Excellent acting and warm storytelling made this one memorable."
        ),
        base_path=Path("aclImdb"),
        analysis_config=ReviewAnalysisConfig(
            analysis_sample_size=20,
            theme_clusters=2,
            embedding_dimensions=8,
            embedding_max_features=256,
            max_reviews_in_report=10,
        ),
        predictor=FakePredictor(),
    )

    assert "fake-transformer" in summary
    assert "Reviews analyzed" in summary
    assert not theme_frame.empty
    assert not review_frame.empty
    assert "priority_score" in review_frame.columns


def test_create_demo_returns_blocks() -> None:
    demo = create_demo(
        base_path=Path("aclImdb"),
        analysis_config=ReviewAnalysisConfig(
            analysis_sample_size=20,
            theme_clusters=2,
            embedding_dimensions=8,
            embedding_max_features=256,
            max_reviews_in_report=10,
        ),
    )

    assert hasattr(demo, "launch")
