from pathlib import Path

import pytest

from feedback_intelligence.app.gradio_app import (
    analyze_reviews_for_demo,
    create_demo,
    _parse_review_file,
)
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
    pytest.importorskip("gradio")
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


def test_parse_review_file_supports_csv(tmp_path: Path) -> None:
    upload_path = tmp_path / "reviews.csv"
    upload_path.write_text(
        "title,review_text\n"
        '"Battery issue","Stopped working after one day."\n'
        '"Easy fix","Support solved it quickly."\n',
        encoding="utf-8",
    )

    review_texts = _parse_review_file(upload_path)

    assert len(review_texts) == 2
    assert review_texts[0] == "Battery issue\n\nStopped working after one day."
    assert review_texts[1] == "Easy fix\n\nSupport solved it quickly."


def test_analyze_reviews_for_demo_accepts_json_upload(tmp_path: Path) -> None:
    upload_path = tmp_path / "reviews.json"
    upload_path.write_text(
        '[{"title": "Late delivery", "text": "Terrible shipping delay."}, '
        '{"title": "Helpful support", "text": "The team resolved it fast."}]',
        encoding="utf-8",
    )

    summary, theme_frame, review_frame = analyze_reviews_for_demo(
        text="",
        upload_path=str(upload_path),
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

    assert "Reviews analyzed" in summary
    assert not theme_frame.empty
    assert not review_frame.empty
