from pathlib import Path

import pytest

from feedback_intelligence.app.gradio_app import (
    analyze_dashboard,
    analyze_reviews_for_demo,
    create_demo,
    _parse_review_file,
    _parse_review_file_records,
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
        "review_id,channel,title,review_text\n"
        '"r-1","email","Battery issue","Stopped working after one day."\n'
        '"r-2","chat","Easy fix","Support solved it quickly."\n',
        encoding="utf-8",
    )

    review_texts = _parse_review_file(upload_path)
    parsed_rows = _parse_review_file_records(upload_path)

    assert len(review_texts) == 2
    assert review_texts[0] == "Battery issue\n\nStopped working after one day."
    assert review_texts[1] == "Easy fix\n\nSupport solved it quickly."
    assert parsed_rows[0].review_id == "r-1"
    assert parsed_rows[0].metadata["channel"] == "email"


def test_analyze_reviews_for_demo_accepts_json_upload(tmp_path: Path) -> None:
    upload_path = tmp_path / "reviews.json"
    upload_path.write_text(
        '[{"review_id": "case-1", "channel": "email", "product": "router", '
        '"title": "Late delivery", "text": "Terrible shipping delay."}, '
        '{"review_id": "case-2", "channel": "chat", "product": "router", '
        '"title": "Helpful support", "text": "The team resolved it fast."}]',
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
            manual_review_confidence_threshold=0.9,
            manual_review_uncertainty_threshold=0.1,
            max_reviews_in_report=10,
        ),
        predictor=FakePredictor(),
    )

    assert "Reviews analyzed" in summary
    assert "Manual-review items in view" in summary
    assert "Channel breakdown" in summary
    assert not theme_frame.empty
    assert not review_frame.empty
    assert "channel" in review_frame.columns
    assert "product" in review_frame.columns
    assert "review_status" in review_frame.columns
    assert "manual_review_reason" in review_frame.columns
    assert "manual_review" in review_frame["review_status"].tolist()


def test_analyze_dashboard_writes_export_csv(tmp_path: Path) -> None:
    upload_path = tmp_path / "reviews.csv"
    upload_path.write_text(
        "review_id,channel,product,title,review_text\n"
        + "\n".join(
            (
                f'"r-{index}","email","router","Issue {index}","Terrible shipping delay {index}."'
                if index % 2
                else f'"r-{index}","chat","router","Resolved {index}","The team resolved it fast {index}."'
            )
            for index in range(1, 13)
        )
        + "\n",
        encoding="utf-8",
    )

    _, summary, _, review_frame, export_path, _ = analyze_dashboard(
        text="",
        upload_path=str(upload_path),
        analysis_config=ReviewAnalysisConfig(
            analysis_sample_size=20,
            theme_clusters=2,
            embedding_dimensions=8,
            embedding_max_features=256,
            max_reviews_in_report=3,
        ),
        predictor=FakePredictor(),
    )

    assert "Visible reviews:** 12 of 12" in summary
    assert len(review_frame) == 12
    assert export_path is not None
    exported = Path(export_path)
    assert exported.exists()
    exported_text = exported.read_text(encoding="utf-8")
    assert "review_id,channel,product" in exported_text
    assert "predicted_label" in exported_text
    assert "theme_label" in exported_text
    assert exported_text.count("\n") >= 13
