from typer.testing import CliRunner

from feedback_intelligence.cli import app
from feedback_intelligence.inference.sentiment import SentimentPrediction
from feedback_intelligence.types import ReviewRecord


def test_status_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "configured" in result.stdout.lower()


class _FakePredictor:
    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        return [
            SentimentPrediction(
                predicted_label="positive",
                negative_probability=0.1,
                positive_probability=0.9,
                confidence=0.9,
                uncertainty=0.1,
            )
            for _ in texts
        ]

    def describe(self) -> dict[str, str]:
        return {"model_name": "fake-cli-model", "backend": "test"}


def test_evaluate_local_feedback_command(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    output_path = tmp_path / "local_eval.json"
    config_path = tmp_path / "eval_config.json"
    config_path.write_text(
        (
            '{'
            '"dataset_path":"data/eval/customer_feedback_amazon_eval_200.csv",'
            '"text_column":"review_text",'
            '"title_column":"title",'
            '"label_column":"label",'
            '"review_id_column":"review_id",'
            '"split_name":"eval",'
            '"source_name":"amazon_polarity_eval_200",'
            '"sentiment_backend":"scikit-learn",'
            '"sentiment_model_path":"artifacts/models/tfidf_logreg_imdb.joblib",'
            '"sentiment_max_length":256,'
            '"max_error_examples":5'
            '}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "feedback_intelligence.cli.load_local_labeled_reviews",
        lambda **kwargs: [
            ReviewRecord(
                review_id="eval-1",
                text="Helpful support and easy setup.",
                label="positive",
                split="eval",
                source="amazon_polarity_eval_200",
            )
        ],
    )
    monkeypatch.setattr(
        "feedback_intelligence.cli.load_sentiment_predictor",
        lambda **kwargs: _FakePredictor(),
    )

    result = runner.invoke(
        app,
        [
            "evaluate-local-feedback",
            "--config-path",
            str(config_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "local feedback evaluation artifact" in result.stdout.lower()
