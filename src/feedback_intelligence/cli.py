"""CLI entrypoints for the project reboot."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from feedback_intelligence.app.gradio_app import create_demo
from feedback_intelligence.benchmarks.tfidf_logreg import run_tfidf_logreg_baseline
from feedback_intelligence.config import (
    AmazonTransferEvaluationConfig,
    BaselineExperimentConfig,
    LocalEvaluationConfig,
    ReviewAnalysisConfig,
    TransformerTrainingConfig,
)
from feedback_intelligence.data.amazon_reviews import (
    load_amazon_polarity_reviews,
    summarize_reviews as summarize_amazon_reviews,
)
from feedback_intelligence.data.imdb import load_local_imdb_reviews, summarize_reviews
from feedback_intelligence.data.local_reviews import load_local_labeled_reviews
from feedback_intelligence.inference.sentiment import load_sentiment_predictor
from feedback_intelligence.pipeline.review_analysis import analyze_reviews_with_predictor
from feedback_intelligence.pipeline.transfer_evaluation import evaluate_reviews_with_predictor
from feedback_intelligence.training.transformer import train_transformer_model
from feedback_intelligence.utils.io import write_json

app = typer.Typer(
    help="Feedback Intelligence project commands.",
    no_args_is_help=True,
)
BASE_PATH_OPTION = typer.Option(
    Path("aclImdb"),
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Path to the local IMDb dataset root.",
)
SAMPLE_SIZE_OPTION = typer.Option(
    2_000,
    min=2,
    help="Balanced number of train rows to sample for inspection.",
)
SEED_OPTION = typer.Option(42, help="Deterministic sampling seed.")
OUTPUT_PATH_OPTION = typer.Option(
    Path("artifacts/benchmarks/tfidf_logreg_imdb.json"),
    help="Where to write the benchmark artifact.",
)
CONFIG_PATH_OPTION = typer.Option(
    None,
    exists=True,
    file_okay=True,
    dir_okay=False,
    help="Optional JSON config for the baseline experiment.",
)
ANALYSIS_OUTPUT_OPTION = typer.Option(
    Path("artifacts/analysis/review_analysis_imdb.json"),
    help="Where to write the review analysis artifact.",
)
ANALYSIS_CONFIG_PATH_OPTION = typer.Option(
    None,
    exists=True,
    file_okay=True,
    dir_okay=False,
    help="Optional JSON config for the review analysis workflow.",
)
HOST_OPTION = typer.Option("127.0.0.1", help="Host interface for the demo server.")
PORT_OPTION = typer.Option(7860, min=1, max=65535, help="Port for the demo server.")
SHARE_OPTION = typer.Option(False, help="Create a public Gradio share link.")
TRAINER_CONFIG_PATH_OPTION = typer.Option(
    None,
    exists=True,
    file_okay=True,
    dir_okay=False,
    help="Optional JSON config for transformer training.",
)
TRANSFER_CONFIG_PATH_OPTION = typer.Option(
    None,
    exists=True,
    file_okay=True,
    dir_okay=False,
    help="Optional JSON config for Amazon transfer evaluation.",
)
TRANSFER_OUTPUT_OPTION = typer.Option(
    Path("artifacts/evaluations/amazon_transfer_tfidf_imdb.json"),
    help="Where to write the Amazon transfer evaluation artifact.",
)
LOCAL_EVAL_CONFIG_PATH_OPTION = typer.Option(
    None,
    exists=True,
    file_okay=True,
    dir_okay=False,
    help="Optional JSON config for evaluating a labeled local feedback CSV.",
)
LOCAL_EVAL_OUTPUT_OPTION = typer.Option(
    Path("artifacts/evaluations/customer_feedback_eval_200.json"),
    help="Where to write the local customer-feedback evaluation artifact.",
)


@app.callback()
def main() -> None:
    """Top-level CLI group."""


@app.command("status")
def status(message: str | None = None) -> None:
    """Show the current reboot status."""
    if message:
        typer.echo(message)
        return
    typer.echo("Feedback Intelligence environment is configured.")


@app.command("describe-dataset")
def describe_dataset(
    base_path: Path = BASE_PATH_OPTION,
    sample_size: int = SAMPLE_SIZE_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    """Print a compact summary of the local IMDb dataset."""
    train_records = load_local_imdb_reviews(
        base_path=base_path,
        split="train",
        sample_size=sample_size,
        seed=seed,
    )
    test_records = load_local_imdb_reviews(
        base_path=base_path,
        split="test",
        sample_size=max(sample_size // 2, 2),
        seed=seed,
    )
    payload = {
        "train": summarize_reviews(train_records),
        "test": summarize_reviews(test_records),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("describe-amazon-dataset")
def describe_amazon_dataset(
    sample_size: int = SAMPLE_SIZE_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    """Print a compact summary of a sampled Amazon polarity dataset slice."""
    train_records = load_amazon_polarity_reviews(
        split="train",
        sample_size=sample_size,
        seed=seed,
    )
    test_records = load_amazon_polarity_reviews(
        split="test",
        sample_size=max(sample_size // 2, 2),
        seed=seed,
    )
    payload = {
        "train": summarize_amazon_reviews(train_records),
        "test": summarize_amazon_reviews(test_records),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("run-baseline")
def run_baseline(
    base_path: Path = BASE_PATH_OPTION,
    output_path: Path = OUTPUT_PATH_OPTION,
    config_path: Path = CONFIG_PATH_OPTION,
) -> None:
    """Run the first reproducible benchmark on the local IMDb dataset."""
    config = (
        BaselineExperimentConfig.from_json(config_path)
        if config_path is not None
        else BaselineExperimentConfig()
    )

    train_records = load_local_imdb_reviews(
        base_path=base_path,
        split="train",
        sample_size=config.sample_size,
        seed=config.seed,
    )
    test_records = load_local_imdb_reviews(
        base_path=base_path,
        split="test",
        sample_size=max(config.sample_size // 2, 2),
        seed=config.seed,
    )
    result = run_tfidf_logreg_baseline(
        train_records=train_records,
        test_records=test_records,
        config=config,
    )
    write_json(output_path, result.to_dict())
    typer.echo(f"Wrote benchmark artifact to {output_path}")
    typer.echo(f"Saved baseline model to {config.model_output_path}")


@app.command("train-transformer")
def train_transformer(
    base_path: Path = BASE_PATH_OPTION,
    config_path: Path = TRAINER_CONFIG_PATH_OPTION,
) -> None:
    """Fine-tune a transformer sentiment model and save it for inference."""
    config = (
        TransformerTrainingConfig.from_json(config_path)
        if config_path is not None
        else TransformerTrainingConfig()
    )
    train_records = load_local_imdb_reviews(
        base_path=base_path,
        split="train",
        sample_size=config.train_sample_size,
        seed=config.seed,
    )
    test_records = load_local_imdb_reviews(
        base_path=base_path,
        split="test",
        sample_size=config.test_sample_size,
        seed=config.seed,
    )
    result = train_transformer_model(
        train_records=train_records,
        test_records=test_records,
        config=config,
    )
    typer.echo(f"Saved transformer model to {result.output_dir}")
    typer.echo(f"Best validation checkpoint came from epoch {result.best_epoch}")
    typer.echo(f"Wrote transformer metrics to {config.metrics_output_path}")


@app.command("analyze-reviews")
def analyze_reviews_command(
    base_path: Path = BASE_PATH_OPTION,
    output_path: Path = ANALYSIS_OUTPUT_OPTION,
    config_path: Path = ANALYSIS_CONFIG_PATH_OPTION,
) -> None:
    """Generate clustered review insights and review priorities."""
    analysis_config = (
        ReviewAnalysisConfig.from_json(config_path)
        if config_path is not None
        else ReviewAnalysisConfig()
    )
    analysis_records = load_local_imdb_reviews(
        base_path=base_path,
        split="test",
        sample_size=analysis_config.analysis_sample_size,
        seed=analysis_config.seed,
    )
    predictor = load_sentiment_predictor(
        model_path=Path(analysis_config.sentiment_model_path).resolve(),
        backend=analysis_config.sentiment_backend,
        max_length=analysis_config.sentiment_max_length,
    )
    artifact = analyze_reviews_with_predictor(
        review_records=analysis_records,
        predictor=predictor,
        analysis_config=analysis_config,
        sentiment_model_info=predictor.describe(),
    )
    write_json(output_path, artifact.to_dict())
    typer.echo(f"Wrote review analysis artifact to {output_path}")


@app.command("evaluate-amazon-transfer")
def evaluate_amazon_transfer(
    output_path: Path = TRANSFER_OUTPUT_OPTION,
    config_path: Path = TRANSFER_CONFIG_PATH_OPTION,
) -> None:
    """Evaluate a saved sentiment model on Amazon polarity reviews."""
    config = (
        AmazonTransferEvaluationConfig.from_json(config_path)
        if config_path is not None
        else AmazonTransferEvaluationConfig()
    )
    amazon_records = load_amazon_polarity_reviews(
        split=str(config.dataset_split),
        sample_size=config.dataset_sample_size,
        seed=config.seed,
        include_title=config.include_title,
        dataset_name=config.dataset_name,
    )
    predictor = load_sentiment_predictor(
        model_path=Path(config.sentiment_model_path).resolve(),
        backend=config.sentiment_backend,
        max_length=config.sentiment_max_length,
    )
    artifact = evaluate_reviews_with_predictor(
        review_records=amazon_records,
        predictor=predictor,
        dataset_info={
            "dataset_name": config.dataset_name,
            "split": config.dataset_split,
            "sample_size": config.dataset_sample_size,
            "include_title": config.include_title,
            "seed": config.seed,
        },
        max_error_examples=config.max_error_examples,
    )
    write_json(output_path, artifact.to_dict())
    typer.echo(f"Wrote Amazon transfer evaluation artifact to {output_path}")


@app.command("evaluate-local-feedback")
def evaluate_local_feedback(
    output_path: Path = LOCAL_EVAL_OUTPUT_OPTION,
    config_path: Path = LOCAL_EVAL_CONFIG_PATH_OPTION,
) -> None:
    """Evaluate a saved model on a fixed local labeled customer-feedback CSV."""
    config = (
        LocalEvaluationConfig.from_json(config_path)
        if config_path is not None
        else LocalEvaluationConfig()
    )
    local_records = load_local_labeled_reviews(
        dataset_path=Path(config.dataset_path),
        text_column=config.text_column,
        title_column=config.title_column,
        label_column=config.label_column,
        review_id_column=config.review_id_column,
        split_name=config.split_name,
        source_name=config.source_name,
    )
    predictor = load_sentiment_predictor(
        model_path=Path(config.sentiment_model_path).resolve(),
        backend=config.sentiment_backend,
        max_length=config.sentiment_max_length,
    )
    artifact = evaluate_reviews_with_predictor(
        review_records=local_records,
        predictor=predictor,
        dataset_info={
            "dataset_name": config.source_name,
            "split": config.split_name,
            "dataset_path": config.dataset_path,
        },
        max_error_examples=config.max_error_examples,
    )
    write_json(output_path, artifact.to_dict())
    typer.echo(f"Wrote local feedback evaluation artifact to {output_path}")


@app.command("launch-demo")
def launch_demo(
    base_path: Path = BASE_PATH_OPTION,
    config_path: Path = ANALYSIS_CONFIG_PATH_OPTION,
    host: str = HOST_OPTION,
    port: int = PORT_OPTION,
    share: bool = SHARE_OPTION,
) -> None:
    """Launch the Gradio feedback-intelligence demo."""
    analysis_config = (
        ReviewAnalysisConfig.from_json(config_path)
        if config_path is not None
        else ReviewAnalysisConfig()
    )
    demo = create_demo(base_path=base_path, analysis_config=analysis_config)
    demo.launch(server_name=host, server_port=port, share=share)
