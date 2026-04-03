"""Gradio demo for the feedback intelligence workflow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from feedback_intelligence.config import ReviewAnalysisConfig
from feedback_intelligence.inference.sentiment import TransformerSentimentPredictor
from feedback_intelligence.pipeline.review_analysis import analyze_review_records
from feedback_intelligence.types import ReviewRecord

DEFAULT_DEMO_TEXT = """
The acting was excellent and the story felt sincere, but the pacing dragged in the middle.

This movie was painfully boring, badly written, and felt much longer than it needed to.

I expected a cheap horror movie, but it was surprisingly fun and the practical effects were great.
"""


def create_demo(
    base_path: Path = Path("aclImdb"),
    analysis_config: ReviewAnalysisConfig | None = None,
):
    """Create the Gradio Blocks app lazily."""
    import gradio as gr

    resolved_config = analysis_config or ReviewAnalysisConfig()

    with gr.Blocks(title="Feedback Intelligence Demo") as demo:
        gr.Markdown(
            """
            # Feedback Intelligence Demo

            Paste one or more reviews separated by blank lines. The demo will score sentiment,
            assign review priority, and cluster the batch into themes.
            """
        )

        review_input = gr.Textbox(
            label="Reviews",
            lines=14,
            value=DEFAULT_DEMO_TEXT,
            placeholder="Paste one review per paragraph...",
        )
        run_button = gr.Button("Analyze Reviews", variant="primary")

        summary_output = gr.Markdown(label="Summary")
        themes_output = gr.Dataframe(
            headers=[
                "theme_id",
                "theme_terms",
                "review_count",
                "predicted_negative_rate",
                "average_confidence",
                "average_priority_score",
                "example_review_ids",
            ],
            label="Theme Summary",
            wrap=True,
        )
        reviews_output = gr.Dataframe(
            headers=[
                "review_id",
                "predicted_label",
                "negative_probability",
                "confidence",
                "theme_terms",
                "priority_level",
                "priority_score",
                "text_preview",
            ],
            label="Priority Review List",
            wrap=True,
        )

        run_button.click(
            fn=lambda text: analyze_reviews_for_demo(
                text=text,
                base_path=base_path,
                analysis_config=resolved_config,
            ),
            inputs=[review_input],
            outputs=[summary_output, themes_output, reviews_output],
        )

    return demo


def analyze_reviews_for_demo(
    text: str,
    base_path: Path = Path("aclImdb"),
    analysis_config: ReviewAnalysisConfig | None = None,
    predictor=None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Analyze pasted reviews and return Gradio-friendly outputs."""
    review_texts = _split_reviews(text)
    if not review_texts:
        empty = pd.DataFrame()
        return "Add at least one review separated by blank lines.", empty, empty

    resolved_config = analysis_config or ReviewAnalysisConfig()
    loaded_predictor = predictor or _cached_transformer_predictor(
        model_path=Path(resolved_config.sentiment_model_path).resolve(),
        max_length=resolved_config.sentiment_max_length,
    )
    review_records = [
        ReviewRecord(
            review_id=f"user-{index + 1}",
            text=review_text,
            label="unknown",
            split="demo",
            source="gradio-demo",
        )
        for index, review_text in enumerate(review_texts)
    ]

    artifact = analyze_review_records(
        review_records=review_records,
        predictor=loaded_predictor,
        analysis_config=resolved_config,
        sentiment_model_info=loaded_predictor.describe(),
    )

    review_frame = pd.DataFrame(artifact.review_rows)
    theme_frame = pd.DataFrame(artifact.theme_summary)
    review_frame = _format_display_frame(
        review_frame,
        ["negative_probability", "confidence", "priority_score"],
    )
    theme_frame = _format_display_frame(
        theme_frame,
        ["predicted_negative_rate", "average_confidence", "average_priority_score"],
    )
    summary = _build_summary_markdown(
        review_frame,
        artifact.theme_summary,
        artifact.sentiment_model,
    )
    return summary, theme_frame, review_frame[
        [
            "review_id",
            "predicted_label",
            "negative_probability",
            "confidence",
            "theme_terms",
            "priority_level",
            "priority_score",
            "text_preview",
        ]
    ]


def _build_summary_markdown(
    review_frame: pd.DataFrame,
    theme_summary: list[dict[str, object]],
    sentiment_model: dict[str, object],
) -> str:
    urgent_count = int((review_frame["priority_level"] == "urgent").sum())
    predicted_negative_rate = float((review_frame["predicted_label"] == "negative").mean())
    top_theme = theme_summary[0] if theme_summary else None
    theme_text = ", ".join(top_theme["theme_terms"]) if top_theme else "n/a"
    model_name = sentiment_model.get("model_name", "unknown")
    return (
        f"**Sentiment model:** {model_name}\n\n"
        f"**Reviews analyzed:** {len(review_frame)}\n\n"
        f"**Predicted negative rate:** {predicted_negative_rate * 100:.2f}%\n\n"
        f"**Urgent reviews:** {urgent_count}\n\n"
        f"**Top priority theme:** {theme_text}"
    )

@lru_cache(maxsize=2)
def _cached_transformer_predictor(
    model_path: Path,
    max_length: int,
) -> TransformerSentimentPredictor:
    return TransformerSentimentPredictor(model_path=model_path, max_length=max_length)


def _split_reviews(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def _format_display_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.2f}")
    return formatted
