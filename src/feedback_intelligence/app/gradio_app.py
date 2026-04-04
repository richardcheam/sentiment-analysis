"""Gradio dashboard for the feedback intelligence workflow."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from feedback_intelligence.config import ReviewAnalysisConfig
from feedback_intelligence.inference.sentiment import load_sentiment_predictor
from feedback_intelligence.pipeline.review_analysis import analyze_review_records
from feedback_intelligence.types import ReviewRecord

DEFAULT_DEMO_TEXT = """
The delivery was fast and the packaging looked premium, but the product stopped charging after two days of use.

Support replied within a few minutes and solved my account issue right away. The whole experience felt easy.

The update removed two features our team uses every day. We can work around it, but it slowed us down a lot.

I was skeptical at first, but setup was smooth and the reporting dashboard actually saved me time this week.
"""

DEFAULT_LABEL_FILTER = "all"
DEFAULT_PRIORITY_FILTER = "all"
DEFAULT_SORT_BY = "priority_score"
TEXT_COLUMN_CANDIDATES = (
    "review",
    "review_text",
    "text",
    "content",
    "comment",
    "feedback",
    "message",
    "body",
    "description",
)
TITLE_COLUMN_CANDIDATES = ("title", "subject", "summary", "headline")

DASHBOARD_CSS = """
.app-shell {
  max-width: 1220px;
  margin: 0 auto;
}
.hero {
  padding: 1.4rem 1.5rem;
  border-radius: 22px;
  background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #7c3aed 100%);
  color: white;
  margin-bottom: 1rem;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.22);
}
.hero h1 {
  margin: 0 0 0.35rem 0;
  font-size: 2rem;
  line-height: 1.1;
}
.hero p {
  margin: 0;
  max-width: 760px;
  color: rgba(255, 255, 255, 0.88);
}
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.9rem;
}
.score-card {
  padding: 1rem 1.1rem;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff 0%, #eef2ff 100%);
  border: 1px solid #dbe4ff;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
}
.score-card .eyebrow {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #475569;
}
.score-card .value {
  margin-top: 0.35rem;
  font-size: 1.7rem;
  font-weight: 700;
  color: #0f172a;
}
.score-card .subtext {
  margin-top: 0.25rem;
  font-size: 0.88rem;
  color: #475569;
}
.snapshot-card {
  padding: 1rem 1.1rem;
  border-radius: 18px;
  background: linear-gradient(180deg, #f8fafc 0%, #ecfeff 100%);
  border: 1px solid #c7f9ff;
  box-shadow: 0 10px 30px rgba(14, 165, 233, 0.08);
  color: #0f172a;
}
.snapshot-card h3 {
  margin: 0 0 0.35rem 0;
  font-size: 1rem;
  color: #0f172a;
}
.snapshot-card p {
  margin: 0.15rem 0;
  color: #1e293b;
  font-size: 0.92rem;
}
.snapshot-card strong {
  color: #0f172a;
}
.section-note {
  color: #475569;
  font-size: 0.92rem;
  margin: 0.35rem 0 0.7rem 0;
}
"""


def create_demo(
    base_path: Path = Path("aclImdb"),
    analysis_config: ReviewAnalysisConfig | None = None,
):
    """Create the Gradio Blocks app lazily."""
    import gradio as gr

    resolved_config = analysis_config or ReviewAnalysisConfig()
    snapshot_html = _build_model_snapshot_html(resolved_config)

    with gr.Blocks(title="Feedback Intelligence Dashboard", css=DASHBOARD_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="hero">
                  <h1>Customer Feedback Intelligence Dashboard</h1>
                  <p>
                    Triage customer feedback with a saved sentiment model, surface the most urgent
                    reviews, and compare current behavior against benchmark and transfer results.
                  </p>
                </section>
                """
            )

            gr.HTML(snapshot_html)

            artifact_state = gr.State(value=None)

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    review_input = gr.Textbox(
                        label="Customer Feedback Batch",
                        lines=16,
                        value=DEFAULT_DEMO_TEXT.strip(),
                        placeholder="Paste one review per paragraph...",
                    )
                    review_file = gr.File(
                        label="Upload Customer Feedback (CSV recommended)",
                        file_types=[".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".txt", ".md"],
                        type="filepath",
                    )
                    gr.Markdown(
                        "Paste multiple reviews separated by blank lines. The dashboard scores "
                        "sentiment, estimates triage priority, and groups the batch into themes. "
                        "CSV is the most common spreadsheet export format. Uploads also accept "
                        "TSV, JSON, JSONL/NDJSON, TXT, or Markdown files.",
                        elem_classes=["section-note"],
                    )
                    run_button = gr.Button("Analyze Batch", variant="primary")

                with gr.Column(scale=3):
                    label_filter = gr.Dropdown(
                        label="Predicted Label Filter",
                        choices=["all", "negative", "positive"],
                        value=DEFAULT_LABEL_FILTER,
                    )
                    priority_filter = gr.Dropdown(
                        label="Priority Filter",
                        choices=["all", "urgent", "high", "medium", "low"],
                        value=DEFAULT_PRIORITY_FILTER,
                    )
                    sort_by = gr.Dropdown(
                        label="Sort Reviews By",
                        choices=["priority_score", "negative_probability", "confidence"],
                        value=DEFAULT_SORT_BY,
                    )
                    gr.Markdown(
                        "Filters apply to the most recent analysis result without rerunning the model.",
                        elem_classes=["section-note"],
                    )

            kpi_output = gr.HTML()
            summary_output = gr.Markdown(label="Overview")

            with gr.Row(equal_height=False):
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
                    label="Theme Radar",
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
                    label="Priority Queue",
                    wrap=True,
                )

            run_button.click(
                fn=lambda text, upload, selected_label, selected_priority, selected_sort: analyze_dashboard(
                    text=text,
                    upload_path=upload,
                    label_filter=selected_label,
                    priority_filter=selected_priority,
                    sort_by=selected_sort,
                    base_path=base_path,
                    analysis_config=resolved_config,
                ),
                inputs=[review_input, review_file, label_filter, priority_filter, sort_by],
                outputs=[kpi_output, summary_output, themes_output, reviews_output, artifact_state],
            )

            for component in (label_filter, priority_filter, sort_by):
                component.change(
                    fn=render_dashboard_from_state,
                    inputs=[artifact_state, label_filter, priority_filter, sort_by],
                    outputs=[kpi_output, summary_output, themes_output, reviews_output],
                )

    return demo


def analyze_dashboard(
    text: str,
    upload_path: str | None = None,
    label_filter: str = DEFAULT_LABEL_FILTER,
    priority_filter: str = DEFAULT_PRIORITY_FILTER,
    sort_by: str = DEFAULT_SORT_BY,
    base_path: Path = Path("aclImdb"),
    analysis_config: ReviewAnalysisConfig | None = None,
    predictor=None,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """Analyze pasted reviews and return dashboard-ready outputs plus raw state."""
    try:
        review_texts = _collect_review_texts(text=text, upload_path=upload_path)
    except ValueError as exc:
        empty = pd.DataFrame()
        return (
            _build_empty_kpi_html(),
            f"Could not parse the uploaded feedback file.\n\n{exc}",
            empty,
            empty,
            None,
        )
    if not review_texts:
        empty = pd.DataFrame()
        return (
            _build_empty_kpi_html(),
            "Add at least one review separated by blank lines or upload a supported file.",
            empty,
            empty,
            None,
        )

    resolved_config = analysis_config or ReviewAnalysisConfig()
    loaded_predictor = predictor or _cached_predictor(
        backend=resolved_config.sentiment_backend,
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
    artifact_payload = artifact.to_dict()
    kpi_html, summary, theme_frame, review_frame = _render_dashboard_outputs(
        artifact_payload=artifact_payload,
        label_filter=label_filter,
        priority_filter=priority_filter,
        sort_by=sort_by,
    )
    return kpi_html, summary, theme_frame, review_frame, artifact_payload


def render_dashboard_from_state(
    artifact_payload: dict[str, Any] | None,
    label_filter: str = DEFAULT_LABEL_FILTER,
    priority_filter: str = DEFAULT_PRIORITY_FILTER,
    sort_by: str = DEFAULT_SORT_BY,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    """Render dashboard tables and summaries from saved state."""
    if artifact_payload is None:
        empty = pd.DataFrame()
        return _build_empty_kpi_html(), "Analyze a batch to populate the dashboard.", empty, empty
    return _render_dashboard_outputs(
        artifact_payload=artifact_payload,
        label_filter=label_filter,
        priority_filter=priority_filter,
        sort_by=sort_by,
    )


def analyze_reviews_for_demo(
    text: str,
    upload_path: str | None = None,
    base_path: Path = Path("aclImdb"),
    analysis_config: ReviewAnalysisConfig | None = None,
    predictor=None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible helper for tests and simple demo usage."""
    _, summary, theme_frame, review_frame, _ = analyze_dashboard(
        text=text,
        upload_path=upload_path,
        label_filter=DEFAULT_LABEL_FILTER,
        priority_filter=DEFAULT_PRIORITY_FILTER,
        sort_by=DEFAULT_SORT_BY,
        base_path=base_path,
        analysis_config=analysis_config,
        predictor=predictor,
    )
    return summary, theme_frame, review_frame


def _render_dashboard_outputs(
    artifact_payload: dict[str, Any],
    label_filter: str,
    priority_filter: str,
    sort_by: str,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    review_frame = pd.DataFrame(artifact_payload["review_rows"])
    filtered_reviews = _filter_review_frame(
        review_frame=review_frame,
        label_filter=label_filter,
        priority_filter=priority_filter,
    )
    sorted_reviews = _sort_review_frame(filtered_reviews, sort_by=sort_by)

    theme_frame = _build_theme_frame_from_reviews(sorted_reviews)
    summary = _build_summary_markdown(
        review_frame=sorted_reviews,
        artifact_payload=artifact_payload,
        label_filter=label_filter,
        priority_filter=priority_filter,
    )
    kpi_html = _build_kpi_html(sorted_reviews, artifact_payload["sentiment_model"])

    formatted_review_frame = _format_display_frame(
        sorted_reviews[
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
        ],
        ["negative_probability", "confidence", "priority_score"],
    )
    formatted_theme_frame = _format_display_frame(
        theme_frame,
        ["predicted_negative_rate", "average_confidence", "average_priority_score"],
    )
    return kpi_html, summary, formatted_theme_frame, formatted_review_frame


def _filter_review_frame(
    review_frame: pd.DataFrame,
    label_filter: str,
    priority_filter: str,
) -> pd.DataFrame:
    filtered = review_frame.copy()
    if label_filter != "all":
        filtered = filtered[filtered["predicted_label"] == label_filter]
    if priority_filter != "all":
        filtered = filtered[filtered["priority_level"] == priority_filter]
    return filtered.reset_index(drop=True)


def _sort_review_frame(review_frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if review_frame.empty:
        return review_frame
    allowed_sorts = {"priority_score", "negative_probability", "confidence"}
    sort_column = sort_by if sort_by in allowed_sorts else DEFAULT_SORT_BY
    return review_frame.sort_values(by=sort_column, ascending=False).reset_index(drop=True)


def _build_theme_frame_from_reviews(review_frame: pd.DataFrame) -> pd.DataFrame:
    if review_frame.empty:
        return pd.DataFrame(
            columns=[
                "theme_id",
                "theme_terms",
                "review_count",
                "predicted_negative_rate",
                "average_confidence",
                "average_priority_score",
                "example_review_ids",
            ]
        )

    rows: list[dict[str, Any]] = []
    for theme_id, theme_group in review_frame.groupby("theme_id", observed=False):
        rows.append(
            {
                "theme_id": int(theme_id),
                "theme_terms": theme_group["theme_terms"].iloc[0],
                "review_count": int(len(theme_group)),
                "predicted_negative_rate": round(
                    float((theme_group["predicted_label"] == "negative").mean()),
                    2,
                ),
                "average_confidence": round(float(theme_group["confidence"].mean()), 2),
                "average_priority_score": round(float(theme_group["priority_score"].mean()), 2),
                "example_review_ids": ", ".join(theme_group["review_id"].head(3).tolist()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(by="average_priority_score", ascending=False).reset_index(drop=True)


def _build_summary_markdown(
    review_frame: pd.DataFrame,
    artifact_payload: dict[str, Any],
    label_filter: str,
    priority_filter: str,
) -> str:
    if review_frame.empty:
        return (
            f"**Filtered reviews:** 0\n\n"
            f"**Predicted label filter:** {label_filter}\n\n"
            f"**Priority filter:** {priority_filter}\n\n"
            "No reviews match the current filter selection."
        )

    top_theme = review_frame.iloc[0]["theme_terms"]
    urgent_count = int((review_frame["priority_level"] == "urgent").sum())
    predicted_negative_rate = float((review_frame["predicted_label"] == "negative").mean())
    model_name = artifact_payload["sentiment_model"].get("model_name", "unknown")
    total_reviews = len(artifact_payload["review_rows"])
    return (
        f"**Sentiment model:** {model_name}\n\n"
        f"**Reviews analyzed:** {total_reviews}\n\n"
        f"**Visible reviews:** {len(review_frame)} of {total_reviews}\n\n"
        f"**Predicted negative rate:** {predicted_negative_rate * 100:.2f}%\n\n"
        f"**Urgent reviews in view:** {urgent_count}\n\n"
        f"**Current top theme:** {top_theme}\n\n"
        f"**Filters:** label=`{label_filter}`, priority=`{priority_filter}`"
    )


def _build_kpi_html(review_frame: pd.DataFrame, sentiment_model: dict[str, Any]) -> str:
    if review_frame.empty:
        return _build_empty_kpi_html()

    urgent_count = int((review_frame["priority_level"] == "urgent").sum())
    negative_rate = float((review_frame["predicted_label"] == "negative").mean()) * 100.0
    avg_confidence = float(review_frame["confidence"].mean()) * 100.0
    top_priority = float(review_frame["priority_score"].max())
    backend = str(sentiment_model.get("backend", "unknown"))

    cards = [
        ("Visible Reviews", str(len(review_frame)), "Current reviews after filters"),
        ("Negative Rate", f"{negative_rate:.1f}%", "Predicted negative share"),
        ("Urgent Items", str(urgent_count), "Reviews flagged for fastest triage"),
        ("Avg Confidence", f"{avg_confidence:.1f}%", f"Model backend: {backend}"),
        ("Top Priority", f"{top_priority:.2f}", "Highest current risk score"),
    ]
    return (
        '<div class="card-grid">'
        + "".join(
            f"""
            <section class="score-card">
              <div class="eyebrow">{title}</div>
              <div class="value">{value}</div>
              <div class="subtext">{subtext}</div>
            </section>
            """
            for title, value, subtext in cards
        )
        + "</div>"
    )


def _build_empty_kpi_html() -> str:
    return (
        '<div class="card-grid">'
        '<section class="score-card"><div class="eyebrow">Visible Reviews</div>'
        '<div class="value">0</div><div class="subtext">Analyze a batch to populate the dashboard</div></section>'
        "</div>"
    )


def _build_model_snapshot_html(analysis_config: ReviewAnalysisConfig) -> str:
    benchmark_payload = _read_json_if_exists(Path("artifacts/benchmarks/tfidf_logreg_imdb.json"))
    transfer_payload = _read_json_if_exists(Path("artifacts/evaluations/amazon_transfer_tfidf_imdb.json"))

    imdb_accuracy = _extract_nested_metric(benchmark_payload, ["test_metrics", "accuracy"])
    imdb_macro_f1 = _extract_nested_metric(benchmark_payload, ["test_metrics", "macro_f1"])
    amazon_accuracy = _extract_nested_metric(transfer_payload, ["metrics", "accuracy"])
    amazon_macro_f1 = _extract_nested_metric(transfer_payload, ["metrics", "macro_f1"])
    backend = analysis_config.sentiment_backend
    model_path = analysis_config.sentiment_model_path

    return f"""
    <div class="card-grid" style="margin-bottom: 1rem;">
      <section class="snapshot-card">
        <h3>Active Inference Model</h3>
        <p><strong>Backend:</strong> {backend}</p>
        <p><strong>Artifact:</strong> {model_path}</p>
      </section>
      <section class="snapshot-card">
        <h3>IMDb Benchmark</h3>
        <p><strong>Test accuracy:</strong> {_format_metric(imdb_accuracy)}</p>
        <p><strong>Macro F1:</strong> {_format_metric(imdb_macro_f1)}</p>
      </section>
      <section class="snapshot-card">
        <h3>Amazon Transfer</h3>
        <p><strong>Accuracy:</strong> {_format_metric(amazon_accuracy)}</p>
        <p><strong>Macro F1:</strong> {_format_metric(amazon_macro_f1)}</p>
      </section>
    </div>
    """


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_nested_metric(payload: dict[str, Any] | None, keys: list[str]) -> float | None:
    if payload is None:
        return None
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (float, int)):
        return float(current)
    return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.4f}"


@lru_cache(maxsize=2)
def _cached_predictor(
    backend: str,
    model_path: Path,
    max_length: int,
):
    return load_sentiment_predictor(
        model_path=model_path,
        backend=backend,
        max_length=max_length,
    )


def _split_reviews(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def _collect_review_texts(text: str, upload_path: str | None) -> list[str]:
    review_texts = _split_reviews(text)
    if upload_path:
        review_texts.extend(_parse_review_file(upload_path))
    return review_texts


def _parse_review_file(upload_path: str | Path) -> list[str]:
    path = Path(upload_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return _extract_review_texts_from_frame(frame)
    if suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
        return _extract_review_texts_from_frame(frame)
    if suffix in {".txt", ".md"}:
        return _split_reviews(path.read_text(encoding="utf-8"))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _extract_review_texts_from_json_payload(payload)
    if suffix in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _extract_review_texts_from_json_payload(rows)
    raise ValueError(
        f"Unsupported file type '{suffix or 'unknown'}'. Use CSV, TSV, JSON, JSONL/NDJSON, TXT, or MD."
    )


def _extract_review_texts_from_json_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, str) for item in payload):
            return [item.strip() for item in payload if item.strip()]
        if all(isinstance(item, dict) for item in payload):
            return _extract_review_texts_from_frame(pd.DataFrame(payload))
    if isinstance(payload, dict):
        for key in ("reviews", "records", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return _extract_review_texts_from_json_payload(value)
        if any(isinstance(value, list) for value in payload.values()):
            return _extract_review_texts_from_frame(pd.DataFrame(payload))
    raise ValueError(
        "JSON upload must be a list of strings, a list of objects, or an object containing reviews/data/records."
    )


def _extract_review_texts_from_frame(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []

    normalized_columns = {
        column: _normalize_column_name(str(column))
        for column in frame.columns
    }
    text_column = _pick_column(normalized_columns, TEXT_COLUMN_CANDIDATES)
    title_column = _pick_column(normalized_columns, TITLE_COLUMN_CANDIDATES, exclude=text_column)

    if text_column is None:
        if len(frame.columns) == 1:
            text_column = frame.columns[0]
        else:
            string_columns = [
                column
                for column in frame.columns
                if pd.api.types.is_string_dtype(frame[column]) or frame[column].dtype == object
            ]
            if len(string_columns) == 1:
                text_column = string_columns[0]
            else:
                raise ValueError(
                    "Could not find a review text column. Try one of: "
                    f"{', '.join(TEXT_COLUMN_CANDIDATES)}. "
                    f"Available columns: {', '.join(map(str, frame.columns))}"
                )

    texts: list[str] = []
    for row in frame.fillna("").to_dict(orient="records"):
        body = str(row.get(text_column, "")).strip()
        title = str(row.get(title_column, "")).strip() if title_column is not None else ""
        combined = f"{title}\n\n{body}".strip() if title and body else (title or body)
        if combined:
            texts.append(combined)
    return texts


def _pick_column(
    normalized_columns: dict[Any, str],
    candidates: tuple[str, ...],
    exclude: Any | None = None,
) -> Any | None:
    for candidate in candidates:
        for column, normalized in normalized_columns.items():
            if column == exclude:
                continue
            if normalized == candidate:
                return column
    return None


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _format_display_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.2f}")
    return formatted
