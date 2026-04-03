# Customer Feedback Intelligence

This repository is being rebuilt from an older IMDb sentiment-analysis project into a stronger portfolio project focused on customer feedback intelligence.

The target project will combine:

- reproducible NLP benchmarking
- transformer and embedding-based modeling
- aspect and theme discovery
- review prioritization
- an inference demo or service

## Environment

This project uses `uv` for Python and dependency management.

The preferred interpreter is Python 3.12.

### Quick start

```bash
uv sync --dev
```

To include the future app dependencies:

```bash
uv sync --dev --extra app
```

Run the starter CLI:

```bash
uv run customer-feedback-intelligence --help
```

Inspect the local IMDb sample:

```bash
uv run customer-feedback-intelligence describe-dataset --sample-size 200
```

Train the transformer model and save it for inference:

```bash
uv run customer-feedback-intelligence train-transformer --config-path configs/train_roberta_imdb.json
```

This writes the model to `artifacts/models/roberta_imdb/` and training metrics to `artifacts/models/roberta_imdb_metrics.json`.

Run the first reproducible baseline benchmark:

```bash
uv run customer-feedback-intelligence run-baseline --config-path configs/tfidf_logreg_imdb.json
```

This writes a benchmark artifact to `artifacts/benchmarks/tfidf_logreg_imdb.json`.

Generate clustered review insights and review priorities:

```bash
uv run customer-feedback-intelligence analyze-reviews --config-path configs/review_analysis_imdb.json
```

This writes an analysis artifact to `artifacts/analysis/review_analysis_imdb.json`.
By default, the analysis command expects a saved transformer model in `artifacts/models/roberta_imdb/`.

Launch the demo app:

```bash
uv run customer-feedback-intelligence launch-demo --config-path configs/review_analysis_imdb.json
```

The Gradio app lets you paste multiple reviews, see sentiment confidence, theme grouping, and review priority in one interface.

## Recommended Workflow

1. Install dependencies:

```bash
uv sync --dev --extra app
```

2. Fine-tune and save the sentiment model:

```bash
uv run customer-feedback-intelligence train-transformer --config-path configs/train_roberta_imdb.json
```

3. Generate a batch analysis artifact:

```bash
uv run customer-feedback-intelligence analyze-reviews --config-path configs/review_analysis_imdb.json
```

4. Launch the demo:

```bash
uv run customer-feedback-intelligence launch-demo --config-path configs/review_analysis_imdb.json
```

## Important Paths

- Transformer training config: `configs/train_roberta_imdb.json`
- Review analysis config: `configs/review_analysis_imdb.json`
- Saved transformer model: `artifacts/models/roberta_imdb/`
- Transformer metrics: `artifacts/models/roberta_imdb_metrics.json`
- Batch analysis artifact: `artifacts/analysis/review_analysis_imdb.json`

## Current Layout

```text
aclImdb/                 Local IMDb dataset used by the current workflows
artifacts/               Generated benchmark and analysis outputs
docs/                    Project notes, archived report, and reboot plan
legacy/                  Archived code, models, and raw assets from the old project
src/                    Python package for the rebuilt project
configs/                Experiment and application configs
notebooks/archive/      Legacy notebook experiments to keep for reference
tests/                  Automated tests
```

## Legacy Material

The original notebooks, scripts, fine-tuned artifacts, and course materials have been moved into `notebooks/archive`, `legacy/`, and `docs/archive` so the active repo stays focused on the rebuilt project.
