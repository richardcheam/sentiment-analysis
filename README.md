# Customer Feedback Intelligence

This repository is my rebuild of an older IMDb sentiment-analysis project into a
clearer end-to-end portfolio project for customer feedback intelligence.

The project has two connected goals:

- build a reproducible sentiment benchmark on IMDb
- use the saved model in a customer-feedback dashboard that can score, triage,
  summarize, and export uploaded review batches

## Application Features

The application is not just a single-text sentiment demo. I designed it as a
batch review-analysis dashboard.

It currently supports:

- full-batch upload or paste workflows
- CSV, TSV, JSON, JSONL, NDJSON, TXT, and Markdown inputs
- sentiment scoring for every uploaded review
- confidence and uncertainty tracking
- manual-review flags for low-confidence or high-uncertainty cases
- priority scoring so the most urgent reviews rise to the top
- metadata preservation from uploads such as `review_id`, `channel`,
  `product`, and `created_at`
- exploratory theme clustering with representative review previews
- benchmark snapshot cards for the IMDb benchmark and Amazon transfer check
- CSV export of the currently filtered dashboard view

In practical terms, the app is meant to answer questions like:

- what is the current sentiment mix in this batch?
- which reviews should I read first?
- which items should be manually checked by a human?
- are certain channels or products contributing more negative feedback?
- what recurring issue clusters seem to be present in the batch?

## Current Project State

The strongest active path in the repo today is:

1. train and evaluate a reproducible TF-IDF + Logistic Regression sentiment
   model on IMDb
2. save that model as a reusable inference artifact
3. test zero-shot transfer on Amazon polarity reviews
4. test on a fixed local 200-example customer-feedback evaluation set
5. use the saved model inside a Gradio dashboard for customer-feedback batch
   analysis

The RoBERTa training path is also implemented, but because it is more expensive
to run, the current default dashboard model is the saved TF-IDF baseline.

## Environment

This project uses `uv` for Python and dependency management.

The preferred interpreter is Python 3.12.

### Quick Start

Install dependencies:

```bash
uv sync --dev --extra app
```

Inspect the available CLI:

```bash
uv run customer-feedback-intelligence --help
```

Run the IMDb baseline and save the inference artifact:

```bash
uv run customer-feedback-intelligence run-baseline --config-path configs/train_tfidf_logreg_imdb.json
```

Evaluate zero-shot transfer on Amazon polarity:

```bash
uv run customer-feedback-intelligence evaluate-amazon-transfer --config-path configs/evaluate_amazon_transfer_tfidf.json
```

Evaluate the same saved model on the fixed local 200-example customer-feedback set:

```bash
uv run customer-feedback-intelligence evaluate-local-feedback --config-path configs/evaluate_local_feedback_eval_200.json
```

Generate a saved analysis artifact from sampled IMDb reviews:

```bash
uv run customer-feedback-intelligence analyze-reviews --config-path configs/review_analysis_imdb.json
```

Launch the dashboard:

```bash
uv run customer-feedback-intelligence launch-demo --config-path configs/review_analysis_imdb.json
```

Optional: fine-tune RoBERTa when more compute is available:

```bash
uv run customer-feedback-intelligence train-transformer --config-path configs/train_roberta_imdb.json
```

## Recommended Workflow

1. Install dependencies:

```bash
uv sync --dev --extra app
```

2. Train and save the IMDb baseline:

```bash
uv run customer-feedback-intelligence run-baseline --config-path configs/train_tfidf_logreg_imdb.json
```

3. Measure transfer on customer-feedback-style data:

```bash
uv run customer-feedback-intelligence evaluate-amazon-transfer --config-path configs/evaluate_amazon_transfer_tfidf.json
```

```bash
uv run customer-feedback-intelligence evaluate-local-feedback --config-path configs/evaluate_local_feedback_eval_200.json
```

4. Launch the dashboard and upload customer feedback:

```bash
uv run customer-feedback-intelligence launch-demo --config-path configs/review_analysis_imdb.json
```

5. Optionally train the RoBERTa model later:

```bash
uv run customer-feedback-intelligence train-transformer --config-path configs/train_roberta_imdb.json
```

## IMDb Data Loading And Preprocessing

The current benchmark workflows use the local IMDb directory in `aclImdb/` and
expect the standard split layout:

```text
aclImdb/
  train/
    pos/
    neg/
  test/
    pos/
    neg/
```

I keep the preprocessing intentionally light so the benchmark remains easy to
reason about:

- reviews are loaded from raw `.txt` files in the selected split
- labels are derived from the folder name: `pos -> positive`, `neg -> negative`
- `review_id` and the original IMDb rating are parsed from the filename pattern
  such as `12345_9.txt`
- sampling is deterministic with a seed and balanced across positive and
  negative classes
- if `sample_size=4000`, the loader samples `2000` positive and `2000`
  negative reviews
- the selected records are shuffled with the same seed after loading
- no HTML stripping, lowercasing, lemmatization, or stop-word removal happens at
  dataset-load time
- `word_count` is computed from whitespace tokenization and is later reused for
  summaries and priority scoring

The `describe-dataset` CLI command summarizes the sampled IMDb subset with:

- row count
- label distribution
- word-count min / median / mean / max

## Training Procedure

### TF-IDF + Logistic Regression Baseline

The baseline benchmark is designed to be fast, reproducible, and easy to
inspect.

1. Load deterministic balanced train and test samples from `aclImdb/`.
2. Split the sampled training set into train and validation subsets with
   stratified `train_test_split`.
3. Fit a `TfidfVectorizer` with the configured `max_features`, `min_df`, and
   n-gram range.
4. Train a `LogisticRegression` classifier on the TF-IDF features.
5. Evaluate on validation and test sets with:
   - accuracy
   - macro F1
   - full classification report
   - review-length slice metrics
   - top high-confidence error examples
6. Save the benchmark artifact to
   `artifacts/benchmarks/tfidf_logreg_imdb.json`.
7. Save the fitted sklearn pipeline to
   `artifacts/models/tfidf_logreg_imdb.joblib`.

This is the current default inference artifact for the dashboard.

### RoBERTa Fine-Tuning

The transformer workflow fine-tunes `roberta-base` for binary sentiment
classification on sampled IMDb reviews.

1. Load deterministic train and test samples from `aclImdb/`.
2. Split the sampled training set into train and validation subsets with
   stratified `train_test_split`.
3. Seed Python, NumPy, and Torch so the run is more reproducible.
4. Tokenize each review with a Hugging Face tokenizer using:
   - truncation enabled
   - `padding="max_length"`
   - the configured `max_length`
5. Load `AutoModelForSequenceClassification` with two labels:
   `negative` and `positive`.
6. Fine-tune the model with `AdamW` for the configured number of epochs.
7. Evaluate on the validation split after every epoch.
8. Save the best validation checkpoint to `artifacts/models/roberta_imdb/`.
9. Evaluate that saved checkpoint on the test split.
10. Save run metadata and metrics, including per-epoch history, to
    `artifacts/models/roberta_imdb_metrics.json`.

This path is implemented and ready, but it is not the current default demo
backend because of compute cost.

## Customer Feedback Dashboard Approach

From an application point of view, I treat sentiment classification as only one
part of the system. The dashboard is meant to answer a broader question:
"given a batch of customer feedback, which reviews should I look at first, what
is the overall sentiment mix, what needs manual review, and what issue patterns
seem to be emerging?"

### 1. Input Handling

The dashboard accepts:

- pasted text separated by blank lines
- CSV uploads
- TSV uploads
- JSON / JSONL / NDJSON uploads
- TXT / Markdown uploads

For uploaded structured files, I look for one main text column such as
`review_text`, `text`, `review`, `comment`, `feedback`, `message`, or `body`.

If the file also contains a title-like field such as `title`, `subject`,
`summary`, or `headline`, I concatenate it with the main review body before
inference.

The predictor sees text only, but the dashboard preserves metadata from the
uploaded file when possible. That means fields like `review_id`, `channel`,
`product`, and `created_at` can still be used for slicing, display, and export
even though they are not part of the sentiment model input.

### 2. Full-Batch Analysis

The dashboard now keeps the full uploaded batch in the analysis artifact.

That means:

- KPIs are computed from the full batch
- summaries reflect the full batch
- filters and sorting operate over the full batch
- exports include the full currently visible filtered batch

I do not silently clip the uploaded reviews to a smaller top-N queue anymore.
If a user uploads 20, 200, or more reviews, the dashboard keeps that full set.

The tradeoff is that very large uploads will take longer because sentiment
inference, embedding generation, and clustering all scale with batch size.

### 3. Sentiment Inference

Once I have the review texts, I load the configured sentiment backend and run
batch inference.

Right now the default app configuration uses the saved
`TF-IDF + Logistic Regression` model. The same inference interface also supports
the transformer path when a saved RoBERTa checkpoint is available.

For each review, I store:

- predicted label
- negative probability
- positive probability
- confidence
- uncertainty

This gives me more than a hard class label. I can use confidence and
uncertainty later when ranking and triaging the batch.

### 4. Manual Review And Priority Scoring

Sentiment alone is not enough for a customer-feedback workflow, so I add a
simple triage layer on top of the predictions.

First, I flag reviews for manual review when:

- confidence is below the configured threshold
- uncertainty is above the configured threshold

Then I compute a `priority_score` using:

- negative probability
- review length
- uncertainty

The current weighting is:

- `0.65 * negative_probability`
- `0.20 * normalized_review_length`
- `0.15 * uncertainty`

I then map that numeric score to a priority bucket:

- `urgent`
- `high`
- `medium`
- `low`

The goal is straightforward: strongly negative, longer, and less certain
reviews are more likely to deserve attention first.

### 5. Theme Discovery

I also wanted the app to produce a lightweight topic-style summary of the batch,
not just individual sentiment predictions.

My current theme pipeline works like this:

1. build TF-IDF features over the uploaded review texts
2. reduce them into dense vectors with Truncated SVD
3. cluster the vectors with KMeans
4. label each cluster with its highest-weighted TF-IDF terms
5. pick a representative review close to each cluster centroid

This produces:

- a `theme_id` per review
- a human-readable `theme_label`
- a keyword signature per cluster
- representative review previews
- aggregate theme summaries such as review count, negative rate, manual-review
  count, and average priority

I keep this feature because it shows the broader review-intelligence approach I
wanted to explore: not just classification, but also batch summarization and
emerging issue grouping.

That said, I do not treat these themes as ground-truth business taxonomies.
They are heuristic clusters derived from text similarity, not supervised issue
labels such as `billing_problem`, `delivery_delay`, or `refund_request`. On
small or noisy batches, the theme labels can still be weak or overly literal.

### 6. Dashboard Outputs

At the dashboard level, I expose more than raw predictions.

The interface shows:

- full-batch review counts
- predicted negative rate
- urgent review count
- manual-review count
- average confidence
- top priority score
- metadata-aware breakdowns such as channel and product
- a sortable review queue
- theme-cluster summaries
- saved benchmark and transfer snapshots
- downloadable CSV export of the current filtered view

This matters because the app is meant to feel like an analyst tool, not just a
single-text sentiment demo.

## Evaluation Beyond IMDb

IMDb is the benchmark dataset, but I also wanted to know how far that benchmark
transfers toward a customer-feedback use case.

### Amazon Transfer Evaluation

I added a zero-shot transfer evaluation on Amazon polarity reviews. This lets me
measure how well a model trained on IMDb generalizes to real product-review
language without retraining.

The artifact is written to:

- `artifacts/evaluations/amazon_transfer_tfidf_imdb.json`

### Fixed Local Customer-Feedback Evaluation Set

I also added a fixed local labeled evaluation set with 200 Amazon customer
reviews:

- `data/eval/customer_feedback_amazon_eval_200.csv`

This gives me a stable local customer-feedback check that I can rerun without
depending on a remote dataset pull every time.

The evaluation artifact is written to:

- `artifacts/evaluations/customer_feedback_eval_200.json`

## What The Analysis Artifact Contains

The saved analysis artifact is meant to make the workflow inspectable outside
the UI as well.

For each run, I store:

- sentiment model metadata
- analysis configuration
- theme summaries
- full review rows for the analyzed batch

Each review row includes:

- `review_id`
- `predicted_label`
- `negative_probability`
- `positive_probability`
- `confidence`
- `uncertainty`
- `word_count`
- `theme_id`
- `theme_label`
- `theme_terms`
- `requires_manual_review`
- `manual_review_reason`
- `review_status`
- `priority_score`
- `priority_level`
- `text_preview`
- `text`
- `metadata`

This gives me a structured record of what the app produced and how it decided to
rank and summarize the batch.

## How I Interpret The Current Limitations

The project is intentionally honest about what is strong today and what is still
heuristic.

- The sentiment benchmark is the strongest part of the repo right now.
- The dashboard triage flow is useful because it combines sentiment,
  confidence, manual-review gating, and prioritization.
- The theme clustering is exploratory and should be read as weak structure, not
  as a final issue taxonomy.
- Metadata is preserved for slicing and display, but it is not yet part of the
  model itself.
- IMDb is a good benchmark dataset, but it is not the same as real customer
  feedback.
- The Amazon transfer evaluation and local 200-example evaluation set are there
  to keep that limitation visible.
- Full-batch uploads are supported, but runtime will naturally grow with batch
  size.

That combination is the story I want a reader to understand: I started from
reproducible sentiment benchmarking, then built a customer-feedback analysis
layer on top of the saved model, including ranking, manual-review gating,
summarization, metadata-aware slicing, export, and exploratory clustering.

## Important Paths

- Baseline training config: `configs/train_tfidf_logreg_imdb.json`
- Transformer training config: `configs/train_roberta_imdb.json`
- Review analysis config: `configs/review_analysis_imdb.json`
- Amazon transfer config: `configs/evaluate_amazon_transfer_tfidf.json`
- Local evaluation config: `configs/evaluate_local_feedback_eval_200.json`
- Saved baseline model: `artifacts/models/tfidf_logreg_imdb.joblib`
- Saved transformer model: `artifacts/models/roberta_imdb/`
- Baseline benchmark artifact: `artifacts/benchmarks/tfidf_logreg_imdb.json`
- Transformer metrics: `artifacts/models/roberta_imdb_metrics.json`
- Amazon transfer artifact: `artifacts/evaluations/amazon_transfer_tfidf_imdb.json`
- Local feedback evaluation artifact:
  `artifacts/evaluations/customer_feedback_eval_200.json`
- Batch analysis artifact: `artifacts/analysis/review_analysis_imdb.json`
- Dashboard exports: `artifacts/exports/`
- Fixed local evaluation set: `data/eval/customer_feedback_amazon_eval_200.csv`
- Sample upload files: `artifacts/sample_uploads/`

## Current Layout

```text
aclImdb/                 Local IMDb dataset used by the benchmark workflows
artifacts/               Generated benchmark, evaluation, analysis, and export outputs
configs/                 Experiment and application configs
data/eval/               Fixed local evaluation datasets
docs/                    Project notes, archived report, and reboot plan
legacy/                  Archived code, models, and raw assets from the old project
src/                     Python package for the rebuilt project
tests/                   Automated tests
```

## Legacy Material

The original notebooks, scripts, fine-tuned artifacts, and course materials
have been moved into `legacy/` and `docs/archive` so the active repo stays
focused on the rebuilt project.
