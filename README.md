# Customer Feedback Intelligence

This repository is my rebuild of an older IMDb sentiment-analysis project into a
clearer portfolio project focused on customer feedback intelligence.

The project has two connected goals:

- build a reproducible sentiment benchmark on IMDb
- use the saved model in a customer-feedback dashboard that can score, triage,
  and summarize uploaded reviews

At the moment, the active end-to-end path is:

1. train and evaluate a sentiment model on IMDb
2. save the model as an inference artifact
3. test zero-shot transfer on Amazon polarity reviews
4. use the saved model inside a Gradio dashboard for batch review analysis

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
uv run customer-feedback-intelligence run-baseline --config-path configs/train_tfidf_logreg_imdb.json
```

This writes a benchmark artifact to `artifacts/benchmarks/tfidf_logreg_imdb.json`
and saves the fitted sklearn pipeline to
`artifacts/models/tfidf_logreg_imdb.joblib`.

Generate clustered review insights and review priorities:

```bash
uv run customer-feedback-intelligence analyze-reviews --config-path configs/review_analysis_imdb.json
```

This writes an analysis artifact to `artifacts/analysis/review_analysis_imdb.json`.
By default, the analysis command uses the saved TF-IDF baseline model in
`artifacts/models/tfidf_logreg_imdb.joblib`.

Launch the demo app:

```bash
uv run customer-feedback-intelligence launch-demo --config-path configs/review_analysis_imdb.json
```

The Gradio app lets me paste or upload customer reviews, inspect sentiment
confidence, review priority, and lightweight theme grouping in one interface.

## Recommended Workflow

1. Install dependencies:

```bash
uv sync --dev --extra app
```

2. Run and save the reproducible baseline model:

```bash
uv run customer-feedback-intelligence run-baseline --config-path configs/train_tfidf_logreg_imdb.json
```

3. Generate a batch analysis artifact:

```bash
uv run customer-feedback-intelligence analyze-reviews --config-path configs/review_analysis_imdb.json
```

4. Launch the dashboard:

```bash
uv run customer-feedback-intelligence launch-demo --config-path configs/review_analysis_imdb.json
```

Optional: train the RoBERTa model when you have more compute available:

```bash
uv run customer-feedback-intelligence train-transformer --config-path configs/train_roberta_imdb.json
```

## IMDb Data Loading And Preprocessing

The current workflows use the local IMDb directory in `aclImdb/` and expect the
standard split layout:

```text
aclImdb/
  train/
    pos/
    neg/
  test/
    pos/
    neg/
```

The loader keeps the preprocessing intentionally light so the benchmark remains easy
to reason about:

- reviews are loaded from raw `.txt` files in the selected split
- labels are derived from the folder name: `pos -> positive`, `neg -> negative`
- `review_id` and the original IMDb rating are parsed from the filename pattern
  such as `12345_9.txt`
- sampling is deterministic with a seed and balanced across positive and negative
  classes
- if `sample_size=4000`, the loader samples `2000` positive and `2000` negative
  reviews
- the selected records are shuffled with the same seed after loading
- no HTML stripping, lowercasing, lemmatization, or stop-word removal happens at
  dataset-load time
- `word_count` is computed from whitespace tokenization and is used later for
  dataset summaries and slice metrics

The `describe-dataset` CLI command summarizes the sampled IMDb subset with:

- row count
- label distribution
- word-count min / median / mean / max

## Training Procedure

### TF-IDF + Logistic Regression Baseline

The baseline benchmark is designed to be fast, reproducible, and easy to inspect.

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

### RoBERTa Fine-Tuning

The transformer workflow fine-tunes `roberta-base` for binary sentiment
classification on the sampled IMDb reviews.

1. Load deterministic train and test samples from `aclImdb/`.
2. Split the sampled training set into train and validation subsets with
   stratified `train_test_split`.
3. Tokenize each review with a Hugging Face tokenizer using:
   - truncation enabled
   - `padding="max_length"`
   - the configured `max_length`
4. Load `AutoModelForSequenceClassification` with two labels:
   `negative` and `positive`.
5. Fine-tune the model with `AdamW` for the configured number of epochs.
6. Evaluate the final model on validation and test sets with:
   - accuracy
   - macro F1
   - full classification report
7. Save the final model and tokenizer to `artifacts/models/roberta_imdb/`.
8. Save run metadata and metrics to
   `artifacts/models/roberta_imdb_metrics.json`.

The current training implementation saves only the final checkpoint. It does not
yet save per-epoch checkpoints, a separate best-validation checkpoint, or
TensorBoard logs.

## Review Analysis And Dashboard Approach

From an application point of view, I treat sentiment classification as only one
part of the system. The dashboard is meant to answer a broader question:
"given a batch of customer feedback, which reviews should I look at first, what
is the overall sentiment mix, and what kinds of issues seem to be recurring?"

I split that workflow into a few steps.

### 1. Input Handling

The dashboard accepts:

- pasted text separated by blank lines
- CSV uploads
- TSV uploads
- JSON / JSONL / NDJSON uploads
- TXT / Markdown uploads

For uploaded structured files, I look for one main text column such as
`review_text`, `text`, `review`, `comment`, `feedback`, or `body`.

If the file also contains a title-like field such as `title`, `subject`,
`summary`, or `headline`, I concatenate it with the main review body before
inference. In other words, the model only sees text. Metadata such as `channel`,
`rating`, or timestamps is currently ignored by the predictor.

### 2. Sentiment Inference

Once I have a clean list of texts, I load the configured sentiment backend and
run batch inference.

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
uncertainty later when ranking which reviews deserve attention first.

### 3. Priority Scoring

Sentiment alone is not enough for a customer-feedback workflow, so I add a
simple triage layer on top of the predictions.

For every review, I compute a `priority_score` using:

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

The idea is straightforward: strongly negative, longer, and less certain
reviews are more likely to deserve manual review.

### 4. Theme Discovery

I also wanted the app to produce a lightweight topic-style summary of the batch,
not just individual sentiment predictions.

My current theme pipeline works like this:

1. build TF-IDF features over the uploaded review texts
2. reduce them into dense vectors with Truncated SVD
3. cluster the vectors with KMeans
4. label each cluster with its highest-weighted TF-IDF terms

This produces:

- a `theme_id` per review
- a list of representative `theme_terms` per cluster
- aggregate theme summaries such as review count, negative-rate, confidence, and
  average priority

I keep this feature in the project because it shows the full review-intelligence
approach I wanted to explore: not just classification, but also batch
summarization and emerging issue grouping.

That said, I do not treat these themes as ground-truth business taxonomies.
They are heuristic clusters derived from text similarity, not supervised issue
labels such as `billing_problem`, `delivery_delay`, or `refund_request`. On
small or noisy batches, the theme labels can be weak or overly literal.

### 5. Batch Summaries

At the dashboard level, I expose more than raw predictions.

The interface shows:

- total visible reviews
- predicted negative rate
- urgent review count
- average confidence
- top priority score
- sorted review queue
- theme-level aggregates
- saved benchmark and transfer snapshots

This matters because the app is meant to feel like an analyst tool, not just a
single-text sentiment demo.

## What The Analysis Artifact Contains

The saved analysis artifact is designed to make the workflow inspectable outside
the UI as well.

For each run, I store:

- sentiment model metadata
- analysis configuration
- theme summaries
- prioritized review rows

Each prioritized review row includes:

- `review_id`
- `predicted_label`
- `negative_probability`
- `positive_probability`
- `confidence`
- `uncertainty`
- `word_count`
- `theme_id`
- `theme_terms`
- `priority_score`
- `priority_level`
- `text_preview`

This gives me a structured record of what the app produced and how it decided to
rank the batch.

## How I Interpret The Current Limitations

The project is intentionally honest about what is strong today and what is still
heuristic.

- The sentiment benchmark is the strongest part of the repo right now.
- The dashboard triage flow is useful because it combines sentiment,
  confidence, and prioritization.
- The theme clustering is exploratory and should be read as weak structure, not
  as a final taxonomy.
- IMDb is a good benchmark dataset, but it is not the same as real customer
  feedback.
- The Amazon transfer evaluation is helpful because it shows how much of the
  sentiment signal transfers across domains without retraining.

That combination is the story I want a reader to understand: I started from
reproducible sentiment benchmarking, then built a customer-feedback analysis
layer on top of the saved model, including ranking, summarization, and
exploratory clustering.

## Important Paths

- Baseline training config: `configs/train_tfidf_logreg_imdb.json`
- Transformer training config: `configs/train_roberta_imdb.json`
- Review analysis config: `configs/review_analysis_imdb.json`
- Saved baseline model: `artifacts/models/tfidf_logreg_imdb.joblib`
- Saved transformer model: `artifacts/models/roberta_imdb/`
- Baseline benchmark artifact: `artifacts/benchmarks/tfidf_logreg_imdb.json`
- Transformer metrics: `artifacts/models/roberta_imdb_metrics.json`
- Amazon transfer artifact: `artifacts/evaluations/amazon_transfer_tfidf_imdb.json`
- Batch analysis artifact: `artifacts/analysis/review_analysis_imdb.json`
- Sample upload files: `artifacts/sample_uploads/`

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
