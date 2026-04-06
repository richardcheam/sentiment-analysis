---
title: Customer Feedback Intelligence
emoji: "📬"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.24.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Customer Feedback Intelligence

Public demo of a customer-feedback intelligence workflow built from an IMDb
sentiment benchmark and applied to batch customer-review analysis.

## What this Space does

- accepts pasted customer feedback or uploaded files
- supports CSV, TSV, JSON, JSONL, NDJSON, TXT, and Markdown
- predicts sentiment with a saved TF-IDF + Logistic Regression model
- surfaces confidence, uncertainty, manual-review flags, and priority scores
- groups reviews into exploratory theme clusters
- preserves metadata such as `review_id`, `channel`, and `product`
- lets you export the current filtered review view as CSV

## Model context

- IMDb benchmark accuracy / macro F1: `0.9015`
- Amazon polarity transfer accuracy: `0.8565`
- local 200-example customer-feedback eval accuracy: `0.8500`

This Space intentionally uses the lightweight scikit-learn baseline so the demo
stays fast and inexpensive to host on CPU.

## Notes

- Theme labels are heuristic text clusters, not supervised issue categories.
- Large uploads are supported, but bigger batches will take longer to analyze
  because inference, embeddings, and clustering all scale with batch size.
- The underlying project repository is available on GitHub:
  `https://github.com/richardcheam/customer-feedback-intelligence`
