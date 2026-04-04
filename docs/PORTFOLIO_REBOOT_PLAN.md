# Sentiment Analysis Portfolio Reboot

## What This Repository Is Today

This project is a sentiment analysis study centered on IMDb movie reviews. It contains:

- A small reusable Python core for dataset loading, preprocessing, model definitions, and train/eval helpers.
- Several exploratory notebooks for TF-IDF, BERT, DistilBERT, RoBERTa, DeBERTa, and a Mixture-of-Experts (MoE) approach.
- Local copies of the IMDb dataset and fine-tuned model artifacts.

The current repo shows curiosity and experimentation, which is valuable. The main issue is that it still looks more like a course project workspace than a polished AI/ML engineering portfolio project.

## Current Strengths

- The problem is still relevant and easy to explain.
- You explored multiple modeling families instead of stopping at one baseline.
- The MoE direction is a genuinely interesting idea for a portfolio if we make it rigorous and reproducible.
- There is enough existing work here to turn this into a stronger end-to-end project instead of starting from zero.

## Main Weaknesses Holding It Back

### 1. The repo is notebook-first, not productized

Most of the story lives in notebooks, while the reusable code is thin. That makes it hard for a recruiter or hiring manager to see engineering discipline quickly.

### 2. Reproducibility is weak

- The dataset loader randomly samples reviews without a passed seed in [imdb_dataset.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/imdb_dataset.py#L25), so runs are not stable.
- The repo has no real environment lock file and [requirements.txt](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/requirements.txt) only lists `pandas`.
- Large local artifacts are committed into the project folder, which makes the project heavy and harder to reuse.

### 3. The training utilities are not robust enough for a modern portfolio

- The train loop mixes support for tensors, tuples, and Hugging Face-style inputs, but the validation and test paths fall back to `model(x=inputs)` and assume `.to(device)` works directly on `inputs`; that breaks the promised generic behavior in [train_test.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/train_test.py#L12).
- The MoE model freezes a DistilBERT encoder and routes over shallow experts in [classifier.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/classifier.py#L168), which is fine for experimentation, but it is not yet packaged or evaluated in a way that proves engineering maturity.

### 4. Some modeling choices are historically interesting but not very persuasive now

- The LSTM-based models operate on a single vector as a fake sequence via `x.unsqueeze(1)` in [classifier.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/classifier.py#L37), [classifier.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/classifier.py#L78), and [classifier.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/classifier.py#L133). That is not how sequence modeling would normally be presented today.
- The preprocessing pipeline is classic NLP and good to keep as a baseline, but not as the headline of the project in 2026.

### 5. The portfolio story is not explicit

Right now the project does not clearly answer:

- What is the production-style objective?
- Which model is best and why?
- How do you evaluate quality beyond accuracy?
- What engineering decisions make this portfolio-worthy?

## Best Repositioning

Do not pitch this as "an old sentiment classifier project I improved a bit."

Pitch it as:

"A modern NLP benchmarking and deployment project that compares classic baselines, transformer fine-tuning, and mixture-of-experts routing for sentiment classification, with reproducible training, error analysis, and an inference demo."

That framing lets the repo showcase both data science and ML engineering.

## Recommended Target Version

The strongest reboot is a clean, modern repo with these layers:

### 1. Reproducible training pipeline

- `src/` package structure
- config-driven experiments
- deterministic seeds
- train/validate/test split logic
- artifact tracking
- clear metrics output

### 2. Benchmark suite

At minimum:

- TF-IDF + Logistic Regression
- Sentence embeddings + linear head
- One fine-tuned transformer
- One MoE-style experimental model

This gives you a strong "baseline to advanced" story.

### 3. Evaluation and analysis

- accuracy, precision, recall, F1
- confusion matrix
- calibration
- class-wise failure examples
- slice analysis by review length or polarity strength
- expert-routing analysis for the MoE model

### 4. Inference/demo layer

One of:

- a Streamlit or Gradio app
- a FastAPI inference service
- both, if you want to emphasize engineering depth

### 5. Project documentation that reads like a portfolio asset

- strong README with project narrative
- architecture diagram
- experiment table
- sample predictions
- lessons learned and future work

## What I Would Keep vs Replace

Keep:

- IMDb as the main dataset
- the baseline comparison idea
- the MoE angle
- selected visualizations and confusion matrices

Replace or refactor:

- ad hoc notebook-only workflows
- local raw dataset copies inside the main repo
- fragile training helpers
- outdated sequence-model framing
- missing dependency and experiment management

## Best Upgrade Paths

### Option A: Data Scientist portfolio version

Focus on:

- rigorous benchmarking
- better error analysis
- visual storytelling
- model comparison
- interpretability

Best if you want the repo to signal experimentation quality and analytical thinking.

### Option B: AI/ML Engineer portfolio version

Focus on:

- package structure
- config-based pipelines
- tests
- model registry/artifacts
- API or app deployment
- CI and reproducibility

Best if you want the repo to signal shipping ability.

### Option C: Hybrid version

This is my recommendation.

Build one polished benchmark + one deployable inference path. That shows both modeling skill and engineering maturity without over-scoping the project.

## Suggested Roadmap

### Phase 1: Clean foundation

- create a real repo structure: `src`, `configs`, `notebooks`, `tests`, `artifacts`
- move exploratory notebooks into `notebooks/archive`
- replace raw local dataset usage with a reproducible loader
- create a proper dependency file
- add a seed utility and consistent experiment config

### Phase 2: Strong benchmark

- implement 3 to 4 comparable baselines
- standardize metrics and logging
- create a single benchmark results table
- add error analysis notebooks or reports

### Phase 3: Signature feature

Choose one standout addition:

- a better MoE implementation with routing analysis
- retrieval-augmented sentiment explanation
- LLM-based evaluator / judge comparison
- calibration and uncertainty estimation for predictions
- lightweight online inference API

### Phase 4: Portfolio polish

- write README as a case study
- add architecture and experiment diagrams
- include example CLI commands
- include screenshots of the app or API docs

## Best First Milestone

If we want the highest return quickly, the first strong milestone is:

"Rebuild the repo as a reproducible benchmark comparing TF-IDF, sentence embeddings, and one fine-tuned transformer, then add a small demo app."

That alone would already be much stronger than the current version.

## Concrete Problems Found In The Existing Code

- [train_test.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/train_test.py#L42) handles multiple input styles during training, but [train_test.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/train_test.py#L68) and [train_test.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/train_test.py#L96) do not preserve that generality.
- [imdb_dataset.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/imdb_dataset.py#L55) samples examples randomly but does not make the randomness configurable.
- [imdb_preprocessing.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/imdb_preprocessing.py#L59) fits TF-IDF inside the transform object, which couples preprocessing and training in a way that will be awkward to scale or reuse.
- [classifier.py](/Users/macbookpro/Desktop/projet_perso/sentiment-analysis/classifier.py#L171) downloads and embeds a specific pretrained encoder directly inside the model class, which makes experimentation and deployment less flexible.

## My Recommendation

Use this project as a controlled reboot, not as a patch-up exercise.

The winning move is to preserve the core idea, but rebuild the project around:

- reproducibility
- benchmark clarity
- one strong experimental contribution
- one deployable interface

If we do that, this can become a genuinely good portfolio project.

## Next Step I Recommend

In the next pass, we should turn this plan into implementation work by doing one of these:

1. Restructure the repo into a clean modern project skeleton.
2. Define the benchmark experiments and configs we want to support.
3. Build the new README and portfolio story first, then implement toward it.

My recommendation is `1`, then `2`, then `3`.
