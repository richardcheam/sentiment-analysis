"""Build a self-contained Hugging Face Space bundle from the local project."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BundleItem:
    """One file or directory to copy into the Space bundle."""

    source: str
    destination: str
    required: bool = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "deploy" / "huggingface_space"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "deploy" / "huggingface-space"

BUNDLE_ITEMS = [
    BundleItem("deploy/huggingface_space/app.py", "app.py"),
    BundleItem("deploy/huggingface_space/README.md", "README.md"),
    BundleItem("deploy/huggingface_space/requirements.txt", "requirements.txt"),
    BundleItem("configs/review_analysis_space.json", "configs/review_analysis_space.json"),
    BundleItem("src/feedback_intelligence", "src/feedback_intelligence"),
    BundleItem("artifacts/models/tfidf_logreg_imdb.joblib", "artifacts/models/tfidf_logreg_imdb.joblib"),
    BundleItem("artifacts/benchmarks/tfidf_logreg_imdb.json", "artifacts/benchmarks/tfidf_logreg_imdb.json"),
    BundleItem(
        "artifacts/evaluations/amazon_transfer_tfidf_imdb.json",
        "artifacts/evaluations/amazon_transfer_tfidf_imdb.json",
        required=False,
    ),
    BundleItem(
        "artifacts/evaluations/customer_feedback_eval_200.json",
        "artifacts/evaluations/customer_feedback_eval_200.json",
        required=False,
    ),
    BundleItem("artifacts/sample_uploads", "artifacts/sample_uploads", required=False),
    BundleItem("LICENSE", "LICENSE", required=False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face Space deployment bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the bundle will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []

    for item in BUNDLE_ITEMS:
        source_path = PROJECT_ROOT / item.source
        destination_path = output_dir / item.destination

        if not source_path.exists():
            if item.required:
                raise FileNotFoundError(
                    f"Required bundle asset is missing: {source_path}"
                )
            skipped.append(item.source)
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        copied.append(item.destination)

    deployment_notes = _build_deployment_notes(output_dir)
    notes_path = output_dir / "SPACE_DEPLOYMENT.md"
    notes_path.write_text(deployment_notes, encoding="utf-8")

    print(f"Built Hugging Face Space bundle at: {output_dir}")
    print("Copied assets:")
    for path in copied:
        print(f"  - {path}")
    if skipped:
        print("Skipped optional assets:")
        for path in skipped:
            print(f"  - {path}")
    print("Next step: push the bundle contents to a new Gradio Space repository.")


def _build_deployment_notes(output_dir: Path) -> str:
    return f"""# Hugging Face Space Deployment

This folder is a self-contained Space bundle generated from the main project.

## Recommended next steps

1. Create a new public Gradio Space on Hugging Face.
2. Copy the contents of this folder into the root of that Space repository.
3. Commit and push.
4. Once the Space is live, add the public `hf.space` URL back into your
   portfolio site.

## Bundle path

`{output_dir}`

## Important files

- `README.md`: Hugging Face Space metadata and public-facing overview
- `app.py`: Gradio entrypoint
- `requirements.txt`: lightweight CPU-friendly dependencies
- `configs/review_analysis_space.json`: public demo configuration
- `artifacts/models/tfidf_logreg_imdb.joblib`: saved inference artifact
"""


if __name__ == "__main__":
    main()
