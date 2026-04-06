"""Gradio entrypoint for the public Hugging Face Space bundle."""

from __future__ import annotations

from pathlib import Path
import sys

SPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SPACE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feedback_intelligence.app.gradio_app import create_demo
from feedback_intelligence.config import ReviewAnalysisConfig

CONFIG_PATH = SPACE_ROOT / "configs" / "review_analysis_space.json"

demo = create_demo(
    base_path=SPACE_ROOT,
    analysis_config=ReviewAnalysisConfig.from_json(CONFIG_PATH),
)

if __name__ == "__main__":
    demo.launch()
