"""Shared data structures for the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ReviewRecord:
    """One labeled review example."""

    review_id: str
    text: str
    label: str
    split: str
    source: str
    score: int | None = None
    metadata: dict[str, Any] | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
