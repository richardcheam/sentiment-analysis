from __future__ import annotations

from feedback_intelligence.data.amazon_reviews import (
    load_amazon_polarity_reviews,
    summarize_reviews,
)


class FakeStreamingDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def shuffle(self, seed: int, buffer_size: int) -> FakeStreamingDataset:
        del seed, buffer_size
        return self

    def __iter__(self):
        return iter(self.rows)


def _fake_load_dataset(dataset_name: str, split: str, streaming: bool = False):
    del dataset_name, split
    rows = [
        {"label": 0, "title": "Bad item", "text": "Stopped working after a day."},
        {"label": 1, "title": "Great item", "text": "Works perfectly and feels sturdy."},
        {"label": 0, "title": "Poor quality", "text": "Cheap plastic and disappointing."},
        {"label": 1, "title": "Very happy", "text": "Would buy again without hesitation."},
    ]
    if streaming:
        return FakeStreamingDataset(rows)
    return rows


def test_load_amazon_polarity_reviews_balances_labels() -> None:
    records = load_amazon_polarity_reviews(
        split="test",
        sample_size=4,
        seed=7,
        load_dataset_fn=_fake_load_dataset,
    )

    assert len(records) == 4
    assert {record.label for record in records} == {"positive", "negative"}
    assert all(record.source == "SetFit/amazon_polarity" for record in records)
    assert any("Bad item" in record.text for record in records)


def test_summarize_reviews_returns_basic_stats_for_amazon_records() -> None:
    records = load_amazon_polarity_reviews(
        split="train",
        sample_size=4,
        seed=3,
        load_dataset_fn=_fake_load_dataset,
    )
    summary = summarize_reviews(records)

    assert summary["rows"] == 4
    assert summary["label_distribution"] == {"negative": 2, "positive": 2}
