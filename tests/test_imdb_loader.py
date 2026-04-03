from pathlib import Path

from feedback_intelligence.data.imdb import load_local_imdb_reviews, summarize_reviews


def test_local_imdb_loader_is_deterministic(tmp_path: Path) -> None:
    _write_fake_imdb_split(tmp_path, "train", "pos", count=4, score=8)
    _write_fake_imdb_split(tmp_path, "train", "neg", count=4, score=2)

    first = load_local_imdb_reviews(tmp_path, split="train", sample_size=4, seed=7)
    second = load_local_imdb_reviews(tmp_path, split="train", sample_size=4, seed=7)

    assert [record.review_id for record in first] == [record.review_id for record in second]
    assert {record.label for record in first} == {"positive", "negative"}


def test_summarize_reviews_returns_basic_stats(tmp_path: Path) -> None:
    _write_fake_imdb_split(tmp_path, "test", "pos", count=2, score=9)
    _write_fake_imdb_split(tmp_path, "test", "neg", count=2, score=1)

    records = load_local_imdb_reviews(tmp_path, split="test", sample_size=4, seed=3)
    summary = summarize_reviews(records)

    assert summary["rows"] == 4
    assert summary["label_distribution"] == {"positive": 2, "negative": 2}


def _write_fake_imdb_split(
    root: Path,
    split: str,
    label_dir: str,
    count: int,
    score: int,
) -> None:
    target = root / split / label_dir
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        text = f"{label_dir} review {index} with a few tokens"
        (target / f"{index}_{score}.txt").write_text(text, encoding="utf-8")
