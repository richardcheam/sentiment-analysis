from pathlib import Path

from feedback_intelligence.data.local_reviews import load_local_labeled_reviews


def test_load_local_labeled_reviews_parses_text_and_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "customer_feedback_eval.csv"
    dataset_path.write_text(
        "review_id,title,review_text,label,channel,product\n"
        '"eval-1","Late delivery","Terrible shipping delay.","negative","email","router"\n'
        '"eval-2","Helpful support","The team resolved it fast.","positive","chat","router"\n',
        encoding="utf-8",
    )

    records = load_local_labeled_reviews(dataset_path=dataset_path)

    assert len(records) == 2
    assert records[0].review_id == "eval-1"
    assert records[0].label == "negative"
    assert "Late delivery" in records[0].text
    assert records[0].metadata == {"channel": "email", "product": "router"}
