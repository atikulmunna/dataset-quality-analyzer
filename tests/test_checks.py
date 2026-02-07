from dqa.checks.duplicates import run_exact_duplicates
from dqa.checks.leakage import run_leakage


def test_duplicates_and_leakage_ids() -> None:
    index_payload = {
        "images": [
            {"split": "train", "image": "train/images/a.jpg", "label": "train/labels/a.txt", "sha256": "abc"},
            {"split": "val", "image": "val/images/b.jpg", "label": "val/labels/b.txt", "sha256": "abc"},
        ]
    }

    dup_ids = {f.id for f in run_exact_duplicates(index_payload)}
    leak_ids = {f.id for f in run_leakage(index_payload)}

    assert "DUPLICATE_ACROSS_SPLITS" in dup_ids
    assert "LEAKAGE_EXACT_TRAIN_VAL" in leak_ids
