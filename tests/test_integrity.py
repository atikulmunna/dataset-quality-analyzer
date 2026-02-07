from dqa.checks.integrity import run_integrity


def test_integrity_missing_label_detected() -> None:
    index_payload = {
        "splits": {"train": {"orphan_labels": []}},
        "images": [
            {
                "split": "train",
                "image": "train/images/a.jpg",
                "label": None,
                "label_exists": False,
                "label_rows": [],
                "label_parse_errors": [],
                "image_error": None,
            }
        ],
    }

    findings = run_integrity(index_payload, class_count=1)
    ids = [f.id for f in findings]
    assert "INTEGRITY_MISSING_LABEL" in ids
