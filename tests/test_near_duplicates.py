from __future__ import annotations

import random

import pytest

from dqa.checks.near_duplicates import _candidate_pairs, _hamming_distance


@pytest.mark.parametrize("threshold", [0, 1, 4, 8, 16, 64])
def test_bk_tree_returns_every_true_hamming_match(threshold: int) -> None:
    rng = random.Random(20260712)
    hashes = [rng.getrandbits(64) for _ in range(200)]
    candidates = set(_candidate_pairs(hashes, threshold))
    expected = {
        (i, j)
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
        if _hamming_distance(hashes[i], hashes[j]) <= threshold
    }

    assert expected == candidates
    assert all(i < j for i, j in candidates)


def test_bk_tree_does_not_return_false_candidates() -> None:
    rng = random.Random(7)
    hashes = [rng.getrandbits(64) for _ in range(1000)]

    candidates = _candidate_pairs(hashes, threshold=8)

    assert all(_hamming_distance(hashes[i], hashes[j]) <= 8 for i, j in candidates)
