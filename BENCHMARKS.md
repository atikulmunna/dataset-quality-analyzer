# Development Benchmarks

These measurements are engineering checks, not service-level guarantees. Results depend on CPU, storage, image size, similarity, and cache state.

Environment: Windows, Python 3.13, local filesystem, measured 2026-07-12.

## Hash worker check

Fixture: 1,000 PNG files, 141.2 MiB total, checks disabled to isolate indexing. Both runs used new output directories; the filesystem cache was already warm to reduce run-order bias.

| Workers | Time |
|---:|---:|
| 1 | 1.07 s |
| 4 | 0.96 s |

Observed improvement: 1.12x. Tiny or already cached files may see less benefit; four workers is therefore the conservative default and the CLI caps the value at 32.

## Near-duplicate candidate search

Fixture: deterministic uniformly random 64-bit hashes, Hamming threshold 8, exact BK-tree search. Image decoding/hash generation is excluded.

| Hashes | Search time | Matching pairs |
|---:|---:|---:|
| 100 | 0.003 s | 0 |
| 1,000 | 0.297 s | 0 |
| 5,000 | 8.309 s | 0 |

A 25,000-hash development run exceeded 70 seconds and was stopped. Hosted-alpha near-duplicate analysis therefore remains capped at 5,000 images even though core audits allow 25,000. Similar-image collections can take longer and emit many pairs.

## Incremental index check

Fixture: 2,000 tiny YOLO images and labels.

| Run | Time | Cache result |
|---|---:|---|
| Cold | 4.66 s | 2,000 misses |
| Warm | 1.12 s | 2,000 hits |

Observed warm-run improvement: 4.18x.
