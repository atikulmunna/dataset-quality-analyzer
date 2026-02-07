# DQA v1 Specification

Version: 1.0.0-draft  
Date: 2026-02-06

## 1) Scope

DQA v1 audits YOLO detection datasets without model training and outputs deterministic, machine-readable findings plus an HTML report.

In-scope checks:
- Integrity checks (files, labels, class IDs, annotation format)
- Class distribution and split drift
- Bounding-box sanity checks
- Exact duplicate detection
- Optional near-duplicate detection
- Split leakage detection

Out of scope for v1:
- Semantic label correctness (human-in-the-loop tasks)
- Segmentation/keypoint formats
- Auto-fixing labels

## 2) CLI Contract

Primary command:
```bash
dqa audit --data /path/to/data.yaml --out runs/dqa_001
```

Recommended flags for v1:
```bash
--config dqa.yaml
--splits train,val,test
--workers 8
--max-images 0
--near-dup
--format html,json
--fail-on high
```

Exit codes:
- 0: Completed, no findings at or above fail threshold
- 1: Completed, findings at or above fail threshold
- 2: Usage/config error (bad args, invalid config)
- 3: Runtime/data access error (cannot proceed)

## 3) Configuration (dqa.yaml)

```yaml
version: 1

fail_on: high

checks:
  integrity:
    enabled: true
  class_distribution:
    enabled: true
    min_instances_per_class_warn: 50
    max_class_share_warn: 0.80
    split_drift_jsd_warn: 0.10
    split_drift_jsd_high: 0.20
  bbox_sanity:
    enabled: true
    min_box_area_ratio_warn: 0.0001
    max_box_area_ratio_warn: 0.90
    max_boxes_per_image_warn: 300
    aspect_ratio_warn: 20.0
  duplicates:
    enabled: true
  near_duplicates:
    enabled: false
    phash_hamming_threshold: 8
  leakage:
    enabled: true
    include_near_dup: false
```

Notes:
- Unknown keys must fail config validation.
- All thresholds are deterministic and explicitly logged in `summary.json`.

## 4) Severity Model

Severities:
- critical: blocks training immediately
- high: likely harmful; block in CI for production datasets
- medium: quality risk; triage required
- low: informational

Default fail threshold:
- `high`

Default severities by finding ID are frozen in `FINDING_CATALOG.md`.

## 5) Output Schemas

### 5.1 summary.json

```json
{
  "schema_version": "1.0.0",
  "run": {
    "run_id": "20260206_230000",
    "dqa_version": "1.0.0",
    "started_at": "2026-02-06T23:00:00Z",
    "finished_at": "2026-02-06T23:04:20Z",
    "duration_sec": 260.2,
    "config": {
      "fail_on": "high",
      "enabled_checks": ["integrity", "class_distribution", "bbox_sanity", "duplicates", "leakage"]
    }
  },
  "dataset": {
    "data_yaml": "/path/to/data.yaml",
    "root": "/path/to/dataset",
    "splits": {
      "train": {"images": 10234, "labeled": 10021, "unlabeled": 213},
      "val": {"images": 1250, "labeled": 1210, "unlabeled": 40},
      "test": {"images": 0, "labeled": 0, "unlabeled": 0}
    },
    "classes": {
      "count": 18,
      "names": ["class_0", "class_1"]
    }
  },
  "checks": {
    "integrity": {"status": "completed", "counts": {"critical": 2, "high": 8, "medium": 0, "low": 0}},
    "class_distribution": {"status": "completed", "counts": {"critical": 0, "high": 0, "medium": 4, "low": 3}},
    "bbox_sanity": {"status": "completed", "counts": {"critical": 0, "high": 2, "medium": 11, "low": 7}},
    "duplicates": {"status": "completed", "counts": {"critical": 0, "high": 1, "medium": 0, "low": 0}},
    "leakage": {"status": "completed", "counts": {"critical": 3, "high": 2, "medium": 0, "low": 0}}
  },
  "totals": {
    "findings": 43,
    "by_severity": {"critical": 5, "high": 13, "medium": 15, "low": 10},
    "fail_threshold": "high",
    "build_failed": true
  }
}
```

### 5.2 flags.json

`flags.json` contains schema version and a list of atomic findings.

```json
{
  "schema_version": "1.0.0",
  "findings": [
    {
      "id": "INTEGRITY_MISSING_LABEL",
      "severity": "high",
      "split": "train",
      "image": "train/images/img_001.jpg",
      "label": "train/labels/img_001.txt",
      "message": "Image has no matching label file.",
      "metrics": {},
      "suggested_action": "Add label file or move image to unlabeled pool.",
      "fingerprint": "sha1:2f7d..."
    }
  ]
}
```

Required fields per finding:
- `id` (stable check code from `FINDING_CATALOG.md`)
- `severity`
- `message`
- `fingerprint` (stable dedupe key)

Recommended fields:
- `split`, `image`, `label`, `class_id`, `metrics`, `suggested_action`

### 5.3 index.json

Contains cached deterministic metadata for reruns:
- file path, size, mtime, sha256 (optional lazy), image dimensions, split
- parsed label rows and parse errors

Cache key must include:
- dataset root path
- file stats snapshot
- dqa version
- enabled checks and relevant thresholds

## 6) Check Definitions

### R1 Integrity

Detect:
- Missing label for image
- Orphan label without image
- Invalid class ID (<0 or >= num_classes)
- Malformed rows (not 5 tokens, non-numeric)
- Normalized coords/size out of [0,1]
- Corrupt images that cannot be decoded

### R2 Class Distribution

Compute:
- Per-class instance count
- Per-split class proportion
- Long-tail ratio
- Split drift via Jensen-Shannon divergence

Flags:
- `CLASS_IMBALANCE_HIGH` when dominant class share > threshold
- `CLASS_LOW_SUPPORT` when class instances < threshold
- `CLASS_SPLIT_DRIFT` using JSD thresholds

### R3 Bounding Box Sanity

Checks:
- Tiny boxes below area ratio threshold
- Oversized boxes above area ratio threshold
- Extreme aspect ratios
- Too many boxes per image

### R4 Exact Duplicates

Method:
- Content hash (sha256) of image bytes

Flags:
- `DUPLICATE_WITHIN_SPLIT`
- `DUPLICATE_ACROSS_SPLITS`

### R5 Near Duplicates (optional)

Method:
- pHash with configurable Hamming threshold

Flags:
- `NEAR_DUPLICATE_WITHIN_SPLIT`
- `NEAR_DUPLICATE_ACROSS_SPLITS`

### R6 Leakage

Detect:
- Exact duplicate leakage between train and val/test
- Optional near-duplicate leakage

Flags:
- `LEAKAGE_EXACT_TRAIN_VAL`
- `LEAKAGE_EXACT_TRAIN_TEST`
- `LEAKAGE_NEAR_TRAIN_VAL`
- `LEAKAGE_NEAR_TRAIN_TEST`

## 7) CI Gating Policy

Recommended default policy (`fail_on: high`):
- Fail on any `critical` or `high`
- Pass with warnings if only `medium`/`low`

Strict policy (`fail_on: medium`):
- Fail on `critical`, `high`, `medium`

Release policy (`fail_on: critical`):
- Fail only on `critical`

## 8) Determinism and Performance

Determinism requirements:
- Stable ordering of outputs (sort by `id`, `split`, `path`)
- Stable fingerprint generation for same underlying issue
- Fixed random seeds where sampling is used

Performance targets for v1:
- 100k images, exact duplicate + core checks, 8 workers: target <= 20 min on modern 8-core machine
- Re-run with warm cache: target >= 40% faster

## 9) Test Strategy

Minimum test suite:
- Unit tests for each check module
- Parser tests for malformed YOLO rows
- Snapshot tests for `summary.json` and `flags.json` schema compliance
- Integration tests with golden datasets:
  - clean dataset
  - leakage dataset
  - malformed labels dataset
  - heavy imbalance dataset

Required CI assertions:
- CLI exit code matrix
- deterministic outputs on repeated runs
- no crash on malformed data

## 10) Implementation Plan (Reordered)

1. CLI + config parser + schema models
2. Indexer + integrity checks
3. Exact duplicates + leakage
4. Class distribution + bbox sanity
5. Reporter (HTML + JSON)
6. Near-duplicate optional module
7. Hardening, benchmarks, v1.0 tag

## 11) Non-Goals for v1.0

- Auto-correction of labels
- Distributed execution across multiple machines
- Active-learning recommendation engine

## 12) Versioning

- Semantic versioning for CLI and output schema
- Breaking schema changes require major bump
- Include schema version in every JSON artifact
