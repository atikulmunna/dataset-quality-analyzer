# DQA Project Wiki (Newcomer Guide)

This wiki explains the Dataset Quality Analyzer (DQA) project end-to-end: what it does, why it exists, how it is structured, how data flows through the pipeline, how to operate it, and what challenges were encountered and mitigated.

## 1. What This Project Is

DQA is a read-only dataset quality auditing tool focused on computer-vision training data quality before model training.

Core goals:

- Catch data quality issues early (before expensive training runs)
- Generate machine-readable artifacts for automation and CI
- Provide human-readable summaries for decision-making
- Support practical dataset workflows (local and remote ingestion)

Primary user outcomes:

- Determine if a dataset is safe to train on
- Understand which issue types dominate quality risk
- Compare two dataset versions and detect regressions
- Gate merges in CI based on configurable severity

## 2. What DQA Checks

DQA evaluates six quality dimensions:

1. Integrity
2. Class distribution
3. Bounding-box sanity
4. Exact duplicates
5. Near duplicates (optional)
6. Leakage between splits

### 2.1 Integrity checks

Looks for structural/data correctness problems, for example:

- Missing label files
- Orphan label files (no matching image)
- Malformed annotation rows
- Invalid class IDs
- Out-of-range normalized coordinates
- Corrupt image files

### 2.2 Class distribution checks

Focuses on label-balance quality:

- Class under-support (`CLASS_LOW_SUPPORT`)
- Dominant class skew
- Split drift (train vs val/test distribution divergence)

### 2.3 Bounding-box sanity

Heuristic checks on geometric plausibility:

- Tiny boxes
- Oversized boxes
- Extreme aspect ratio
- Too many boxes per image

Important note: For segmentation datasets, polygon-derived bounding boxes may trigger many bbox findings. This is why segmentation-tuned configs are included.

### 2.4 Duplicate and leakage checks

- Duplicate content within/across splits
- Train/val and train/test contamination detection (critical for trustworthy validation)

### 2.5 Near-duplicate checks

- Optional perceptual duplicate detection using pHash
- Useful for overfitting risk, but can be expensive/noisy depending on dataset

## 3. Supported Dataset Inputs

DQA currently supports:

- YOLO datasets via `data.yaml`
- YOLO segmentation rows (polygon format)
- COCO-style annotation JSON (split-wise)
- Roboflow URLs via provider adapter (with API key)

This gives practical flexibility for real-world data pipelines.

## 4. High-Level Architecture

Top-level folders/files:

- `dqa/`: core library and CLI
- `schemas/`: JSON contract schemas for artifacts
- `tests/`: unit/integration tests
- `dqa.yaml`, `dqa_seg.yaml`, `dqa_seg_low_noise.yaml`: configuration presets
- `web_dashboard.py`: local browser UI wrapper over CLI
- `.github/workflows/dqa-ci.yml`: CI workflow

### 4.1 Core package structure (`dqa/`)

- `cli.py`: command entrypoint and orchestration
- `config.py`: config loading/validation
- `indexer.py`: dataset indexing (YOLO + COCO)
- `io_yolo.py`: YOLO spec loading
- `remote.py`: remote source routing
- `providers/roboflow.py`: Roboflow ingestion/download/cache logic
- `checks/`: each quality check module
- `models/`: finding and summary data models
- `report/html.py`, `report/json_writer.py`: report/artifact emitters

## 5. Pipeline: How a Run Works Internally

The main run path is `python -m dqa audit ...`.

### Step 1: Parse CLI arguments

`dqa/cli.py` builds subcommands:

- `audit`
- `explain`
- `validate`
- `diff`

### Step 2: Resolve dataset source

- If `--data` is used: local path
- If `--data-url` is used: provider logic (currently Roboflow)
- Cache rules apply for remote datasets (`--no-remote-cache`, `--remote-cache-ttl-hours`)

### Step 3: Build index

DQA converts dataset into deterministic index rows:

- image path, split, labels, parse status, hashes, dimensions, etc.

This standardized index is what checks run against.

### Step 4: Run enabled checks

Each check module receives index payload and emits findings.

### Step 5: Serialize artifacts

Outputs under `--out`:

- `index.json`
- `flags.json`
- `summary.json`
- `report.html`
- `run.log`

### Step 6: Evaluate fail gate

If any finding at/above `--fail-on` threshold exists:

- `build_failed=true`
- process exit code `1`

Else exit `0`.

## 6. Output Artifacts and Why They Matter

### `index.json`

- Deterministic input state snapshot
- Basis for reproducibility and troubleshooting

### `flags.json`

- Atomic issue list
- Best for filtering/grouping by finding ID

### `summary.json`

- Aggregated totals and severity breakdown
- Best for dashboards and CI gates

### `report.html`

- Human-friendly inspection
- Good for quick handoff to non-engineering stakeholders

### `run.log`

- Execution metadata and run outcome

## 7. Secondary Commands and Their Roles

### 7.1 `dqa explain`

Purpose:

- turn raw artifacts into prioritized action summary

Now supports:

- `--format text|markdown|json`
- `--out-file` for export to files/PR comments

### 7.2 `dqa validate`

Purpose:

- verify artifact schema contracts (`summary.json`, `flags.json`)

Why useful:

- prevents silent contract drift
- makes CI enforcement explicit and reliable

### 7.3 `dqa diff`

Purpose:

- compare two runs (old vs new)
- detect regressions and improvements by severity and finding IDs

Supports:

- `--fail-on-regression` to hard-fail CI when quality worsens at chosen severity

## 8. Configuration Strategy

### `dqa.yaml`

General default, stronger bbox/class thresholds. Best for detection-first datasets.

### `dqa_seg.yaml`

Segmentation-tuned preset; keeps integrity/leakage meaningful while reducing segmentation-induced bbox noise.

### `dqa_seg_low_noise.yaml`

Most relaxed segmentation preset, useful when bbox heuristics still dominate and hide higher-value signals.

## 9. Real Validation Story (What Was Learned)

A COCO segmentation dataset showed:

- Using `dqa.yaml`: `67725` findings
- Using `dqa_seg.yaml`: `1524` findings
- Diff: `-66201` total, no regressions

Interpretation:

- baseline checks were too noisy for segmentation geometry
- segmentation-specific tuning dramatically improved signal-to-noise
- integrity and class-support insights remained actionable

Practical policy derived:

- Use `dqa_seg.yaml` for segmentation by default
- Use `dqa_seg_low_noise.yaml` only if needed
- Keep CI gate at `high`/`critical` for segmentation workflows

## 10. Web Dashboard (No-Terminal Workflow)

`web_dashboard.py` provides a local UI around CLI commands.

What it includes:

- Audit form
- Explain form
- Validate form
- Diff form
- inline command/stdout/stderr display

Design principle:

- Don’t replace CLI internals; wrap them safely
- Keep one source of truth in CLI behavior

Recent robustness fix:

- Input normalization strips accidental wrapping quotes from path fields
- avoids `Data file not found: "C:\..."` style issues from copy/paste

## 11. CI Design

Workflow file: `.github/workflows/dqa-ci.yml`

Jobs:

1. `tests` job:
   - runs `pytest -q`
2. `dqa-gate` job:
   - builds tiny smoke dataset
   - runs `dqa audit`
   - runs `dqa validate` on summary/flags
   - uploads run artifacts

Outcome:

- Every push/PR gets baseline quality and contract checks
- Artifact outputs are inspectable in Actions

## 12. Key Challenges and Mitigations

### Challenge A: GitHub repo looked empty after push

Root cause:

- giant generated artifacts committed (`runs/`, remote zips, huge index files)

Mitigation:

- cleaned git history
- added strict `.gitignore`
- excluded generated artifacts from source control

### Challenge B: Segmentation produced excessive bbox findings

Root cause:

- bbox sanity heuristics tuned for detection can over-flag polygon-derived boxes

Mitigation:

- introduced segmentation presets (`dqa_seg.yaml`, `dqa_seg_low_noise.yaml`)
- validated with before/after diff run

### Challenge C: Schema validation failed on optional null fields

Root cause:

- serialized `None` for optional fields in `flags.json`

Mitigation:

- serializer now omits null optional keys
- schema validation passes consistently

### Challenge D: Dashboard path input errors

Root cause:

- extra literal quote characters from UI copy/paste

Mitigation:

- normalized dashboard form inputs to trim balanced quotes

## 13. How to Work on This Project Safely

Suggested developer loop:

1. Edit code in `dqa/` or `web_dashboard.py`
2. Run `pytest -q`
3. Run one smoke audit locally
4. Validate artifacts with `dqa validate`
5. If behavior changed meaningfully, run `dqa diff` against previous run
6. Update README/WIKI when behavior or usage changes

## 14. File-by-File Quick Map

### Top-level docs/config

- `README.md`: user-facing usage + examples
- `FINDING_CATALOG.md`: finding IDs and semantics
- `V1_SPEC.md`: artifact/spec references
- `wiki.md`: detailed architecture and project narrative (this file)
- `dqa.yaml`: default config
- `dqa_seg.yaml`: segmentation-tuned config
- `dqa_seg_low_noise.yaml`: low-noise segmentation config

### Core source

- `dqa/cli.py`: command routing + orchestration
- `dqa/indexer.py`: indexing pipeline
- `dqa/checks/*.py`: quality checks
- `dqa/providers/roboflow.py`: provider integration
- `dqa/report/*.py`: output generation

### Contracts and tests

- `schemas/summary.schema.json`
- `schemas/flags.schema.json`
- `tests/*`: regression protection

### UI and automation

- `web_dashboard.py`: local UI
- `.github/workflows/dqa-ci.yml`: CI gates

## 15. Operational Recommendations

### Detection projects

- Start with `dqa.yaml`
- Gate at `high` or `medium` based on tolerance

### Segmentation projects

- Start with `dqa_seg.yaml`
- If still noisy, move to `dqa_seg_low_noise.yaml`
- Gate at `high`/`critical` to avoid false-positive-heavy blocking

### Release readiness checklist

- Tests pass
- Audit artifacts validate against schemas
- Explain output generated in markdown/json for reporting
- Diff against prior dataset version shows no severe regressions

## 16. Future Growth Areas

Already implemented major robustness layers (remote ingestion, diff, validate, explain formats, CI, dashboard). Next high-value areas:

- richer HTML report drilldowns
- dataset-type-aware rule packs beyond bbox heuristics
- direct PR comment publishing in CI
- optional persisted history trends across runs

---

If you are new: start with `README.md` commands, then use the web dashboard for day-to-day work, and use this wiki as your engineering reference for internals.
