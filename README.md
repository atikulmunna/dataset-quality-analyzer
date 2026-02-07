# Dataset Quality Analyzer (DQA) for YOLO

DQA is a read-only dataset auditing CLI for YOLO object-detection datasets.
It evaluates data quality before training and produces machine-readable artifacts plus an HTML report.

## What DQA Evaluates

DQA evaluates datasets across six quality dimensions:

1. Integrity (annotation/file correctness)
2. Class distribution (imbalance, low-support, split drift)
3. Bounding-box sanity (area/aspect/count heuristics)
4. Exact duplicates (within/across splits)
5. Near duplicates (optional)
6. Leakage (train vs val/test contamination)

## Supported Dataset Format

DQA expects Ultralytics-style YOLO layout (detection and segmentation):

- `data.yaml` with:
  - `path`
  - `train`, `val`, optional `test`
  - `names` (list or numeric-key map)
- Label rows in YOLO format:

```txt
class_id x_center y_center width height
```

Coordinates are expected to be normalized to `[0,1]`.

## Quick Decision Table

Use this first, then jump to the detailed commands later in this README.

| Dataset Type | Input You Provide | Command (PowerShell) | Recommended Config | Expected Output |
|---|---|---|---|---|
| YOLO Detection (Local) | `data.yaml` path | `python -m dqa audit --data "C:\path\to\data.yaml" --out "runs\dqa_detect" --config "dqa.yaml"` | `dqa.yaml` | `index.json`, `flags.json`, `summary.json`, `report.html`, `run.log` |
| YOLO Detection (Roboflow URL) | Roboflow project/version URL | `python -m dqa audit --data-url "https://app.roboflow.com/workspace/project/1" --data-url-format yolov11 --out "runs\dqa_detect_remote" --config "dqa.yaml"` | `dqa.yaml` | Same artifacts in run folder |
| YOLO Segmentation (Local) | `data.yaml` path | `python -m dqa audit --data "C:\path\to\yolo-seg\data.yaml" --out "runs\dqa_yolo_seg" --config "dqa_seg.yaml"` | `dqa_seg.yaml` | Same artifacts in run folder |
| YOLO Segmentation (Low Noise) | `data.yaml` path | `python -m dqa audit --data "C:\path\to\yolo-seg\data.yaml" --out "runs\dqa_yolo_seg_low_noise" --config "dqa_seg_low_noise.yaml"` | `dqa_seg_low_noise.yaml` | Same artifacts, fewer bbox sanity flags |
| COCO Segmentation (Local Folder) | Dataset folder containing split JSONs | `python -m dqa audit --data "C:\path\to\coco-seg-dataset" --out "runs\dqa_coco_seg" --config "dqa_seg.yaml"` | `dqa_seg.yaml` | Same artifacts in run folder |
| COCO Segmentation (Single Split JSON) | One `_annotations.coco.json` file | `python -m dqa audit --data "C:\path\to\train\_annotations.coco.json" --out "runs\dqa_coco_train_only" --config "dqa_seg.yaml"` | `dqa_seg.yaml` | Same artifacts for that split only |

Quick tips:
- Use `python -m dqa explain --run "runs\<run_name>"` after every audit.
- Use `--fail-on low|medium|high|critical` to enforce quality gates in CI.
- For remote Roboflow runs, set `ROBOFLOW_API_KEY` first.
## Installation and Run

From repository root:

```bash
python -m dqa audit --data /path/to/data.yaml --out runs/dqa_001 --config dqa.yaml
```

Windows PowerShell example:

```powershell
python -m dqa audit --data "C:\path\to\data.yaml" --out "runs\dqa_001" --config "dqa.yaml"
echo $LASTEXITCODE
```

Remote URL scaffold example:

```powershell
python -m dqa audit --data-url "https://universe.roboflow.com/workspace/project/version" --out "runs\dqa_remote" --config "dqa.yaml"
echo $LASTEXITCODE
```

## CLI Options

One of `--data` or `--data-url` is required.

- `--data`: local path to `data.yaml`
- `--data-url`: remote dataset URL (currently Roboflow URLs are supported)
- `--data-url-format`: remote export format (default `yolov11`)
- `--roboflow-api-key`: optional API key override (otherwise uses `ROBOFLOW_API_KEY`)
- `--no-remote-cache`: force fresh remote download even if cached export exists
- `--remote-cache-ttl-hours`: re-download when cached export is older than TTL
- `--out`: required output directory
- `--config`: path to DQA config (`dqa.yaml` by default)
- `--splits`: comma-separated subset (default `train,val,test`)
- `--workers`: reserved for parallelism tuning
- `--max-images`: limit indexed image count (`0` means all)
- `--near-dup`: enable near-duplicate check even if config disables it
- `--format`: output formats, comma-separated (`html,json`)
- `--fail-on`: severity gate (`critical|high|medium|low`)

## Output Artifacts

DQA writes all artifacts to `--out`:

| File | Meaning |
|---|---|
| `index.json` | Deterministic dataset index and cache key basis |
| `flags.json` | Atomic findings (one item per issue) |
| `summary.json` | Aggregated counts, run metadata, pass/fail gate result |
| `report.html` | Human-readable report for quick inspection |
| `run.log` | Basic run metadata and outcome |

## Dataset Evaluation Standards

DQA standards are controlled by `dqa.yaml` thresholds.
Default evaluation behavior:

- Integrity:
  - Missing label for image -> `high`
  - Orphan label (no image) -> `medium`
  - Malformed label row -> `high`
  - Invalid class id -> `high`
  - Out-of-range normalized coords -> `high`
  - Corrupt/unreadable image -> `critical`
- Class distribution:
  - Dominant class share over threshold (`max_class_share_warn`) -> `medium`
  - Class with fewer than `min_instances_per_class_warn` -> `low`
  - Train-vs-val/test drift (JSD over threshold) -> `medium`
- Bounding boxes:
  - Tiny/oversized boxes by area ratio -> `medium`
  - Extreme aspect ratio -> `medium`
  - Too many boxes per image -> `medium`
- Exact duplicates:
  - Within split -> `medium`
  - Across splits -> `high`
- Leakage:
  - Train<->val exact duplicate -> `critical`
  - Train<->test exact duplicate -> `critical`
- Near duplicates:
  - Within split -> `low`
  - Across splits -> `high`

## Severity, Gating, and Exit Codes

Severity levels:

- `critical`: immediate training blocker
- `high`: high-risk issue
- `medium`: quality risk
- `low`: informational/warning

Fail logic:

- If any finding is at or above `--fail-on`, `build_failed=true` and exit code `1`.
- Otherwise, exit code `0`.

Other exit codes:

- `2`: config/usage error (invalid config, bad args)
- `3`: runtime/data access error

## Meaning of Flags (`flags.json`)

Every finding contains:

- `id`: stable finding code
- `severity`: critical/high/medium/low
- `message`: human-readable explanation
- `fingerprint`: stable issue key for dedupe/tracking

Common optional fields:

- `split`, `image`, `label`, `class_id`, `metrics`, `suggested_action`

Finding IDs are frozen in `FINDING_CATALOG.md`.

### Flag Families

- `INTEGRITY_*`: label/image structure and annotation correctness
- `CLASS_*`: class support, imbalance, and split drift
- `BBOX_*`: annotation geometry anomalies
- `DUPLICATE_*`: exact content duplicates
- `LEAKAGE_*`: train-validation/test contamination
- `NEAR_DUPLICATE_*`: perceptual near-duplicate clusters

## How to Interpret Your Run

Typical interpretation flow:

1. Check `summary.json -> totals` for `build_failed` and severity counts.
2. Inspect `checks` block to see which module produced findings.
3. Open `flags.json` and sort/group by `id` to identify dominant failure mode.
4. Use `report.html` for quick stakeholder review.

Example:

- Only `CLASS_LOW_SUPPORT` findings with severity `low` means data is structurally clean but some classes are underrepresented.
- `LEAKAGE_EXACT_TRAIN_VAL` means your split is contaminated and model validation is unreliable.

## Configuration

Default config file:

- `dqa.yaml`

Contract/schema references:

- `V1_SPEC.md`
- `schemas/summary.schema.json`
- `schemas/flags.schema.json`
- `FINDING_CATALOG.md`

## Testing

Run tests:

```bash
pytest -q
```

Current baseline covers:

- config validation behavior
- integrity detection sanity
- duplicates/leakage detection sanity

## Notes

- DQA is read-only on dataset content.
- Near-duplicate detection uses Pillow when available.
- Output ordering is deterministic for stable CI behavior.

## License

MIT

## Troubleshooting

### `config error: ... unknown keys`

Your `dqa.yaml` contains unsupported keys. Keep keys aligned with the current config contract in `V1_SPEC.md`.

### `runtime error: Data file not found`

Your `--data` path is wrong or quoted incorrectly.

PowerShell example:

```powershell
python -m dqa audit --data "C:\full\path\to\data.yaml" --out "runs\dqa_run" --config "dqa.yaml"
```

### Exit code is `1` but run completed

This is expected when findings match or exceed your `--fail-on` threshold. Check:

- `summary.json -> totals.fail_threshold`
- `summary.json -> totals.by_severity`

### `--data-url` with Roboflow URL fails

Check these first:

- `ROBOFLOW_API_KEY` is set (or pass `--roboflow-api-key`)
- URL includes workspace/project/version
- Format is valid for export (`--data-url-format`, default `yolov11`)
- Network access to `api.roboflow.com` is available
- DQA retries transient API/download errors and reuses cached extracted exports under `--out/_remote` by default
- Use `--no-remote-cache` for always-fresh downloads or `--remote-cache-ttl-hours` to bound cache age

### `near_duplicates` shows skipped/completed with zero findings

- If Pillow is unavailable, near-duplicate checks may be skipped.
- If enabled and completed with zero findings, no near duplicates were detected under current threshold.

### Only `CLASS_LOW_SUPPORT` findings appear

Dataset is structurally healthy, but one or more classes have low sample counts under `min_instances_per_class_warn`.

### I typed `--fail-on low` and got a PowerShell parser error

`--fail-on` is a flag for `python -m dqa audit`, not a standalone command.

Correct usage:

```powershell
python -m dqa audit --data "C:\path\to\data.yaml" --out "runs\dqa_low" --config "dqa.yaml" --fail-on low
```

## Future Robustness Features

High-impact additions to make DQA more production robust:

1. Direct remote dataset ingestion
- Add support for `--data-url` (HTTP/S) and provider adapters.
- Roboflow support: accept project/version links, authenticate via API key, download export, audit locally, clean temp files.

2. Roboflow-native command
- Example target UX:
  - `dqa audit --roboflow-workspace ws --roboflow-project dice --roboflow-version 2 --format yolov11 --out runs/dqa_rf`
- Add `ROBOFLOW_API_KEY` env var support and clear errors for auth/rate limits.

3. Schema validation mode
- `dqa validate --artifact runs/x/summary.json --schema schemas/summary.schema.json`
- Useful for CI contract enforcement.

4. Dataset-to-dataset diff mode
- `dqa diff --old runs/a --new runs/b`
- Show regressions in leakage, imbalance, and integrity findings.

5. Better CI integration
- GitHub Action template, PR annotations, and Markdown summary generation.

6. Better explainability
- Per-flag remediation hints and class-wise drilldowns in HTML report.

7. Scale and reliability
- Incremental index cache reuse by file hash.
- Chunked hashing and optional multiprocessing for large datasets.

8. Advanced quality checks
- Image quality signals (blur, exposure, resolution outliers).
- Annotation overlap/pathology checks (crowded boxes, improbable distributions).
## Remote Cache Controls

DQA reuses extracted remote exports under `--out/_remote` by default.

```powershell
# Force fresh remote fetch
python -m dqa audit --data-url "https://universe.roboflow.com/workspace/project/1" --no-remote-cache --out "runs\dqa_remote_fresh" --config "dqa.yaml"

# Reuse cache only if newer than 24 hours
python -m dqa audit --data-url "https://universe.roboflow.com/workspace/project/1" --remote-cache-ttl-hours 24 --out "runs\dqa_remote_ttl" --config "dqa.yaml"
```
## Explain Findings

Summarize and prioritize actions from an existing run:

```powershell
python -m dqa explain --run "runs\dqa_remote_fresh"
```

Or provide explicit files:

```powershell
python -m dqa explain --summary "runs\dqa_remote_fresh\summary.json" --flags "runs\dqa_remote_fresh\flags.json"
```
## Segmentation Config

Use `dqa_seg.yaml` for segmentation-heavy datasets (YOLO-seg / COCO-seg), where polygon-derived boxes can otherwise trigger noisy bbox flags.

```powershell
python -m dqa audit --data "C:\path\to\segmentation-dataset" --out "runs\dqa_seg_tuned" --config "dqa_seg.yaml"
```

This config keeps integrity/leakage checks active but relaxes bbox/class thresholds for segmentation behavior.
### Lower-Noise Segmentation Preset

If bbox-derived findings are still too noisy for polygon-heavy datasets, use `dqa_seg_low_noise.yaml` (disables `bbox_sanity`):

```powershell
python -m dqa audit --data "C:\path\to\segmentation-dataset" --out "runs\dqa_seg_low_noise" --config "dqa_seg_low_noise.yaml"
```
## Dataset-Type Command Guide

Use these commands based on dataset type.

### 1) YOLO Detection (local export)

```powershell
python -m dqa audit --data "C:\path\to\data.yaml" --out "runs\dqa_detect" --config "dqa.yaml"
python -m dqa explain --run "runs\dqa_detect"
```

### 2) YOLO Detection (Roboflow URL)

```powershell
$env:ROBOFLOW_API_KEY="your_key"
python -m dqa audit --data-url "https://app.roboflow.com/workspace/project/1" --data-url-format yolov11 --out "runs\dqa_detect_remote" --config "dqa.yaml"
python -m dqa explain --run "runs\dqa_detect_remote"
```

### 3) YOLO Segmentation (local export)

```powershell
python -m dqa audit --data "C:\path\to\yolo-seg\data.yaml" --out "runs\dqa_yolo_seg" --config "dqa_seg.yaml"
python -m dqa explain --run "runs\dqa_yolo_seg"
```

For lower noise on polygon-heavy datasets:

```powershell
python -m dqa audit --data "C:\path\to\yolo-seg\data.yaml" --out "runs\dqa_yolo_seg_low_noise" --config "dqa_seg_low_noise.yaml"
```

### 4) COCO Segmentation (local export folder)

Point `--data` to the dataset root containing split folders (`train/valid/test`) and COCO annotation JSON files.

```powershell
python -m dqa audit --data "C:\path\to\coco-seg-dataset" --out "runs\dqa_coco_seg" --config "dqa_seg.yaml"
python -m dqa explain --run "runs\dqa_coco_seg"
```

For low-noise mode:

```powershell
python -m dqa audit --data "C:\path\to\coco-seg-dataset" --out "runs\dqa_coco_seg_low_noise" --config "dqa_seg_low_noise.yaml"
```

### 5) COCO Segmentation (single split JSON only)

If needed, point directly to a split annotation JSON (audits that split only):

```powershell
python -m dqa audit --data "C:\path\to\train\_annotations.coco.json" --out "runs\dqa_coco_train_only" --config "dqa_seg.yaml"
```

### 6) Remote cache controls (for `--data-url`)

```powershell
# Force fresh remote download
python -m dqa audit --data-url "https://app.roboflow.com/workspace/project/1" --data-url-format yolov11 --no-remote-cache --out "runs\dqa_remote_fresh" --config "dqa.yaml"

# Reuse remote cache only if newer than 24 hours
python -m dqa audit --data-url "https://app.roboflow.com/workspace/project/1" --data-url-format yolov11 --remote-cache-ttl-hours 24 --out "runs\dqa_remote_ttl" --config "dqa.yaml"
```

### Recommended config by dataset type

- Detection-first datasets: `dqa.yaml`
- Segmentation datasets (balanced): `dqa_seg.yaml`
- Segmentation datasets (reduce bbox-noise): `dqa_seg_low_noise.yaml`





## CI Workflow Gate

GitHub Actions workflow: `.github/workflows/dqa-ci.yml`

What it does on every push/PR:

- Runs `pytest -q`
- Builds a tiny smoke dataset in CI
- Runs `python -m dqa audit` with a severity gate
- Validates `summary.json` and `flags.json` against schemas
- Uploads `runs/ci_smoke` as an artifact

Manual run:

- Use **Actions -> dqa-ci -> Run workflow**
- Optional input: `fail_on` (`critical|high|medium|low`, default `high`)

