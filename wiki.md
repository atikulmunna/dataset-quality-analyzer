# DQA Contributor Map

This document explains where implementation responsibilities live. User commands belong in [README.md](README.md), behavioral contracts in [V1_SPEC.md](V1_SPEC.md), and finding definitions in [FINDING_CATALOG.md](FINDING_CATALOG.md).

## Runtime flow

An audit follows one path:

1. `dqa/cli.py` parses the command and loads configuration.
2. `dqa/remote.py` resolves local input or delegates Roboflow downloads to `dqa/providers/roboflow.py`.
3. `dqa/io_yolo.py` resolves YOLO dataset structure, while `dqa/indexer.py` indexes YOLO or COCO records.
4. Modules under `dqa/checks/` produce `Finding` values.
5. `dqa/report/` writes JSON and HTML artifacts.
6. The CLI applies the severity gate and returns the documented exit code.

Dataset content is read-only. Generated outputs and remote download caches live under the selected run directory.

## Source map

| Path | Responsibility |
|---|---|
| `dqa/cli.py` | Argument parsing, command presentation, and exit-code mapping |
| `dqa/audit.py` | Typed audit service shared by CLI and future workers |
| `dqa/config.py` | Strict configuration parsing and validation |
| `dqa/io_yolo.py` | YOLO `data.yaml` and split resolution |
| `dqa/indexer.py` | YOLO/COCO parsing, image metadata, hashes, and deterministic index |
| `dqa/checks/` | Independent finding producers |
| `dqa/models/flags.py` | Active finding value object |
| `dqa/providers/roboflow.py` | Roboflow API, retry, download, and cache behavior |
| `dqa/report/` | Artifact rendering and writing |
| `schemas/` | JSON output schemas |
| `tests/` | Unit and integration coverage |
| `web_dashboard.py` | Local-only convenience UI |

## Invariants

Preserve these properties when changing the implementation:

- Dataset files are never modified.
- Findings use cataloged IDs and severities.
- Output ordering and fingerprints remain deterministic.
- Unknown configuration keys fail clearly.
- Optional finding fields with `None` values are omitted from JSON.
- Quality-gate failures return `1`; usage/configuration failures return `2`; runtime failures return `3`.
- The local dashboard is not a production security boundary.

## Change checklist

When adding or changing a check:

1. Update `FINDING_CATALOG.md` when its public finding contract changes.
2. Update schemas and `V1_SPEC.md` when artifact structure changes.
3. Add focused tests, including malformed input behavior.
4. Run `pytest -q`.
5. Run a smoke audit and validate both JSON artifacts.

Avoid generic check base classes, dependency-injection containers, plugin registries, and storage abstractions until multiple concrete implementations require them. The current direct function-based design is intentional.
