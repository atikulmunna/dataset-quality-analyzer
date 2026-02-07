# Frozen Finding Catalog (v1)

This file freezes finding IDs and default severities for DQA v1.
Any change to IDs or defaults after v1 requires a documented schema/version update.

## Integrity (R1)

| ID | Default Severity | Description |
|----|----|----|
| `INTEGRITY_MISSING_LABEL` | `high` | Image file has no matching label file. |
| `INTEGRITY_ORPHAN_LABEL` | `medium` | Label file has no matching image file. |
| `INTEGRITY_INVALID_CLASS_ID` | `high` | Label row class ID is < 0 or >= number of classes. |
| `INTEGRITY_MALFORMED_ROW` | `high` | Label row does not contain 5 numeric tokens. |
| `INTEGRITY_COORD_OUT_OF_RANGE` | `high` | Normalized bbox fields are outside [0,1]. |
| `INTEGRITY_CORRUPT_IMAGE` | `critical` | Image cannot be decoded/read safely. |

## Class Distribution (R2)

| ID | Default Severity | Description |
|----|----|----|
| `CLASS_IMBALANCE_HIGH` | `medium` | Dominant class share exceeds threshold. |
| `CLASS_LOW_SUPPORT` | `low` | Class support below minimum threshold. |
| `CLASS_SPLIT_DRIFT` | `medium` | Split distribution drift exceeds JSD threshold. |

## Bounding Box Sanity (R3)

| ID | Default Severity | Description |
|----|----|----|
| `BBOX_TINY_BOX` | `medium` | Box area ratio below minimum threshold. |
| `BBOX_OVERSIZED_BOX` | `medium` | Box area ratio above maximum threshold. |
| `BBOX_EXTREME_ASPECT_RATIO` | `medium` | Box aspect ratio exceeds threshold. |
| `BBOX_TOO_MANY_PER_IMAGE` | `medium` | Image has too many annotations. |

## Exact Duplicates (R4)

| ID | Default Severity | Description |
|----|----|----|
| `DUPLICATE_WITHIN_SPLIT` | `medium` | Exact duplicate images found in same split. |
| `DUPLICATE_ACROSS_SPLITS` | `high` | Exact duplicate images found across splits. |

## Near Duplicates (R5)

| ID | Default Severity | Description |
|----|----|----|
| `NEAR_DUPLICATE_WITHIN_SPLIT` | `low` | Near-duplicate cluster detected in same split. |
| `NEAR_DUPLICATE_ACROSS_SPLITS` | `high` | Near-duplicate cluster spans different splits. |

## Leakage (R6)

| ID | Default Severity | Description |
|----|----|----|
| `LEAKAGE_EXACT_TRAIN_VAL` | `critical` | Exact duplication between train and val. |
| `LEAKAGE_EXACT_TRAIN_TEST` | `critical` | Exact duplication between train and test. |
| `LEAKAGE_NEAR_TRAIN_VAL` | `high` | Near-duplicate leakage between train and val. |
| `LEAKAGE_NEAR_TRAIN_TEST` | `high` | Near-duplicate leakage between train and test. |

## Notes

- `flags.json` entries must use only IDs listed here for v1.
- Check-specific runtime context is recorded in finding fields like `split`, `image`, `label`, and `metrics`.
- CI gating uses actual `severity` emitted for each finding and `fail_on` threshold.
