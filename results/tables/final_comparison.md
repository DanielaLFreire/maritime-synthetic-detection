# Final Comparison — All Experimental Arms

> **Note:** A' v4 results and B (3 seeds) pending. v3 results deprecated due to data leakage.

## Main Results (test set, CITRA-3D-Real)

| Rank | Arm | Pipeline | mAP50 | mAP50-95 | Δ mAP50 vs B2 | Seeds |
|------|-----|----------|-------|----------|---------------|-------|
| **1** | **A' (synthetic v4)** | **COCO → synthetic → CITRA-3D** | **TBD** | **TBD** | **TBD** | 3 |
| 2 | B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0024 | 0.5055 ± 0.0027 | ref | 3 |
| 3 | B1 (baseline) | Random init → CITRA-3D | 0.8008 ± 0.0073 | 0.4742 ± 0.0008 | −3.43% | 3 |
| 4 | B (random) | COCO → random pool → CITRA-3D | TBD (3 seeds) | TBD | TBD | 3 |
| 5 | A (curated) | COCO → InaTechShips → CITRA-3D | 0.7936 ± 0.0060 | 0.4692 ± 0.0021 | −4.15% | 3 |

## Deprecated v3 Results (data leakage — DO NOT USE)

| Arm | mAP50 | Note |
|-----|-------|------|
| A' v3 | 0.8541 ± 0.0043 | **INVALIDATED** — synthetic train contained test backgrounds |

## Data Integrity (v4)

- Synthetic train generated ONLY from CITRA-3D-Real train (1,348 × 13 = 17,524 images)
- Synthetic val generated ONLY from CITRA-3D-Real val (332 × 13 = 4,316 images)
- Synthetic test generated ONLY from CITRA-3D-Real test (401 × 13 = 5,213 images)
- Total: 27,053 images, 91,035 objects, 3.37 obj/img

## Ablation: Pre-training Epochs (seed 42)

| Pre-training epochs | mAP50 | Δ vs B2 |
|---------------------|-------|---------|
| 0 (B2 baseline) | 0.8351 | ref |
| 10 | 0.8200 | −1.51% |
| 20 | 0.8171 | −1.80% |
| 50 | 0.8037 | −3.14% |
| 100 (arm A) | 0.8006 | −3.45% |

## Dataset Statistics

| Dataset | Images | Objects | Obj/img |
|---------|--------|---------|---------|
| CITRA-3D-Real | 2,081 | 7,003 | 3.37 |
| dataset_25k_v2 (curated) | 27,796 | ~27,796 | ~1.0 |
| random_pool_v2 | 27,964 | — | ~1.0 |
| dataset_sintetico_v4 | 27,053 | 91,035 | 3.37 |
