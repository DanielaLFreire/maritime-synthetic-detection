# Final Comparison — All Experimental Arms

## Main Results (test set, CITRA-3D-Real)

| Rank | Arm | Pipeline | mAP50 | mAP50-95 | Δ mAP50 vs B2 | Seeds |
|------|-----|----------|-------|----------|---------------|-------|
| **1** | **A' (synthetic)** | **COCO → copy-paste → CITRA-3D** | **0.8541 ± 0.0043** | **0.5281 ± 0.0056** | **+1.90%** | 3 |
| 2 | B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0024 | 0.5055 ± 0.0027 | ref | 3 |
| 3 | B1 (baseline) | Random init → CITRA-3D | 0.8008 ± 0.0073 | 0.4742 ± 0.0008 | −3.43% | 3 |
| 4 | B (random) | COCO → random pool → CITRA-3D | 0.7997 | 0.4711 | −3.54% | 1 |
| 5 | A (curated) | COCO → InaTechShips → CITRA-3D | 0.7936 ± 0.0060 | 0.4692 ± 0.0021 | −4.15% | 3 |

## Ablation: Pre-training Epochs (seed 42)

| Pre-training epochs | mAP50 | Δ vs B2 |
|---------------------|-------|---------|
| 0 (B2 baseline) | 0.8351 | ref |
| 10 | 0.8200 | −1.51% |
| 20 | 0.8171 | −1.80% |
| 50 | 0.8037 | −3.14% |
| 100 (arm A) | 0.8006 | −3.45% |

## Statistical Separation (A' vs B2)

| Metric | A' | B2 | Separation |
|--------|----|----|------------|
| mAP50 mean | 0.8541 | 0.8351 | Δ = +0.0190 |
| mAP50 CI | [0.8498, 0.8584] | [0.8327, 0.8375] | No overlap |
| σ separation | — | — | 7.9σ |
| mAP50-95 mean | 0.5281 | 0.5055 | Δ = +0.0226 (+4.5% rel.) |

## Dataset Statistics

| Dataset | Images | Objects | Obj/img | Small (COCO) |
|---------|--------|---------|---------|--------------|
| CITRA-3D-Real | 2,081 | 7,003 | 3.37 | 71.6% |
| dataset_25k_v2 (curated) | 27,796 | ~27,796 | ~1.0 | <5% |
| random_pool_v2 | 27,964 | — | ~1.0 | <5% |
| dataset_sintetico (v3) | 27,796 | 93,480 | 3.36 | ~71% |
