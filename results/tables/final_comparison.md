# Final Comparison — All Experimental Arms

## Main Results (test set, CITRA-3D-Real)

| Rank | Arm | Pipeline | mAP50 | mAP50-95 | P | R | F1 | Δ vs B2 |
|------|-----|----------|-------|----------|---|---|----|----|
| **1** | **A' joint balanced** | **COCO → real+synth balanced** | **0.8451 ± 0.0033** | **0.5206 ± 0.0017** | **0.857** | **0.805** | **0.830** | **+1.00 pp** |
| 2 | B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0020 | 0.5055 ± 0.0022 | 0.857 | 0.783 | 0.818 | ref |
| 3 | A' frozen backbone | COCO → synth (freeze) → CITRA-3D | 0.8342 ± 0.0039 | 0.5074 ± 0.0035 | 0.855 | 0.774 | 0.812 | −0.09 pp |
| 3 | Synthetic 20ep | COCO → synth (20ep) → CITRA-3D | 0.8344 ± 0.0033 | 0.5011 ± 0.0065 | 0.855 | 0.773 | 0.812 | −0.08 pp |
| 5 | A' sequential (100ep) | COCO → synth (100ep) → CITRA-3D | 0.8221 ± 0.0085 | 0.4933 ± 0.0059 | 0.828 | 0.769 | 0.797 | −1.31 pp |
| 6 | B1 (baseline) | Random → CITRA-3D | 0.8008 ± 0.0061 | 0.4742 ± 0.0006 | 0.829 | 0.750 | 0.787 | −3.43 pp |
| 7 | B (random) | COCO → random pool → CITRA-3D | 0.7945 ± 0.0046 | 0.4728 ± 0.0045 | 0.858 | 0.742 | 0.796 | −4.06 pp |
| 8 | A (curated) | COCO → InaTechShips → CITRA-3D | 0.7936 ± 0.0049 | 0.4692 ± 0.0017 | 0.834 | 0.735 | 0.781 | −4.15 pp |

## Ablation: Pre-training Epochs on InaTechShips Direct (seed 42)

| Epochs | mAP50 | Δ vs B2 |
|--------|-------|---------|
| 0 (B2) | 0.8351 | ref |
| 10 | 0.8200 | −1.51 pp |
| 20 | 0.8171 | −1.80 pp |
| 50 | 0.8037 | −3.14 pp |
| 100 | 0.8006 | −3.45 pp |

## Dataset Statistics

| Dataset | Images | Objects | Obj/img |
|---------|--------|---------|---------|
| CITRA-3D-Real | 2,081 | 7,003 | 3.37 |
| dataset_25k_v2 (curated) | 27,796 | ~27,796 | ~1.0 |
| random_pool_v2 | 27,964 | — | ~1.0 |
| Synthetic (in-place) | 27,053 | 91,035 | 3.37 |
