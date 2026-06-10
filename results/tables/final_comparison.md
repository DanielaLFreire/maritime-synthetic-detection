# Final Comparison — Test Set Results

Test set: CITRA-3D-Real (401 images, 1,247 instances).
Metrics: mAP50, mAP50-95, Precision, Recall, F1 (IoU = 0.5).
All entries use seeds 42, 123, 2024 (mean ± std, n=3).

## Main results

| Arm | Pipeline | mAP50 | mAP50-95 | P | R | F1 | n | Δ vs B2 (mAP50) |
|-----|----------|-------|----------|---|---|---|---|-----------------|
| **A' joint-rand** ⭐ | **COCO → real+synth balanced (random)** | **0.8457 ± 0.0058** | **0.5208 ± 0.0024** | **0.852 ± 0.009** | **0.804 ± 0.010** | **0.827 ± 0.005** | 3 | **+1.06** |
| **A' joint** | **COCO → real+synth balanced (in-place)** | **0.8451 ± 0.0033** | **0.5206 ± 0.0017** | **0.857 ± 0.007** | **0.805 ± 0.003** | **0.830 ± 0.002** | 3 | **+1.00** |
| B2 | COCO → CITRA-3D | 0.8351 ± 0.0020 | 0.5055 ± 0.0022 | 0.857 ± 0.005 | 0.783 ± 0.004 | 0.818 ± 0.002 | 3 | ref |
| B2-long | COCO → CITRA-3D ×13 (volume control) | 0.8351 ± 0.0025 | 0.5100 ± 0.0038 | 0.851 ± 0.004 | 0.796 ± 0.012 | 0.823 ± 0.008 | 3 | **0.00** |
| A' frozen | COCO → synth (freeze) → CITRA-3D | 0.8342 ± 0.0039 | 0.5074 ± 0.0035 | 0.855 ± 0.001 | 0.774 ± 0.009 | 0.812 ± 0.005 | 3 | −0.09 |
| A' seq | COCO → synth (100ep) → CITRA-3D | 0.8221 ± 0.0085 | 0.4933 ± 0.0059 | 0.828 ± 0.018 | 0.769 ± 0.010 | 0.797 ± 0.013 | 3 | −1.31 |
| B1 | Random init → CITRA-3D | 0.8008 ± 0.0061 | 0.4742 ± 0.0006 | 0.829 ± 0.003 | 0.750 ± 0.002 | 0.787 ± 0.000 | 3 | −3.43 |
| B | COCO → random InaTech → CITRA-3D | 0.7945 ± 0.0046 | 0.4728 ± 0.0045 | 0.856 ± 0.003 | 0.734 ± 0.014 | 0.790 ± 0.009 | 3 | −4.06 |
| A | COCO → curated InaTech → CITRA-3D | 0.7936 ± 0.0049 | 0.4692 ± 0.0017 | 0.834 ± 0.008 | 0.735 ± 0.013 | 0.781 ± 0.004 | 3 | −4.15 |

## Key observations

- **A' joint and A' joint-rand are statistically equivalent** in mAP50 (Δ = 0.0006; well within the std of either method). They are also equivalent in mAP50-95 (Δ = 0.0002).
- **Both joint-trained variants surpass B2** by ~+1.0 pp mAP50 with non-overlapping ranges (preliminary evidence given n=3).
- **B2-long achieves mAP50 = 0.8351, identical to B2** — confirming that the +1.00 pp gain of A' joint is entirely attributable to synthetic diversity + joint regime, with no contribution from training volume.
- The Recall picture is more nuanced: B2-long Recall (0.796 ± 0.012) sits between B2 (0.783) and A' joint (0.805), suggesting that training volume contributes partially to the A' joint Recall gain, with synthetic appearance diversity contributing the remainder.
- **Sequential pre-training (A' seq) degrades performance** relative to B2 (−1.31 pp), illustrating a forgetting-like effect that the joint regime avoids.
- **Frozen backbone (A' frozen) largely neutralises** the degradation observed in A' seq but does not match the joint variants.
- **Direct pre-training on public ship images (A, B)** produces the largest negative transfer (−4.06 to −4.15 pp), confirming domain incompatibility.

## Decomposition of the A' joint gain over B2

Using B2-long (volume-matched real-only control) to isolate volume vs. synthetic effects:

| Metric | A' joint − B2 | Volume (B2-long − B2) | Synthetics+regime (residual) |
|---|---|---|---|
| mAP50 | +1.00 pp | **0.0 pp (0%)** | **+1.00 pp (100%)** |
| mAP50-95 | +1.51 pp | +0.45 pp (~30%) | +1.06 pp (~70%) |
| Recall | +2.2 pp | +1.3 pp (~59%) | +0.9 pp (~41%) |
| Precision | 0.0 pp | −0.6 pp | +0.6 pp |
| F1 | +1.2 pp | +0.5 pp | +0.7 pp |

**Interpretation**:
- The mAP50 gain is **entirely** attributable to synthetic data + joint regime
- The mAP50-95 gain is **mostly** attributable to synthetic data
- The Recall gain is **partially** attributable to volume

## Ablation: spatial anchoring (Section 5.5)

The A' joint-rand arm differs from A' joint only in the placement of synthetic crops: instead of being placed at the exact positions and scales of annotated real vessels (in-place anchoring), crops are placed at uniformly sampled positions within an adaptive sea region. Real objects are removed via Telea inpainting (radius = 3) before synthetic placement to avoid contradictory label signal. All other aspects — crop pool, scale distribution, 13× variations, training hyperparameters — are held constant.

The near-equivalence of the two variants (Δ mAP50 = 0.0006, Δ mAP50-95 = 0.0002) indicates that, within the joint balanced training regime, the model is robust to placement choice provided that the placement remains within the operational scene's sea region. The performance gain over B2 is therefore attributable to:

1. Crop diversity from InaTechShips (visual appearance);
2. Scale and density alignment with the operational domain;
3. The joint balanced training regime itself;

rather than to spatial anchoring at the exact positions of real objects.
