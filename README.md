# Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()
[![YOLO](https://img.shields.io/badge/YOLOv11m-Ultralytics-purple)]()

## Overview

This repository contains the code, configurations, and documentation for reproducing the experiments described in our paper *"Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection"*.

**Key finding:** Direct use of public ship images (InaTechShips, ~28k images) as pre-training data **hurts** detection performance (−4.15 pp mAP50) due to domain incompatibility (scale, density, capture conditions). However, **in-place synthetic composition** — replacing real vessels in operational scenes with segmented crops at the same position and scale — combined with **balanced joint training** (50% real, 50% synthetic) **improves** performance by +1.00 pp mAP50 over the COCO baseline, with non-overlapping confidence intervals.

## Results Summary

| Arm | Pipeline | mAP50 | Δ vs B2 |
|-----|----------|-------|---------|
| **A' joint** | **COCO → real+synth balanced** | **0.8451 ± 0.0033** | **+1.00 pp** |
| B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0020 | ref |
| A' frozen | COCO → synth (freeze) → CITRA-3D | 0.8342 ± 0.0039 | −0.09 pp |
| A' sequential | COCO → synth (100ep) → CITRA-3D | 0.8221 ± 0.0085 | −1.31 pp |
| B1 (baseline) | Random init → CITRA-3D | 0.8008 ± 0.0061 | −3.43 pp |
| B (random) | COCO → random pool → CITRA-3D | 0.7945 ± 0.0046 | −4.06 pp |
| A (curated) | COCO → InaTechShips → CITRA-3D | 0.7936 ± 0.0049 | −4.15 pp |

## Repository Structure

```
├── README.md
├── requirements.txt
├── configs/                           # Experiment configurations (YAML)
├── scripts/
│   ├── 01_data_preparation/           # CITRA-3D-Real preparation
│   ├── 02_dataset_acquisition/        # InaTechShips download & curation
│   ├── 03_analysis/                   # Scale profiling
│   ├── 04_synthetic_generation/       # In-place composition pipeline
│   ├── 05_training/                   # Training, HPO, evaluation
│   └── 06_figures/                    # Paper figure generation
├── paper/                             # Manuscript (LaTeX, figures)
├── docs/                              # Research documentation
├── results/                           # CSVs, JSONs, figures
└── references/                        # BibTeX
```

## Reproduction Guide

### Prerequisites

```bash
pip install ultralytics>=8.4 segment-anything opencv-python torch torchvision
```

### Datasets

| Dataset | Size | Access |
|---------|------|--------|
| CITRA-3D-Real | 2,081 images | Restricted — contact authors |
| InaTechShips | ~28k images | [GitHub](https://github.com/EduardoHT/InaTechShips) |

### Step-by-Step

1. **Data preparation:** `scripts/01_*` — prepare CITRA-3D-Real
2. **Dataset acquisition:** `scripts/02_*` — download InaTechShips subsets
3. **Analysis:** `scripts/03_*` — extract scale profile
4. **Synthetic generation:** `scripts/04_*/gerar_dataset_copypaste.py` — in-place composition (each synthetic partition generated exclusively from the corresponding source partition)
5. **Training:** `scripts/05_*` — baselines, ablation, joint balanced

## Citation

```bibtex
@article{freire2026maritime,
  title={Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection},
  author={Freire, Daniela L. and Teixeira, Eduardo H. and Moreira, Leandro A. S.},
  year={2026},
  note={In preparation for Ocean Engineering}
}
```

## Authors

- **Daniela L. Freire** — ICMC/USP
- **Eduardo H. Teixeira** — INATEL
- **Leandro A. S. Moreira** — supervisor

## License

MIT
