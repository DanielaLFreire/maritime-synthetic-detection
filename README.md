# Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()
[![YOLO](https://img.shields.io/badge/YOLOv11m-Ultralytics-purple)]()

## Overview

This repository contains the code, configurations, and documentation for reproducing the experiments described in our paper *"Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection"*.

**Key findings:**

1. **Negative transfer from naive use of public ship images.** Direct use of public ship images (InaTechShips, ~28k images) as pre-training data **hurts** detection performance (−4.15 pp mAP50) due to domain incompatibility in scale, density, and capture conditions.

2. **Domain-adapted synthesis recovers the gap.** **In-place synthetic composition** — replacing real vessels in operational scenes with segmented crops at the same position and scale — combined with **balanced joint training** (50% real, 50% synthetic) **improves** performance by +1.00 pp mAP50 over the COCO baseline.

3. **The mAP50 gain is from synthetic diversity, not volume.** A volume-matched control (B2-long: CITRA-3D-Real oversampled 13× with no synthetic data) achieves mAP50 statistically identical to standard B2, confirming that the +1.00 pp gain is entirely attributable to synthetic vessel diversity and the joint training regime, not to increased training volume.

4. **Spatial anchoring is data hygiene, not the main performance driver.** A controlled ablation (A' joint-rand) using sea-aware random placement instead of in-place anchoring achieves statistically equivalent mAP50 to A' joint. The performance gain over B2 is attributable to crop diversity, scale/density alignment, and the joint balanced training regime — not to the specific spatial placement of synthetic objects.

## Results Summary

Test set: CITRA-3D-Real (401 images). All values are mean ± std over 3 seeds.

| Arm | Pipeline | mAP50 | mAP50-95 | Δ vs B2 (mAP50) |
|-----|----------|-------|----------|-----------------|
| **A' joint-rand** ⭐ | **COCO → real+synth balanced (random placement)** | **0.8457 ± 0.0058** | **0.5208 ± 0.0024** | **+1.06 pp** |
| **A' joint** | **COCO → real+synth balanced (in-place)** | **0.8451 ± 0.0033** | **0.5206 ± 0.0017** | **+1.00 pp** |
| B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0020 | 0.5055 ± 0.0022 | ref |
| B2-long | COCO → CITRA-3D ×13 (volume control) | 0.8351 ± 0.0025 | 0.5100 ± 0.0038 | **0.00 pp** |
| A' frozen | COCO → synth (freeze) → CITRA-3D | 0.8342 ± 0.0039 | 0.5074 ± 0.0035 | −0.09 pp |
| A' sequential | COCO → synth (100ep) → CITRA-3D | 0.8221 ± 0.0085 | 0.4933 ± 0.0059 | −1.31 pp |
| B1 (baseline) | Random init → CITRA-3D | 0.8008 ± 0.0061 | 0.4742 ± 0.0006 | −3.43 pp |
| B (random pool) | COCO → random InaTech → CITRA-3D | 0.7945 ± 0.0046 | 0.4728 ± 0.0045 | −4.06 pp |
| A (curated) | COCO → curated InaTech → CITRA-3D | 0.7936 ± 0.0049 | 0.4692 ± 0.0017 | −4.15 pp |

⭐ **A' joint-rand** confirms that the balanced joint training regime is robust to placement choice within the sea region. See Section 5.5 of the manuscript for the ablation analysis.

**B2-long volume control**: mAP50 identical to B2 confirms that the +1.00 pp gain of A' joint is fully attributable to synthetic data + joint training regime, not to training volume. See Section 5.4 of the manuscript.

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                                  # Experiment configurations (YAML)
│   ├── baselines.yaml
│   ├── braco_a.yaml
│   ├── braco_a_sintetico.yaml
│   ├── braco_b.yaml
│   ├── braco_balanced.yaml                   # A' joint
│   ├── braco_b2_long.yaml                    # B2-long (volume control)
│   ├── braco_random_copypaste.yaml           # A' joint-rand
│   └── hyperparams.yaml
├── scripts/
│   ├── 01_data_preparation/                  # CITRA-3D-Real preparation
│   ├── 02_dataset_acquisition/               # InaTechShips download & curation
│   ├── 03_analysis/                          # Scale profiling
│   ├── 04_synthetic_generation/
│   │   ├── extrair_crops_sam.py
│   │   ├── gerar_dataset_copypaste.py        # In-place composition (A' joint)
│   │   ├── gerar_dataset_copypaste_v4.py
│   │   └── gerar_baseline_random_copypaste_v4.py   # Sea-aware random
│   ├── 05_training/
│   │   ├── treinar_baselines.py              # B1, B2
│   │   ├── treinar_b2_long.py                # B2-long (volume control)
│   │   ├── treinar_braco_a.py
│   │   ├── treinar_random_copypaste_joint.py # A' joint-rand training
│   │   └── avaliar_todos.py
│   └── 06_figures/
│       └── gerar_fig6_comparison.py
├── paper/                                    # Manuscript (LaTeX, figures)
│   ├── main.tex
│   ├── references.bib
│   ├── cover_letter.tex
│   └── figs/
├── docs/                                     # Research documentation
└── results/
    ├── metricas_detalhadas.json              # All arms, n=3 metrics
    ├── avaliacao_completa.json
    ├── test_set_consolidated.json            # Re-evaluation of all arms
    ├── results_joint_rand_n3.json            # A' joint-rand, n=3
    ├── results_b2_long_n3.json               # B2-long, n=3
    ├── tables/
    │   └── final_comparison.md               # Complete comparison table
    └── synthetic_generation/
        ├── composicao_report.json
        ├── composicao_report_v4.json
        └── smoke_tests/                      # Visual validation
```

## Reproduction Guide

### Prerequisites

```bash
pip install -r requirements.txt
```

Tested with: Python 3.10+, PyTorch 2.0+, Ultralytics 8.4.60+, CUDA 12.x. Trained on NVIDIA A100-80GB and L4 GPUs via Google Colab Pro+.

### Datasets

| Dataset | Size | Access |
|---------|------|--------|
| **CITRA-3D-Real** | 2,081 images | **Restricted** — operational surveillance imagery from the Brazilian Navy. Available upon reasonable request to the corresponding author with CASNAV/DMarSup approval. |
| **InaTechShips** | ~28k images | Publicly available at [EduardoHT/InaTechShips](https://github.com/EduardoHT/InaTechShips). The curated 25k subset used in this work can be reproduced via `scripts/02_dataset_acquisition/` using `ids_alvo.txt`. |
| **Synthetic datasets and weights** | ~5 GB | To be archived on Zenodo upon paper acceptance (DOI to be added). |

### Step-by-Step

1. **Data preparation** (`scripts/01_*`): prepare CITRA-3D-Real splits and single-class labels.
2. **Dataset acquisition** (`scripts/02_*`): download InaTechShips curated and random subsets.
3. **Analysis** (`scripts/03_*`): scale profiling (Figure 2 of the paper).
4. **Synthetic generation** (`scripts/04_*`):
   - `gerar_dataset_copypaste_v4.py` — in-place composition (A' joint).
   - `gerar_baseline_random_copypaste_v4.py` — sea-aware random placement (A' joint-rand).
5. **Training** (`scripts/05_*`): baselines, ablation experiments, joint training.
6. **Figures** (`scripts/06_*`): regenerate paper figures from saved results.

### Reproducing B2-long (volume control)

```bash
# Trains 3 seeds with CITRA-3D-Real oversampled 13×, evaluates on test set
python scripts/05_training/treinar_b2_long.py

# Output: results/results_b2_long_n3.json
#         (per-seed metrics + aggregated mean ± std for n=3)
```

### Reproducing A' joint-rand

```bash
# 1. Generate synthetic dataset (sea-aware random placement)
python scripts/04_synthetic_generation/gerar_baseline_random_copypaste_v4.py

# 2. Train 3 seeds and evaluate on test set
python scripts/05_training/treinar_random_copypaste_joint.py

# Output: results/results_joint_rand_n3.json
```

## Data and Code Availability

All source code, experiment configurations, and analysis scripts are available in this repository. The CITRA-3D-Real images and derived synthetic images cannot be publicly released because they contain restricted operational maritime surveillance imagery. Aggregated statistics, configuration files, and per-seed metrics are available here. Trained weights and synthetic datasets will be archived on Zenodo upon acceptance.

## Citation

```bibtex
@article{freire2026visual,
  title   = {Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection},
  author  = {Freire, Daniela L. and Teixeira, Eduardo H. and Moreira, Leandro A. S.},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the InaTechShips dataset:

```bibtex
@article{teixeira2025inatechships,
  title   = {InaTechShips: A validation study of a novel ship dataset through deep learning-based classification and detection models for maritime applications},
  author  = {Teixeira, Eduardo H. and Mafra, Samuel B. and De Figueiredo, Felipe A.P.},
  journal = {Ocean Engineering},
  volume  = {326},
  pages   = {120823},
  year    = {2025},
  doi     = {10.1016/j.oceaneng.2025.120823}
}
```

## Authors

- **Daniela L. Freire** — Institute of Mathematics and Computer Science (ICMC), University of São Paulo (USP) — [danielalfreire@icmc.usp.br](mailto:danielalfreire@icmc.usp.br)
- **Eduardo H. Teixeira** — Federal University of Itajubá (UNIFEI)
- **Leandro A. S. Moreira** — National Laboratory for Scientific Computing (LNCC) — supervisor

## Acknowledgements

We thank CASNAV for providing the CITRA-3D-Real dataset and Eduardo H. Teixeira for sharing the complete PointRend label set for InaTechShips. Computational resources were provided by Google Colab Pro+ (NVIDIA A100 GPU).

## License

MIT — see [LICENSE](LICENSE) file.
