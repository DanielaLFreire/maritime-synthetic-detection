# Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()
[![YOLO](https://img.shields.io/badge/YOLOv11m-Ultralytics-purple)]()

## Overview

This repository contains the code, configurations, and documentation for reproducing the experiments described in our paper *"Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection"*.

**Key finding:** Direct use of public ship images (InaTechShips, ~28k images) as pre-training data **hurts** detection performance (−4.15% mAP50) due to domain incompatibility (scale, density, capture conditions). However, **domain adaptation via in-place synthetic composition** — replacing real vessels in operational scenes with segmented crops from public images at the same position and scale — **improves** performance over the COCO baseline, with non-overlapping confidence intervals across seeds.

## Application Context

This work addresses a general challenge in operational maritime surveillance: limited annotated data from coastal monitoring, port safety, search and rescue, and maritime traffic systems. While the operational dataset was collected under an institutional maritime monitoring programme, the methods and findings are applicable to any maritime surveillance scenario where public ship imagery is available but direct transfer fails.

## Results Summary

*Results being updated — v4 synthetic generation (leak-free) in progress.*

| Arm | Pipeline | mAP50 | Δ vs B2 |
|-----|----------|-------|---------|
| **A' (synthetic v4)** | **COCO → synthetic → CITRA-3D** | **TBD** | **TBD** |
| B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0024 | ref |
| B1 (baseline) | Random init → CITRA-3D | 0.8008 ± 0.0073 | −3.43% |
| B (random) | COCO → random pool → CITRA-3D | TBD (3 seeds) | TBD |
| A (curated) | COCO → InaTechShips → CITRA-3D | 0.7936 ± 0.0060 | −4.15% |

> **Note on data integrity:** v4 synthetic generation ensures each synthetic partition is derived exclusively from the corresponding CITRA-3D-Real partition, preventing data leakage. Earlier v3 results (mAP50=0.8541) are deprecated.

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
│
├── configs/                           # Experiment configurations
│   ├── hyperparams.yaml               # Validated hyperparameters (HPO)
│   ├── baselines.yaml                 # B1 and B2 configurations
│   ├── braco_a.yaml                   # Arm A (curated direct)
│   ├── braco_b.yaml                   # Arm B (random direct, 3 seeds)
│   └── braco_a_sintetico.yaml         # Arm A' (synthetic v4, leak-free)
│
├── scripts/
│   ├── 01_data_preparation/           # CITRA-3D-Real preparation
│   │   ├── preparar_citra3d.py
│   │   ├── limpar_labels_citra3d.py
│   │   ├── gerar_labels_single_class.py
│   │   └── preparar_dados_locais.py
│   │
│   ├── 02_dataset_acquisition/        # InaTechShips download & curation
│   │   ├── reconstruir_dataset_25k.py
│   │   ├── baixar_random_pool_v2.py
│   │   ├── gerar_ids_rodada2.py
│   │   ├── validar_imagens_random_pool.py
│   │   ├── downsample_random_pool_v2.py
│   │   └── filtrar_labels_eduardo.py
│   │
│   ├── 03_analysis/                   # Dataset analysis & scale profiling
│   │   ├── analisar_escala_citra3d.py
│   │   └── recalcular_distribuicao_decis_v2.py
│   │
│   ├── 04_synthetic_generation/       # Scale-Aware Copy-Paste pipeline
│   │   ├── extrair_crops_sam.py
│   │   ├── extrair_fundos_citra3d.py
│   │   ├── gerar_dataset_copypaste.py     # v3 (deprecated — data leakage)
│   │   └── gerar_dataset_copypaste_v4.py  # v4 (split-isolated, leak-free)
│   │
│   ├── 05_training/                   # Training & evaluation
│   │   ├── treinar_baselines.py
│   │   ├── hpo_b2.py
│   │   ├── treinar_braco_a.py
│   │   └── ablation_epocas_pretreino.py
│   │
│   └── 06_figures/                    # Paper figure generation
│       └── gerar_figuras_paper.py
│
├── docs/                              # Research documentation
│   ├── resumo_experimentos.md
│   ├── experimento_descricao_v04.md
│   └── cronograma_v04.md
│
├── results/                           # Experiment outputs
│   ├── scale_analysis/
│   │   └── escala_citra3d_report.json
│   ├── synthetic_generation/
│   │   ├── composicao_report.json     # v3 (deprecated)
│   │   └── composicao_report_v4.json  # v4 (leak-free)
│   ├── labels_filtering/
│   │   └── filtrar_labels_report.json
│   └── tables/
│       └── final_comparison.md
│
└── references/
    └── citation.bib
```

## Reproduction Guide

### Prerequisites

- Python 3.10+
- Google Colab Pro+ (A100 GPU recommended for training; CPU sufficient for synthetic generation)
- Google Drive (~100 GB free space)

```bash
pip install ultralytics>=8.4 segment-anything opencv-python torch torchvision
```

### Datasets

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| **CITRA-3D-Real** | Institutional maritime programme | 2,081 images | Restricted — contact authors |
| **InaTechShips** | Teixeira et al. (2025) | ~28k images | [GitHub](https://github.com/EduardoHT/InaTechShips) / shipspotting.com |
| **SAM weights** | Meta AI | 375 MB | Auto-downloaded by scripts |

> **Note:** CITRA-3D-Real is an operational maritime surveillance dataset and cannot be publicly distributed. Contact the corresponding author for access.

### Step-by-Step Reproduction

#### Stage 1–3: Data preparation, acquisition, analysis
See scripts in `scripts/01_*` through `scripts/03_*`.

#### Stage 4: Synthetic generation (v4, leak-free)
```bash
python scripts/04_synthetic_generation/gerar_dataset_copypaste_v4.py
```
**Important:** Use v4, not v3. v4 generates each synthetic partition exclusively from the corresponding source partition, preventing data leakage.

#### Stage 5: Training
```bash
python scripts/05_training/treinar_baselines.py --all
python scripts/05_training/treinar_braco_a.py --seeds 42 123 2024
```

## Data Integrity

To prevent data leakage, the following protocol was observed:

1. The CITRA-3D-Real train/validation/test split was established before synthetic generation.
2. Synthetic training images were generated exclusively from the training partition.
3. No background, annotation, or object position from validation or test partitions was used during pre-training or model selection.
4. Final evaluation was conducted solely on the held-out real CITRA-3D-Real test set.

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

- **Daniela L. Freire** — ICMC/USP (lead researcher)
- **Eduardo H. Teixeira** — INATEL (dataset collaboration, co-author)
- **Leandro A. S. Moreira** — supervisor

## License

MIT License — see [LICENSE](LICENSE) for details.
