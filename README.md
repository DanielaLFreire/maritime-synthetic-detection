# Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()
[![YOLO](https://img.shields.io/badge/YOLOv11m-Ultralytics-purple)]()

## Overview

This repository contains the code, configurations, and documentation for reproducing the experiments described in our paper *"Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection"*.

**Key finding:** Direct use of public ship images (InaTechShips, ~28k images) as pre-training data **hurts** detection performance (−4.15% mAP50) due to domain incompatibility (scale, density, capture conditions). However, **domain adaptation via in-place synthetic composition** — replacing real vessels in operational scenes with segmented crops from public images at the same position and scale — **improves** performance by +1.90% mAP50 over the COCO baseline, with complete statistical separation.

## Results Summary

| Arm | Pipeline | mAP50 | Δ vs B2 |
|-----|----------|-------|---------|
| **A' (synthetic)** | **COCO → copy-paste → CITRA-3D** | **0.8541 ± 0.0043** | **+1.90%** |
| B2 (baseline) | COCO → CITRA-3D | 0.8351 ± 0.0024 | ref |
| B1 (baseline) | Random init → CITRA-3D | 0.8008 ± 0.0073 | −3.43% |
| B (random) | COCO → random pool → CITRA-3D | 0.7997 | −3.54% |
| A (curated) | COCO → InaTechShips → CITRA-3D | 0.7936 ± 0.0060 | −4.15% |

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
│   ├── braco_b.yaml                   # Arm B (random direct)
│   └── braco_a_sintetico.yaml         # Arm A' (synthetic)
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
│   │   └── gerar_dataset_copypaste.py
│   │
│   └── 05_training/                   # Training & evaluation
│       ├── treinar_baselines.py
│       ├── hpo_b2.py
│       ├── treinar_braco_a.py
│       └── ablation_epocas_pretreino.py
│
├── docs/                              # Research documentation
│   ├── resumo_experimentos.md         # Experiment summary (Portuguese)
│   ├── experimento_descricao_v04.md   # Living research document
│   └── cronograma_v04.md             # Task tracker
│
├── results/                           # Experiment outputs
│   ├── scale_analysis/
│   │   └── escala_citra3d_report.json
│   ├── synthetic_generation/
│   │   └── composicao_report.json
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
- Google Colab Pro+ (A100 GPU recommended)
- Google Drive (~100 GB free space)

```bash
pip install ultralytics>=8.4 segment-anything opencv-python torch torchvision
```

### Datasets

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| **CITRA-3D-Real** | Brazilian Navy (CASNAV) | 2,081 images | Restricted — contact authors |
| **InaTechShips** | Teixeira et al. (2025) | ~28k images | [GitHub](https://github.com/EduardoHT/InaTechShips) / shipspotting.com |
| **SAM weights** | Meta AI | 375 MB | Auto-downloaded by scripts |

> **Note:** CITRA-3D-Real is an operational dataset from the Brazilian Navy and cannot be publicly distributed. Contact the authors for access under the CASNAV/DMarSup project (Termo 66/2025).

### Step-by-Step Reproduction

The experiment follows 5 stages. Each stage has numbered scripts that should be run in order.

#### Stage 1 — Data Preparation (`scripts/01_data_preparation/`)

Prepare the CITRA-3D-Real dataset: clean labels, generate single-class annotations, copy to local SSD.

```bash
python scripts/01_data_preparation/preparar_citra3d.py
python scripts/01_data_preparation/limpar_labels_citra3d.py
python scripts/01_data_preparation/gerar_labels_single_class.py
python scripts/01_data_preparation/preparar_dados_locais.py
```

#### Stage 2 — Dataset Acquisition (`scripts/02_dataset_acquisition/`)

Download and prepare InaTechShips subsets (curated and random).

```bash
# Reconstruct curated subset (dataset_25k_v2) with disjoint splits
python scripts/02_dataset_acquisition/reconstruir_dataset_25k.py

# Download random pool
python scripts/02_dataset_acquisition/baixar_random_pool_v2.py
python scripts/02_dataset_acquisition/validar_imagens_random_pool.py
python scripts/02_dataset_acquisition/downsample_random_pool_v2.py

# Filter labels (requires Eduardo's PointRend labels)
python scripts/02_dataset_acquisition/filtrar_labels_eduardo.py --labels-src /path/to/labels
```

#### Stage 3 — Analysis (`scripts/03_analysis/`)

Extract scale profile of CITRA-3D-Real to calibrate synthetic generation.

```bash
python scripts/03_analysis/analisar_escala_citra3d.py --plot
```

Output: `results/scale_analysis/escala_citra3d_report.json`

#### Stage 4 — Synthetic Generation (`scripts/04_synthetic_generation/`)

The core contribution: Scale-Aware Copy-Paste via in-place substitution.

```bash
# Step 1: Extract ship crops with SAM segmentation
python scripts/04_synthetic_generation/extrair_crops_sam.py --mode sam

# Step 2: Extract ocean backgrounds from CITRA-3D-Real
python scripts/04_synthetic_generation/extrair_fundos_citra3d.py

# Step 3: Generate synthetic dataset (in-place substitution)
python scripts/04_synthetic_generation/gerar_dataset_copypaste.py --preview  # test first
python scripts/04_synthetic_generation/gerar_dataset_copypaste.py            # full batch
```

Output: ~27,796 synthetic images with 93,480 objects (3.36 obj/img)

#### Stage 5 — Training (`scripts/05_training/`)

```bash
# Baselines (B1: random init, B2: COCO pre-trained)
python scripts/05_training/treinar_baselines.py --all

# HPO validation (30 Optuna trials)
python scripts/05_training/hpo_b2.py

# Arm A: curated direct pre-training (expected: negative transfer)
python scripts/05_training/treinar_braco_a.py --seeds 42 123 2024

# Ablation: epoch sweep (10, 20, 50 epochs)
python scripts/05_training/ablation_epocas_pretreino.py

# Arm A': synthetic pre-training (main result)
# Use treinar_braco_a.py with dataset_sintetico path
```

## Key Findings

### 1. Direct transfer fails (−4.15%)

Pre-training on InaTechShips causes catastrophic forgetting due to domain gap:
- **Scale:** ships occupy ~80% of image (InaTechShips) vs ~3% (CITRA-3D)
- **Density:** 1 object/image vs 3.37 objects/image
- **Context:** professional photos vs operational captures

### 2. The gap is structural, not curation-dependent

Random subset (B = 0.7997) ≈ curated subset (A = 0.7936), confirming that CLIP visual similarity is an insufficient proxy for transfer learning compatibility.

### 3. In-place composition resolves the gap (+1.90%)

Replacing real vessels with InaTechShips crops at the same position and scale produces a synthetic dataset that improves over COCO baseline with complete statistical separation.

## Citation

```bibtex
@article{freire2026maritime,
  title={Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection},
  author={Freire, Daniela L. and Teixeira, Eduardo H. and Moreira, Leandro A. S.},
  year={2026},
  note={In preparation}
}
```

### Related Work

```bibtex
@article{TEIXEIRA2025120823,
  title={InaTechShips: A validation study of a novel ship dataset through deep learning-based classification and detection models for maritime applications},
  journal={Ocean Engineering},
  volume={326},
  pages={120823},
  year={2025},
  doi={https://doi.org/10.1016/j.oceaneng.2025.120823},
  author={Eduardo H. Teixeira and Samuel B. Mafra and Felipe A.P. {De Figueiredo}}
}
```

## Authors

- **Daniela L. Freire** — ICMC/USP (lead researcher)
- **Eduardo H. Teixeira** — INATEL (dataset collaboration, co-author)
- **Leandro A. S. Moreira** — ICMC/USP (supervisor)

## Acknowledgments

This work is part of the CASNAV/DMarSup project (Termo 66/2025), Brazilian Navy. We thank the Navy for providing the CITRA-3D-Real dataset.

## License

MIT License — see [LICENSE](LICENSE) for details.
