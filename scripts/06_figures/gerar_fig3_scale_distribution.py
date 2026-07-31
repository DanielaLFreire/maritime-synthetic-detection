#!/usr/bin/env python3
"""
gerar_fig3_scale_distribution.py — regenera a Fig. 3 (distribuição de escala)
com os dados CORRETOS do conjunto de pré-treino.

Motivo (correção C11): a versão publicada anota "median ~54%" no lado
InaTechShips, valor que não se reproduz. A mediana medida no conjunto real de
pré-treino do braço A (dataset_25k_v2, n=27.796, script clip_decile_profile.py)
é ~38,9%. Esta figura lê a distribuição diretamente do CSV com proveniência
(results/clip_structural_profile.csv) em vez de recalcular de labels avulsos.

Entradas:
  --inatech-csv  results/clip_structural_profile.csv  (coluna area_frac;
                 1 embarcação/imagem no conjunto público, logo área por
                 bbox = área por imagem)
  --citra-labels raiz do CITRA-3D-Real com {train,val,test}/{labels*}
                 (lê w*h normalizados de todos os splits — 7.003 bboxes)

Saída:
  figs/fig3_scale_distribution.pdf  (vetorial, 300 dpi, Liberation Serif)

Uso no Colab (CITRA já copiado local, repo clonado):
  python scripts/06_figures/gerar_fig3_scale_distribution.py \
      --citra-labels /content/data/CITRA-3D-Real \
      --inatech-csv results/clip_structural_profile.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── estilo do artigo ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = {"A_prime": "#2E86AB", "A": "#3B1F2B"}
DOUBLE_COL_W = 7.2

LABEL_DIR_CANDIDATES = ("labels_single_class", "labels", "labels_cleaned")


def read_citra_areas(root):
    """Lê áreas normalizadas (w*h) de todos os splits do CITRA."""
    root = Path(root)
    areas = []
    for split in ("train", "val", "test"):
        split_dir = None
        for cand in LABEL_DIR_CANDIDATES:
            d = root / split / cand
            if d.is_dir():
                split_dir = d
                break
        if split_dir is None:
            continue
        for f in split_dir.glob("*.txt"):
            for line in open(f):
                p = line.split()
                if len(p) >= 5:
                    areas.append(float(p[3]) * float(p[4]))
    return np.array(areas)


def read_inatech_areas(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df["area_frac"].dropna().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--citra-labels", default="/content/data/CITRA-3D-Real")
    ap.add_argument("--inatech-csv",
                    default="results/clip_structural_profile.csv")
    ap.add_argument("--out", default="figs/fig3_scale_distribution.pdf")
    args = ap.parse_args()

    citra = read_citra_areas(args.citra_labels)
    inatech = read_inatech_areas(args.inatech_csv)
    if len(citra) == 0:
        raise SystemExit(f"[erro] nenhum label do CITRA em {args.citra_labels} "
                         f"(candidatas: {LABEL_DIR_CANDIDATES})")

    med_c, med_i = float(np.median(citra)), float(np.median(inatech))
    print(f"CITRA-3D:     {len(citra):,} bboxes | mediana = {med_c:.6f} "
          f"({med_c*100:.2f}% da imagem)")
    print(f"InaTechShips: {len(inatech):,} bboxes | mediana = {med_i:.4f} "
          f"({med_i*100:.1f}% da imagem)")
    print(f"Razão de escala (mediana InaTech / mediana CITRA): "
          f"{med_i/med_c:,.0f}×")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.8))

    # (a) CITRA
    ax1.hist(citra, bins=50, color=COLORS["A_prime"], alpha=0.8,
             edgecolor="white", linewidth=0.3, range=(0, 0.05))
    ax1.set_xlabel("Normalized bbox area")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) CITRA-3D-Real")
    ax1.axvline(med_c, color="red", linestyle="--", linewidth=1,
                label=f"Median = {med_c:.4f}")
    ax1.legend()
    small_thresh = (32 / 640) ** 2
    ax1.axvline(small_thresh, color="gray", linestyle=":", linewidth=0.8)
    ax1.text(small_thresh + 0.001, ax1.get_ylim()[1] * 0.9, "COCO\nsmall",
             fontsize=7, color="gray")

    # (b) InaTechShips
    ax2.hist(inatech, bins=50, color=COLORS["A"], alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Normalized bbox area")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) InaTechShips (pre-training set)")
    ax2.axvline(med_i, color="red", linestyle="--", linewidth=1,
                label=f"Median = {med_i:.2f}")
    ax2.legend()

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"✓ {out}")
    print("\nLembretes de texto (C11): atualizar '~54%' para "
          f"'~{med_i*100:.0f}%' no abstract, related work, Tabela I e "
          "legenda da Fig. 3.")


if __name__ == "__main__":
    main()
