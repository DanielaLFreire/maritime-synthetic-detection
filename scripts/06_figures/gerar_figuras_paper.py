"""
gerar_figuras_paper.py

Gera todas as figuras para o paper "Visual Similarity Is Not Enough"
no padrão Ocean Engineering / Elsevier.

USO

  python gerar_figuras_paper.py                    # gera gráficos
  python gerar_figuras_paper.py --with-images      # gera tudo (requer acesso ao Drive)

SAÍDA

  figs/
  ├── fig2_domain_gap.pdf          (requer --with-images)
  ├── fig3_scale_distribution.pdf
  ├── fig4_composition_examples.pdf (requer --with-images)
  ├── fig5_ablation_curve.pdf
  ├── fig6_comparison_bar.pdf
  └── fig7_detection_examples.pdf   (requer --with-images)
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Estilo de publicação
# ═══════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
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

# Cores consistentes
COLORS = {
    "A_prime": "#2E86AB",   # azul — resultado principal
    "B2": "#A23B72",        # roxo — baseline
    "B1": "#F18F01",        # laranja
    "B": "#C73E1D",         # vermelho
    "A": "#3B1F2B",         # escuro
    "highlight": "#2E86AB",
    "neutral": "#888888",
}

FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# Larguras para Ocean Engineering double-column
SINGLE_COL_W = 3.5   # polegadas (1 coluna)
DOUBLE_COL_W = 7.2    # polegadas (2 colunas)


# ═══════════════════════════════════════════════════════════════════
# Fig. 3 — Distribuição de escala (histograma)
# ═══════════════════════════════════════════════════════════════════

def fig3_scale_distribution(citra_labels=None, inatech_labels=None):
    """Histograma com distribuição REAL de áreas dos bboxes."""
    
    def read_areas_from_labels(labels_dir):
        """Lê áreas normalizadas (w*h) de todos os labels YOLO."""
        areas = []
        labels_path = Path(labels_dir)
        for split in ("train", "val", "test"):
            split_dir = labels_path / split / "labels_single_class"
            if not split_dir.exists():
                split_dir = labels_path / split / "labels"
            if not split_dir.exists():
                continue
            for f in split_dir.glob("*.txt"):
                for line in open(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        areas.append(w * h)
        return areas

    # Lê dados reais
    citra_areas = read_areas_from_labels(citra_labels)
    inatech_areas = read_areas_from_labels(inatech_labels)
    
    print(f"   CITRA-3D: {len(citra_areas):,} bboxes")
    print(f"   InaTechShips: {len(inatech_areas):,} bboxes")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.8))

    # CITRA-3D
    ax1.hist(citra_areas, bins=50, color=COLORS["A_prime"], alpha=0.8,
             edgecolor="white", linewidth=0.3, range=(0, 0.05))
    ax1.set_xlabel("Normalized bbox area")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) CITRA-3D-Real")
    median_c = sorted(citra_areas)[len(citra_areas)//2]
    ax1.axvline(median_c, color="red", linestyle="--",
                linewidth=1, label=f"Median = {median_c:.4f}")
    ax1.legend()
    small_thresh = (32/640)**2
    ax1.axvline(small_thresh, color="gray", linestyle=":", linewidth=0.8)
    ax1.text(small_thresh + 0.001, ax1.get_ylim()[1]*0.9, "COCO\nsmall",
             fontsize=7, color="gray")

    # InaTechShips
    ax2.hist(inatech_areas, bins=50, color=COLORS["A"], alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Normalized bbox area")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) InaTechShips")
    median_i = sorted(inatech_areas)[len(inatech_areas)//2]
    ax2.axvline(median_i, color="red", linestyle="--",
                linewidth=1, label=f"Median = {median_i:.2f}")
    ax2.legend()

    plt.tight_layout()
    out = FIGS_DIR / "fig3_scale_distribution.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 5 — Curva de ablation de épocas
# ═══════════════════════════════════════════════════════════════════

def fig5_ablation_curve():
    """Curva de degradação monotônica com épocas de pré-treino."""
    epochs = [0, 10, 20, 50, 100]
    map50 = [0.8351, 0.8200, 0.8171, 0.8037, 0.8006]
    map5095 = [0.5055, 0.4999, 0.4960, 0.4731, 0.4680]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.8, 3.0))

    ax.plot(epochs, map50, "o-", color=COLORS["A_prime"], linewidth=2,
            markersize=6, label="mAP50", zorder=5)
    ax.plot(epochs, map5095, "s--", color=COLORS["B2"], linewidth=1.5,
            markersize=5, label="mAP50-95", zorder=4)

    # Baseline B2 reference line
    ax.axhline(0.8351, color=COLORS["neutral"], linestyle=":", linewidth=0.8,
               alpha=0.5)
    ax.text(102, 0.838, "B2", fontsize=7, color=COLORS["neutral"])

    # B1 reference line
    ax.axhline(0.8008, color=COLORS["B"], linestyle=":", linewidth=0.8,
               alpha=0.5)
    ax.text(102, 0.803, "B1", fontsize=7, color=COLORS["B"])

    ax.set_xlabel("Pre-training epochs on InaTechShips")
    ax.set_ylabel("mAP on CITRA-3D-Real test set")
    ax.set_xticks(epochs)
    ax.set_ylim(0.44, 0.86)
    ax.legend(loc="center right")

    # Annotation: monotonic degradation
    ax.annotate("Monotonic\ndegradation",
                xy=(50, 0.8037), xytext=(65, 0.82),
                fontsize=7, color=COLORS["A"],
                arrowprops=dict(arrowstyle="->", color=COLORS["A"],
                                lw=0.8))

    plt.tight_layout()
    out = FIGS_DIR / "fig5_ablation_curve.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 6 — Comparação final (bar chart)
# ═══════════════════════════════════════════════════════════════════

def fig6_comparison_bar():
    """Bar chart com todos os braços experimentais."""
    arms = ["A' (synthetic)", "B2 (COCO)", "B1 (random\ninit)",
            "B (random\npool)", "A (curated)"]
    map50 = [0.8541, 0.8351, 0.8008, 0.7997, 0.7936]
    map5095 = [0.5281, 0.5055, 0.4742, 0.4711, 0.4692]
    stds50 = [0.0043, 0.0024, 0.0073, 0, 0.0060]
    colors = [COLORS["A_prime"], COLORS["B2"], COLORS["B1"],
              COLORS["B"], COLORS["A"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 3.0))

    x = np.arange(len(arms))
    width = 0.6

    # mAP50
    bars1 = ax1.bar(x, map50, width, color=colors, edgecolor="white",
                    linewidth=0.5, yerr=stds50, capsize=3, error_kw={"linewidth": 0.8})
    ax1.set_ylabel("mAP50")
    ax1.set_title("(a) mAP50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms, fontsize=7)
    ax1.set_ylim(0.75, 0.88)
    ax1.axhline(0.8351, color=COLORS["neutral"], linestyle=":", linewidth=0.6,
                alpha=0.4)

    # Labels de valor
    for bar, val in zip(bars1, map50):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # mAP50-95
    bars2 = ax2.bar(x, map5095, width, color=colors, edgecolor="white",
                    linewidth=0.5)
    ax2.set_ylabel("mAP50-95")
    ax2.set_title("(b) mAP50-95")
    ax2.set_xticks(x)
    ax2.set_xticklabels(arms, fontsize=7)
    ax2.set_ylim(0.43, 0.56)
    ax2.axhline(0.5055, color=COLORS["neutral"], linestyle=":", linewidth=0.6,
                alpha=0.4)

    for bar, val in zip(bars2, map5095):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.tight_layout()
    out = FIGS_DIR / "fig6_comparison_bar.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 2 — Domain gap visual (requer imagens reais)
# ═══════════════════════════════════════════════════════════════════

def fig2_domain_gap(inatech_path=None, citra_path=None, synth_path=None):
    """
    Composição 3 imagens lado a lado mostrando o gap de domínio.
    
    Requer caminhos para:
    - Uma imagem do InaTechShips (navio grande, close-up)
    - Uma imagem do CITRA-3D-Real (navios pequenos, distantes)
    - Uma imagem sintética (composição in-place)
    """
    if not all([inatech_path, citra_path, synth_path]):
        print("  ⊘ fig2_domain_gap: precisa de --with-images e caminhos reais")
        return

    from PIL import Image

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.4))

    for ax, path, title in zip(axes,
                                [inatech_path, citra_path, synth_path],
                                ["(a) InaTechShips\n(public, close-range)",
                                 "(b) CITRA-3D-Real\n(operational, distant)",
                                 "(c) Synthetic\n(in-place composition)"]):
        img = Image.open(path).convert("RGB").resize((640, 640))
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    out = FIGS_DIR / "fig2_domain_gap.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 4 — Composição in-place (antes → depois)
# ═══════════════════════════════════════════════════════════════════

def fig4_composition_examples(citra_dir=None, synth_dir=None):
    """
    2x2 grid: 2 imagens originais do CITRA-3D (cima) e suas versões
    sintéticas correspondentes (baixo), com bboxes desenhados.
    """
    if not all([citra_dir, synth_dir]):
        print("  ⊘ fig4_composition_examples: precisa de --with-images")
        return

    # TODO: implementar com imagens reais do Drive
    print("  ⊘ fig4_composition_examples: implementar com imagens do Drive")


# ═══════════════════════════════════════════════════════════════════
# Fig. 1 — Pipeline (diagrama)
# ═══════════════════════════════════════════════════════════════════

def fig1_pipeline_placeholder():
    """
    Placeholder para o diagrama de pipeline.
    
    Sugestão: criar no draw.io ou TikZ com os seguintes blocos:
    
    InaTechShips (28k imgs)
         │
         ▼
    SAM Segmentation → Ship Crops (23,828 RGBA)
                              │
    CITRA-3D-Real (2,081 imgs)│
         │                    │
         ▼                    ▼
    Extract Backgrounds  →  In-Place Substitution
         │                    │
         │                    ▼
         │              Synthetic Dataset (27,796 imgs)
         │                    │
         ▼                    ▼
    [Fine-tuning]  ←  [Pre-training on synthetic]
         │
         ▼
    YOLOv11m Detection (mAP50 = 0.854)
    """
    print("  ⊘ fig1_pipeline: criar em draw.io ou TikZ (ver docstring para layout)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-images", action="store_true",
                        help="Gera figuras que requerem imagens reais")
    parser.add_argument("--inatech-img", type=Path, default=None)
    parser.add_argument("--citra-img", type=Path, default=None)
    parser.add_argument("--synth-img", type=Path, default=None)
    parser.add_argument("--citra-labels", type=Path, default=None,
                        help="Root do CITRA-3D-Real (com train/val/test)")
    parser.add_argument("--inatech-labels", type=Path, default=None,
                        help="Root do dataset_25k_v2 (com train/val/test)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gerando figuras para o paper")
    print("=" * 60)

    # Gráficos (não precisam de imagens reais)
    if args.citra_labels and args.inatech_labels:
        fig3_scale_distribution(args.citra_labels, args.inatech_labels)
    else:
        print("  ⊘ fig3: sem --citra-labels e --inatech-labels, pulando distribuição real")

    fig5_ablation_curve()
    fig6_comparison_bar()

    # Pipeline (placeholder)
    fig1_pipeline_placeholder()

    # Figuras com imagens reais
    if args.with_images:
        fig2_domain_gap(args.inatech_img, args.citra_img, args.synth_img)
        fig4_composition_examples()

    print(f"\n  Figuras salvas em: {FIGS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
