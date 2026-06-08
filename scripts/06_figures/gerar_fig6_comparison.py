"""
gerar_fig6_comparison.py

Generate Figure 6 (comparison bar chart) for the paper, including the new
A' joint-rand arm.

Outputs:
  paper/figs/fig6_comparison_bar.pdf       (single-panel, mAP50)
  paper/figs/fig6_comparison_bar_dual.pdf  (dual-panel, mAP50 + mAP50-95)
  paper/figs/fig6_comparison_bar.png       (preview for README)

Usage:
  python scripts/06_figures/gerar_fig6_comparison.py

Or import individual functions:
  from gerar_fig6_comparison import plot_fig6_single, plot_fig6_dual
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# DATA — Final results, test set CITRA-3D-Real (n=3 seeds unless noted)
# Source: results/metricas_detalhadas.json + results/results_joint_rand_n3.json
# ═══════════════════════════════════════════════════════════════════

# Each entry: (label, mAP50_mean, mAP50_std, mAP95_mean, mAP95_std, category, n)
# Categories:
#   'proposed'      — A' joint (main proposed method) — dark green
#   'proposed_var'  — A' joint-rand (variant of proposed)  — light green
#   'baseline_ref'  — B2 (COCO reference baseline) — orange
#   'baseline'      — B1, A, B, B2-long, A' frozen, A' seq — gray

ARMS = [
    # (label,            mAP50_mean, mAP50_std, mAP95_mean, mAP95_std, category,        n)
    ("A' joint-rand",    0.8457,     0.0058,    0.5208,    0.0024,    "proposed_var",  3),
    ("A' joint",         0.8451,     0.0033,    0.5206,    0.0017,    "proposed",      3),
    ("B2",               0.8351,     0.0020,    0.5055,    0.0022,    "baseline_ref",  3),
    ("A' frozen",        0.8342,     0.0039,    0.5074,    0.0035,    "baseline",      3),
    ("B2-long",          0.8324,     0.0000,    0.5108,    0.0000,    "baseline",      1),
    ("A' seq",           0.8221,     0.0085,    0.4933,    0.0059,    "baseline",      3),
    ("B1",               0.8008,     0.0061,    0.4742,    0.0006,    "baseline",      3),
    ("B",                0.7945,     0.0046,    0.4728,    0.0045,    "baseline",      3),
    ("A",                0.7936,     0.0049,    0.4692,    0.0017,    "baseline",      3),
]

COLOURS = {
    "proposed":      "#1f7a3e",   # dark green   — main method (A' joint)
    "proposed_var":  "#5bb87b",   # light green  — variant     (A' joint-rand)
    "baseline_ref":  "#e07b00",   # orange       — B2 reference
    "baseline":      "#7d7d7d",   # neutral grey — other arms
}

EDGE_COLOURS = {
    "proposed":      "#0e4a25",
    "proposed_var":  "#3e8757",
    "baseline_ref":  "#9c5500",
    "baseline":      "#4d4d4d",
}


# ═══════════════════════════════════════════════════════════════════
# STYLE — Elsevier-friendly defaults
# ═══════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         10,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "pdf.fonttype":      42,    # editable text in PDF
        "ps.fonttype":       42,
    })


# ═══════════════════════════════════════════════════════════════════
# PLOT — Single panel (mAP50 only) — default Figure 6
# ═══════════════════════════════════════════════════════════════════

def plot_fig6_single(arms=ARMS, save_path=None, show=False):
    """Single-panel bar chart of mAP50, ordered by value descending."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Sort by mAP50 descending (proposed methods naturally go to top)
    arms_sorted = sorted(arms, key=lambda x: -x[1])

    labels      = [a[0] for a in arms_sorted]
    means       = [a[1] for a in arms_sorted]
    stds        = [a[2] for a in arms_sorted]
    categories  = [a[5] for a in arms_sorted]
    ns          = [a[6] for a in arms_sorted]

    colours      = [COLOURS[c]      for c in categories]
    edge_colours = [EDGE_COLOURS[c] for c in categories]

    # Reference line at B2
    b2_value = next(a[1] for a in arms if a[0] == "B2")

    x = np.arange(len(labels))
    bar_width = 0.62

    bars = ax.bar(
        x, means,
        yerr=stds,
        capsize=3,
        color=colours,
        edgecolor=edge_colours,
        linewidth=1.0,
        width=bar_width,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333"},
    )

    # Reference line
    ax.axhline(
        b2_value, color=COLOURS["baseline_ref"],
        linestyle=":", linewidth=1.1, alpha=0.8,
        label=f"B2 reference ({b2_value:.4f})",
        zorder=0,
    )

    # Value annotations on top of bars
    for xi, (m, s, n) in enumerate(zip(means, stds, ns)):
        text = f"{m:.4f}"
        if n == 1:
            text += "$^\\dagger$"
        ax.text(
            xi, m + s + 0.003, text,
            ha="center", va="bottom",
            fontsize=8, color="#222",
        )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mAP50 (CITRA-3D-Real test set)")
    ymin = min(means) - 0.02
    ymax = max(m + s for m, s in zip(means, stds)) + 0.015
    ax.set_ylim(ymin, ymax)

    # Legend (category-based)
    legend_handles = [
        mpatches.Patch(color=COLOURS["proposed"],     label="A' joint (proposed)"),
        mpatches.Patch(color=COLOURS["proposed_var"], label="A' joint-rand (ablation)"),
        mpatches.Patch(color=COLOURS["baseline_ref"], label="B2 reference"),
        mpatches.Patch(color=COLOURS["baseline"],     label="Other baselines"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.95,
              edgecolor="#bbb", fancybox=False)

    # Footer note about single-seed arm
    has_single_seed = any(n == 1 for n in ns)
    if has_single_seed:
        fig.text(
            0.99, 0.01, r"$^\dagger$ Single seed (42).",
            ha="right", va="bottom", fontsize=8, color="#555",
        )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved {save_path}")
        # Also save PNG preview
        png_path = save_path.with_suffix(".png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"✓ Saved {png_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ═══════════════════════════════════════════════════════════════════
# PLOT — Dual panel (mAP50 + mAP50-95) — recommended for new Figure 6
# Useful because the central finding of Section 5.5 ("anchoring affects
# localisation but not detection") only emerges from comparing the two
# metrics side by side.
# ═══════════════════════════════════════════════════════════════════

def plot_fig6_dual(arms=ARMS, save_path=None, show=False):
    """Dual-panel bar chart: mAP50 (top) and mAP50-95 (bottom)."""
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

    # Sort by mAP50 (consistent ordering between panels)
    arms_sorted = sorted(arms, key=lambda x: -x[1])
    labels      = [a[0] for a in arms_sorted]
    categories  = [a[5] for a in arms_sorted]
    ns          = [a[6] for a in arms_sorted]
    colours      = [COLOURS[c]      for c in categories]
    edge_colours = [EDGE_COLOURS[c] for c in categories]

    metrics = [
        ("mAP50",    [a[1] for a in arms_sorted], [a[2] for a in arms_sorted]),
        ("mAP50-95", [a[3] for a in arms_sorted], [a[4] for a in arms_sorted]),
    ]
    b2_values = {
        "mAP50":    next(a[1] for a in arms if a[0] == "B2"),
        "mAP50-95": next(a[3] for a in arms if a[0] == "B2"),
    }

    x = np.arange(len(labels))
    bar_width = 0.62

    for ax, (metric_name, means, stds) in zip(axes, metrics):
        ax.bar(
            x, means,
            yerr=stds, capsize=3,
            color=colours, edgecolor=edge_colours, linewidth=1.0,
            width=bar_width,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333"},
        )
        ax.axhline(
            b2_values[metric_name], color=COLOURS["baseline_ref"],
            linestyle=":", linewidth=1.1, alpha=0.8,
            label=f"B2 reference ({b2_values[metric_name]:.4f})",
            zorder=0,
        )
        for xi, (m, s, n) in enumerate(zip(means, stds, ns)):
            text = f"{m:.4f}"
            if n == 1:
                text += "$^\\dagger$"
            ax.text(
                xi, m + s + 0.003, text,
                ha="center", va="bottom",
                fontsize=7.5, color="#222",
            )
        ax.set_ylabel(metric_name)
        ymin = min(means) - 0.02
        ymax = max(m + s for m, s in zip(means, stds)) + 0.015
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)

    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    axes[-1].set_xlabel("Experiment arm")

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COLOURS["proposed"],     label="A' joint (proposed)"),
        mpatches.Patch(color=COLOURS["proposed_var"], label="A' joint-rand (ablation)"),
        mpatches.Patch(color=COLOURS["baseline_ref"], label="B2 reference"),
        mpatches.Patch(color=COLOURS["baseline"],     label="Other baselines"),
    ]
    axes[0].legend(
        handles=legend_handles, loc="lower left",
        framealpha=0.95, edgecolor="#bbb", fancybox=False,
    )

    has_single_seed = any(n == 1 for n in ns)
    if has_single_seed:
        fig.text(
            0.99, 0.01, r"$^\dagger$ Single seed (42).",
            ha="right", va="bottom", fontsize=8, color="#555",
        )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved {save_path}")
        png_path = save_path.with_suffix(".png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"✓ Saved {png_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN — Generate both versions
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Output directory — assumes script is run from repo root
    out_dir = Path("paper/figs")

    # Default: single-panel mAP50 (the current Figure 6 style)
    plot_fig6_single(save_path=out_dir / "fig6_comparison_bar.pdf")

    # Recommended for the updated paper: dual-panel mAP50 + mAP50-95
    plot_fig6_dual(save_path=out_dir / "fig6_comparison_bar_dual.pdf")

    print()
    print("Done. Files generated in paper/figs/:")
    print("  - fig6_comparison_bar.pdf       (single panel, mAP50)")
    print("  - fig6_comparison_bar.png       (preview)")
    print("  - fig6_comparison_bar_dual.pdf  (dual panel, mAP50 + mAP50-95)")
    print("  - fig6_comparison_bar_dual.png  (preview)")
    print()
    print("Choose ONE of the two PDFs for the manuscript and reference it")
    print("in paper/main.tex. Recommendation: dual panel (highlights the")
    print("anchoring ablation finding from Section 5.5).")
