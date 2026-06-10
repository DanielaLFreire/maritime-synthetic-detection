"""
gerar_fig6_comparison.py

Generate Figure 4 / Figure 6 (comparison bar chart) for the paper.

UPDATE v2 (10/06/2026): B2-long now has n=3 (was n=1). Error bars added.
The dagger footnote is no longer needed.

Outputs:
  paper/figs/fig6_comparison_bar.pdf       (single-panel, mAP50)
  paper/figs/fig6_comparison_bar_dual.pdf  (dual-panel, mAP50 + mAP50-95)
  paper/figs/fig6_comparison_bar.png       (preview for README)
  paper/figs/fig6_comparison_bar_dual.png  (preview)

Usage:
  python scripts/06_figures/gerar_fig6_comparison.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# DATA — Final results, test set CITRA-3D-Real (n=3 seeds, all arms)
# Source: results/metricas_detalhadas.json + results/results_joint_rand_n3.json
#         + results/results_b2_long_n3.json (NEW)
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
    ("B2-long",          0.8351,     0.0025,    0.5100,    0.0038,    "baseline",      3),  # UPDATED: n=3
    ("A' frozen",        0.8342,     0.0039,    0.5074,    0.0035,    "baseline",      3),
    ("A' seq",           0.8221,     0.0085,    0.4933,    0.0059,    "baseline",      3),
    ("B1",               0.8008,     0.0061,    0.4742,    0.0006,    "baseline",      3),
    ("B",                0.7945,     0.0046,    0.4728,    0.0045,    "baseline",      3),
    ("A",                0.7936,     0.0049,    0.4692,    0.0017,    "baseline",      3),
]

COLOURS = {
    "proposed":      "#1f7a3e",   # dark green
    "proposed_var":  "#5bb87b",   # light green
    "baseline_ref":  "#e07b00",   # orange
    "baseline":      "#7d7d7d",   # neutral grey
}

EDGE_COLOURS = {
    "proposed":      "#0e4a25",
    "proposed_var":  "#3e8757",
    "baseline_ref":  "#9c5500",
    "baseline":      "#4d4d4d",
}


# ═══════════════════════════════════════════════════════════════════
# STYLE
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
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ═══════════════════════════════════════════════════════════════════
# SINGLE PANEL
# ═══════════════════════════════════════════════════════════════════

def plot_fig6_single(arms=ARMS, save_path=None, show=False):
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    arms_sorted = sorted(arms, key=lambda x: -x[1])
    labels      = [a[0] for a in arms_sorted]
    means       = [a[1] for a in arms_sorted]
    stds        = [a[2] for a in arms_sorted]
    categories  = [a[5] for a in arms_sorted]

    colours      = [COLOURS[c]      for c in categories]
    edge_colours = [EDGE_COLOURS[c] for c in categories]

    b2_value = next(a[1] for a in arms if a[0] == "B2")

    x = np.arange(len(labels))
    bar_width = 0.62

    ax.bar(
        x, means, yerr=stds, capsize=3,
        color=colours, edgecolor=edge_colours, linewidth=1.0,
        width=bar_width,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333"},
    )

    ax.axhline(
        b2_value, color=COLOURS["baseline_ref"],
        linestyle=":", linewidth=1.1, alpha=0.8,
        label=f"B2 reference ({b2_value:.4f})",
        zorder=0,
    )

    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.003, f"{m:.4f}",
                ha="center", va="bottom", fontsize=8, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mAP50 (CITRA-3D-Real test set)")
    ymin = min(means) - 0.02
    ymax = max(m + s for m, s in zip(means, stds)) + 0.015
    ax.set_ylim(ymin, ymax)

    legend_handles = [
        mpatches.Patch(color=COLOURS["proposed"],     label="A' joint (proposed)"),
        mpatches.Patch(color=COLOURS["proposed_var"], label="A' joint-rand (ablation)"),
        mpatches.Patch(color=COLOURS["baseline_ref"], label="B2 reference"),
        mpatches.Patch(color=COLOURS["baseline"],     label="Other baselines"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.95,
              edgecolor="#bbb", fancybox=False)

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
# DUAL PANEL (recommended for the paper)
# ═══════════════════════════════════════════════════════════════════

def plot_fig6_dual(arms=ARMS, save_path=None, show=False):
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

    arms_sorted = sorted(arms, key=lambda x: -x[1])
    labels      = [a[0] for a in arms_sorted]
    categories  = [a[5] for a in arms_sorted]
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
            x, means, yerr=stds, capsize=3,
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
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.003, f"{m:.4f}",
                    ha="center", va="bottom", fontsize=7.5, color="#222")
        ax.set_ylabel(metric_name)
        ymin = min(means) - 0.02
        ymax = max(m + s for m, s in zip(means, stds)) + 0.015
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)

    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    axes[-1].set_xlabel("Experiment arm")

    legend_handles = [
        mpatches.Patch(color=COLOURS["proposed"],     label="A' joint (proposed)"),
        mpatches.Patch(color=COLOURS["proposed_var"], label="A' joint-rand (ablation)"),
        mpatches.Patch(color=COLOURS["baseline_ref"], label="B2 reference"),
        mpatches.Patch(color=COLOURS["baseline"],     label="Other baselines"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower left",
                   framealpha=0.95, edgecolor="#bbb", fancybox=False)

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
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = Path("paper/figs")

    plot_fig6_single(save_path=out_dir / "fig6_comparison_bar.pdf")
    plot_fig6_dual(save_path=out_dir / "fig6_comparison_bar_dual.pdf")

    print("\nDone. Files generated in paper/figs/:")
    print("  - fig6_comparison_bar.pdf       (single panel, mAP50)")
    print("  - fig6_comparison_bar.png       (preview)")
    print("  - fig6_comparison_bar_dual.pdf  (dual panel, mAP50 + mAP50-95)")
    print("  - fig6_comparison_bar_dual.png  (preview)")
