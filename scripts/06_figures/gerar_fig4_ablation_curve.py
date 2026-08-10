#!/usr/bin/env python3
"""
gerar_fig4_ablation_curve.py — regenera a curva de ablação de épocas de
pré-treino (Fig. 4 do artigo; fig5 no gerador antigo) com a replicação
multi-seed pedida pelo Reviewer 2 (W3).

Mudanças vs a versão publicada:
  - Pontos 0, 10 e 100 épocas agora têm n=3 sementes -> barras de erro (±std,
    ddof=0, convenção do artigo) e valores atualizados para a MÉDIA das
    sementes (a curva antiga usava só a seed 42; no ponto de 100 ela era a
    semente mais otimista: 0.8006 vs média 0.7936).
  - Pontos 20 e 50 permanecem single-seed (marcador aberto, declarado na
    legenda), conforme a concessão explícita do revisor ("at least replicated
    at the 10 and 100 epoch endpoints").

Fontes de dados (proveniência, nada hardcoded):
  results/metricas_detalhadas.json          -> B2 (0 ep) e braço A (100 ep), per-seed
  <runs>/ablation_epochs_summary.json       -> 10/20/50 ep (o merge da rodada
                                               nova preserva a seed em cada
                                               variante)
Uso:
  python scripts/06_figures/gerar_fig4_ablation_curve.py \
      --summary /content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs/ablation_epochs/ablation_epochs_summary.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
COLORS = {"A_prime": "#2E86AB", "B2": "#A23B72"}
SINGLE_COL_W = 3.5


def seed_stats(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std()), len(v)   # ddof=0 (convenção)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="ablation_epochs_summary.json (rodada com merge)")
    ap.add_argument("--metricas", default="results/metricas_detalhadas.json")
    ap.add_argument("--out", default="figs/fig4_ablation_curve.pdf")
    args = ap.parse_args()

    md = json.load(open(args.metricas))
    summ = json.load(open(args.summary))

    # ── endpoints multi-seed dos resultados principais ───────────────────────
    data = {}   # epochs -> dict(map50=(mean,std,n), map5095=(mean,std,n))
    for ep, arm in ((0, "B2 (COCO)"), (100, "A (curated direct)")):
        m50 = [md[arm]["mAP50"]["per_seed"][s] for s in ("42", "123", "2024")]
        m95 = [md[arm]["mAP50-95"]["per_seed"][s] for s in ("42", "123", "2024")]
        data[ep] = dict(map50=seed_stats(m50), map5095=seed_stats(m95))

    # ── pontos intermediários do summary (10 tem n=3 após o merge) ───────────
    by_ep = {}
    for v in summ["variants"]:
        by_ep.setdefault(v["pretrain_epochs"], []).append(v)
    for ep, vs in sorted(by_ep.items()):
        m50 = [v["metrics"]["mAP50"] for v in vs]
        m95 = [v["metrics"]["mAP50-95"] for v in vs]
        data[ep] = dict(map50=seed_stats(m50), map5095=seed_stats(m95))
        seeds = sorted(v.get("seed", "?") for v in vs)
        print(f"  {ep:>3} ep: n={len(vs)} seeds={seeds}  "
              f"mAP50={data[ep]['map50'][0]:.4f}±{data[ep]['map50'][1]:.4f}")

    epochs = sorted(data)
    multi = [ep for ep in epochs if data[ep]["map50"][2] >= 3]
    single = [ep for ep in epochs if data[ep]["map50"][2] < 3]
    print(f"\nmulti-seed: {multi} | single-seed: {single}")

    # ── figura ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.8, 3.0))

    for key, color, ls, marker, label in (
            ("map50", COLORS["A_prime"], "-", "o", "mAP50"),
            ("map5095", COLORS["B2"], "--", "s", "mAP50-95")):
        means = [data[ep][key][0] for ep in epochs]
        ax.plot(epochs, means, ls, color=color, linewidth=1.8, zorder=3,
                label=label)
        # multi-seed: marcador cheio + barra de erro
        me = [data[ep][key][0] for ep in multi]
        se = [data[ep][key][1] for ep in multi]
        ax.errorbar(multi, me, yerr=se, fmt=marker, color=color,
                    markersize=6, capsize=3, linewidth=0, elinewidth=1.2,
                    zorder=5)
        # single-seed: marcador aberto
        ax.plot(single, [data[ep][key][0] for ep in single], marker,
                color=color, markersize=6, markerfacecolor="white",
                markeredgewidth=1.2, linestyle="none", zorder=5)

    # linhas de referência B1/B2 (como na versão publicada; a legenda as cita)
    from json import load as _jl
    md2 = _jl(open(args.metricas))
    for key, style in (("mAP50", dict(map50=True)), ("mAP50-95", dict(map50=False))):
        pass
    b1_50 = np.mean([md2["B1 (random init)"]["mAP50"]["per_seed"][s_]
                     for s_ in ("42", "123", "2024")])
    b1_95 = np.mean([md2["B1 (random init)"]["mAP50-95"]["per_seed"][s_]
                     for s_ in ("42", "123", "2024")])
    b2_50 = data[0]["map50"][0]
    for y, c in ((b2_50, COLORS["A_prime"]), (b1_50, "#F18F01"),
                 (b1_95, "#F18F01")):
        ax.axhline(y, color=c, linestyle=":", linewidth=0.9, alpha=0.6, zorder=1)
    ax.text(102, b2_50, "B2", fontsize=7, color=COLORS["A_prime"],
            va="center", alpha=0.8)
    ax.text(102, b1_50, "B1", fontsize=7, color="#F18F01", va="center", alpha=0.9)
    ax.text(102, b1_95, "B1", fontsize=7, color="#F18F01", va="center", alpha=0.9)
    ax.set_xlim(-4, 112)

    ax.set_xlabel("Pre-training epochs on InaTechShips")
    ax.set_ylabel("Score on CITRA-3D-Real test")
    ax.set_xticks(epochs)

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [
        Line2D([], [], marker="o", color="gray", linestyle="none",
               markersize=6, label="mean ± std (n=3 seeds)"),
        Line2D([], [], marker="o", color="gray", linestyle="none",
               markersize=6, markerfacecolor="white", label="single seed"),
    ]
    ax.legend(handles=handles, loc="center",
              bbox_to_anchor=(0.5, 0.5), framealpha=0.95)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"✓ {out}")

    print("\nValores para o texto/legenda:")
    for ep in epochs:
        m, s, n = data[ep]["map50"]
        tag = f"± {s:.4f} (n={n})" if n >= 3 else "(single seed)"
        print(f"  {ep:>3} ep: mAP50 = {m:.4f} {tag}")


if __name__ == "__main__":
    main()
