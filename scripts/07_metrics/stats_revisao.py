#!/usr/bin/env python3
"""
stats_revisao.py — testes de significancia pedidos por R1.3 e R2.W2,
computados a partir dos dados por seed JA existentes no repositorio
maritime-synthetic-detection (nenhum treino novo necessario).

Fontes:
  results/metricas_detalhadas.json   -> B2, A' joint, A' frozen, A' seq, A, B, B1
  results/results_joint_rand_n3.json -> A' joint-rand
  results/results_b2_long_n3.json    -> B2-long
  results/smd_crossdomain.json       -> zero-shot SMD
"""
import json
import numpy as np
from scipy import stats

R = "results/"   # rodar da raiz do repo maritime-synthetic-detection
SEEDS = [42, 123, 2024]

md = json.load(open(R + "metricas_detalhadas.json"))
jr = json.load(open(R + "results_joint_rand_n3.json"))
bl = json.load(open(R + "results_b2_long_n3.json"))
smd = json.load(open(R + "smd_crossdomain.json"))


def from_md(arm, metric):
    return np.array([md[arm][metric]["per_seed"][str(s)] for s in SEEDS])


# --- monta o dicionario de series por seed (ordem 42, 123, 2024) ---
D = {
    ("B2", "mAP50"): from_md("B2 (COCO)", "mAP50"),
    ("B2", "mAP50-95"): from_md("B2 (COCO)", "mAP50-95"),
    ("B2", "Recall"): from_md("B2 (COCO)", "Recall"),
    ("A'joint", "mAP50"): from_md("A' joint balanced", "mAP50"),
    ("A'joint", "mAP50-95"): from_md("A' joint balanced", "mAP50-95"),
    ("A'joint", "Recall"): from_md("A' joint balanced", "Recall"),
    ("A'seq", "mAP50"): from_md("A' sequential 100ep", "mAP50"),
    ("A", "mAP50"): from_md("A (curated direct)", "mAP50"),
    ("B", "mAP50"): from_md("B (random direct)", "mAP50"),
    ("A'joint-rand", "mAP50"): np.array([d["mAP50"] for d in jr["per_seed"]]),
    ("A'joint-rand", "mAP50-95"): np.array([d["mAP50_95"] for d in jr["per_seed"]]),
    ("A'joint-rand", "Recall"): np.array([d["recall"] for d in jr["per_seed"]]),
    ("B2-long", "mAP50"): np.array([d["mAP50"] for d in bl["per_seed"]]),
    ("B2-long", "mAP50-95"): np.array([d["mAP50_95"] for d in bl["per_seed"]]),
    ("B2-long", "Recall"): np.array([d["R"] for d in bl["per_seed"]]),
    # zero-shot SMD (mesma ordem de seeds)
    ("SMD B2", "mAP50"): np.array(smd["B2 (COCO)"]["mAP50"]),
    ("SMD A'joint", "mAP50"): np.array(smd["A' joint (proposto)"]["mAP50"]),
    ("SMD A'joint-rand", "mAP50"): np.array(smd["A' joint-rand (ablacao)"]["mAP50"]),
    ("SMD A", "mAP50"): np.array(smd["A (InaTech curated)"]["mAP50"]),
}


def compare(a, b, metric, label=""):
    x, y = D[(a, metric)], D[(b, metric)]
    diff = x - y
    t_p, p_p = stats.ttest_rel(x, y)          # pareado por seed
    t_w, p_w = stats.ttest_ind(x, y, equal_var=False)   # Welch
    # permutacao exata por troca de sinais: com n=3 sao 8 permutacoes,
    # logo o menor p possivel e 2/8 = 0.25. Reportado apenas como sanidade;
    # o teste principal e o t pareado.
    from itertools import product
    perm = [np.dot(sg, np.abs(diff)) for sg in product((1, -1), repeat=3)]
    p_perm = np.mean([abs(v) >= abs(diff.sum()) for v in perm])
    coh = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.inf
    print(f"{label or a+' vs '+b:26} {metric:9} "
          f"delta={diff.mean()*100:+.2f} pp  "
          f"[{diff.min()*100:+.2f},{diff.max()*100:+.2f}]  "
          f"pareado t={t_p:6.2f} p={p_p:.4f}  "
          f"Welch p={p_w:.4f}  perm p={p_perm:.3f}  dz={coh:5.2f}  "
          f"{'3/3' if (diff > 0).all() else str((diff > 0).sum())+'/3'}")


print("=" * 130)
print("IN-DOMAIN (CITRA-3D-Real test) — n=3 seeds, pareado por seed")
print("=" * 130)
compare("A'joint", "B2", "mAP50")
compare("A'joint", "B2", "mAP50-95")
compare("A'joint", "B2", "Recall")
compare("A'joint", "B2-long", "mAP50")
compare("A'joint-rand", "B2", "mAP50")
compare("A'joint-rand", "A'joint", "mAP50", "A'joint-rand vs A'joint")
compare("A'joint-rand", "A'joint", "Recall", "A'joint-rand vs A'joint")
compare("B2-long", "B2", "mAP50")
compare("A", "B2", "mAP50")
compare("B", "B2", "mAP50")
compare("A", "B", "mAP50", "A vs B (curada vs aleat.)")
compare("A'seq", "B2", "mAP50")

print()
print("=" * 130)
print("ZERO-SHOT (SMD on-shore)")
print("=" * 130)
compare("SMD A'joint", "SMD B2", "mAP50")
compare("SMD A'joint-rand", "SMD B2", "mAP50")
compare("SMD A'joint-rand", "SMD A'joint", "mAP50", "SMD rand vs joint")
compare("SMD A", "SMD B2", "mAP50")

print()
print("Valores por seed (42, 123, 2024) — mAP50 in-domain:")
for arm in ["B2", "B2-long", "A'joint", "A'joint-rand", "A'seq", "A", "B"]:
    v = D[(arm, "mAP50")]
    print(f"  {arm:14} {v[0]:.4f}  {v[1]:.4f}  {v[2]:.4f}   "
          f"mean={v.mean():.4f} std(ddof=0)={v.std():.4f}")
