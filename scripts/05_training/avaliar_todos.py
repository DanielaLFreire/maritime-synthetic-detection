"""
avaliar_todos.py — Caminhos corrigidos

Avalia TODOS os braços experimentais no test set do CITRA-3D-Real.
Métricas: mAP50, mAP50-95, Precision, Recall, F1.
"""

from pathlib import Path
from ultralytics import YOLO
import numpy as np
import json
import time

CITRA_YAML = "/content/data/CITRA-3D-Real/data_single_class.yaml"
R = Path("/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs")

EXPERIMENTS = {
    # ── Baselines ──
    "B1 (random init)": {
        42:   R / "baselines/B1_random/seed_42/train/weights/best.pt",
        123:  R / "baselines/B1_random/seed_123/train/weights/best.pt",
        2024: R / "baselines/B1_random/seed_2024/train/weights/best.pt",
    },
    "B2 (COCO)": {
        42:   R / "baselines/B2_coco/seed_42/train/weights/best.pt",
        123:  R / "baselines/B2_coco/seed_123/train/weights/best.pt",
        2024: R / "baselines/B2_coco/seed_2024/train/weights/best.pt",
    },

    # ── Braço A (curado direto) ──
    "A (curated direct)": {
        42:   R / "braco_a/seed_0042/finetune/weights/best.pt",
        123:  R / "braco_a/seed_0123/finetune/weights/best.pt",
        2024: R / "braco_a/seed_2024/finetune/weights/best.pt",
    },

    # ── Braço B (aleatório) ──
    # NOTA: seed_0123_finetune contém seed 2024 (sobrescrito)
    "B (random direct)": {
        42:   R / "braco_b/seed_0042_finetune/weights/best.pt",
        # 123: PRECISA RE-RODAR
        2024: R / "braco_b/seed_0123_finetune/weights/best.pt",
    },

    # ── Ablation épocas InaTechShips direto ──
    "InaTech direct 10ep": {
        42: R / "ablation_epochs/ep_010/finetune/weights/best.pt",
    },
    "InaTech direct 20ep": {
        42: R / "ablation_epochs/ep_020/finetune/weights/best.pt",
    },
    "InaTech direct 50ep": {
        42: R / "ablation_epochs/ep_050/finetune/weights/best.pt",
    },

    # ── A' v4 (sintético sequencial 100ep) ──
    "A' v4 sequential (100ep)": {
        42:   R / "braco_a_sintetico_v4/seed_0042_finetune/weights/best.pt",
        123:  R / "braco_a_sintetico_v4/seed_0123_finetune/weights/best.pt",
        2024: R / "braco_a_sintetico_v4/seed_2024_finetune/weights/best.pt",
    },

    # ── Frozen backbone ──
    "Frozen backbone (100ep)": {
        42:   R / "braco_frozen/seed_0042_finetune/weights/best.pt",
        123:  R / "braco_frozen/seed_0123_finetune/weights/best.pt",
        2024: R / "braco_frozen/seed_2024_finetune/weights/best.pt",
    },

    # ── Ablation épocas sintético v4 ──
    "Synthetic 10ep": {
        42: R / "ablation_sintetico/ep010_finetune/weights/best.pt",
    },
    "Synthetic 20ep": {
        42:   R / "ablation_sintetico/ep020_finetune/weights/best.pt",
        123:  R / "ablation_sintetico/ep020_seed0123_finetune/weights/best.pt",
        2024: R / "ablation_sintetico/ep020_seed2024_finetune/weights/best.pt",
    },

    # ── Conjunto desbalanceado ──
    "Joint unbalanced (7/93)": {
        42: R / "braco_combined/seed_0042/weights/best.pt",
    },

    # ── Conjunto balanceado ──
    "Joint balanced (50/50)": {
        42:   R / "braco_balanced/seed_0042/weights/best.pt",
        123:  R / "braco_balanced/seed_0123/weights/best.pt",
        2024: R / "braco_balanced/seed_2024/weights/best.pt",
    },

}


def evaluate_model(best_pt, citra_yaml, device=0):
    model = YOLO(str(best_pt))
    metrics = model.val(data=citra_yaml, split="test", device=device, verbose=False)
    p = float(metrics.box.p.mean()) if hasattr(metrics.box.p, 'mean') else float(metrics.box.p)
    r = float(metrics.box.r.mean()) if hasattr(metrics.box.r, 'mean') else float(metrics.box.r)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"Precision": p, "Recall": r, "F1": f1, "mAP50": map50, "mAP50-95": map5095}


def main():
    print("=" * 100)
    print("  AVALIAÇÃO COMPLETA — TODOS OS BRAÇOS NO CITRA-3D-Real TEST SET")
    print("=" * 100)

    citra_test = Path("/content/data/CITRA-3D-Real/test/images")
    n_test = len(list(citra_test.glob("*"))) if citra_test.exists() else 0
    if n_test == 0:
        print("\n  ✗ CITRA-3D-Real test set não encontrado! Rode a célula de cópia primeiro.")
        return
    print(f"\n  ✓ Test set: {n_test} imagens\n")

    all_results = {}

    for exp_name, paths in EXPERIMENTS.items():
        print(f"{'─'*80}")
        print(f"  {exp_name}")
        seed_results = {}

        for seed, best_pt in paths.items():
            if not best_pt.exists():
                print(f"    Seed {seed}: ✗ não encontrado")
                continue
            try:
                result = evaluate_model(best_pt, CITRA_YAML)
                seed_results[seed] = result
                print(f"    Seed {seed}: mAP50={result['mAP50']:.4f}  "
                      f"mAP50-95={result['mAP50-95']:.4f}  "
                      f"P={result['Precision']:.4f}  R={result['Recall']:.4f}  "
                      f"F1={result['F1']:.4f}")
            except Exception as e:
                print(f"    Seed {seed}: ERRO — {e}")

        if seed_results:
            summary = {}
            for metric in ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]:
                vals = [r[metric] for r in seed_results.values()]
                summary[metric] = {"mean": np.mean(vals), "std": np.std(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
            all_results[exp_name] = {"per_seed": seed_results, "summary": summary}
            if len(seed_results) > 1:
                print(f"    Média ({len(seed_results)}s): "
                      f"mAP50={summary['mAP50']['mean']:.4f}±{summary['mAP50']['std']:.4f}  "
                      f"mAP50-95={summary['mAP50-95']['mean']:.4f}±{summary['mAP50-95']['std']:.4f}")

    # ── Tabela final ──
    print(f"\n\n{'=' * 100}")
    print(f"  TABELA FINAL")
    print(f"{'=' * 100}")

    header = f"{'Experiment':<30} {'#':>2} {'mAP50':>14} {'mAP50-95':>14} {'P':>8} {'R':>8} {'F1':>8} {'Δ mAP50':>9}"
    print(header)
    print("─" * len(header))

    b2_map50 = all_results.get("B2 (COCO)", {}).get("summary", {}).get("mAP50", {}).get("mean", 0.8351)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["summary"]["mAP50"]["mean"], reverse=True)

    for exp_name, data in sorted_results:
        s = data["summary"]
        n = s["mAP50"]["n"]
        m50 = f"{s['mAP50']['mean']:.4f}±{s['mAP50']['std']:.4f}" if n > 1 else f"{s['mAP50']['mean']:.4f}"
        m95 = f"{s['mAP50-95']['mean']:.4f}±{s['mAP50-95']['std']:.4f}" if n > 1 else f"{s['mAP50-95']['mean']:.4f}"
        delta = s["mAP50"]["mean"] - b2_map50
        d_str = f"{delta:+.4f}" if exp_name != "B2 (COCO)" else "ref"
        print(f"{exp_name:<30} {n:>2} {m50:>14} {m95:>14} "
              f"{s['Precision']['mean']:>8.4f} {s['Recall']['mean']:>8.4f} "
              f"{s['F1']['mean']:>8.4f} {d_str:>9}")

    # ── Salva JSON ──
    def native(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, dict): return {k: native(v) for k, v in obj.items()}
        if isinstance(obj, list): return [native(v) for v in obj]
        return obj

    out = R / "avaliacao_completa.json"
    with open(out, "w") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "test_set": "CITRA-3D-Real test (401 imgs)",
                    "reference": "B2 (COCO)", "ref_mAP50": b2_map50,
                    "results": native(all_results)}, f, indent=2)
    print(f"\n  ✓ Salvo: {out}")


if __name__ == "__main__":
    main()
