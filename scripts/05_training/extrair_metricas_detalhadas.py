"""
extrair_metricas_detalhadas.py

Extrai métricas faltantes para a revisão:
1. P, R, F1 com desvio-padrão por seed
2. AP small / medium / large
3. Nº de steps por época de cada braço (para comparabilidade)
4. Threshold de confiança usado pelo Ultralytics
"""

from pathlib import Path
from ultralytics import YOLO
import numpy as np
import json

CITRA_YAML = "/content/data/CITRA-3D-Real/data_single_class.yaml"
R = Path("/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs")

# Braços principais (3 seeds cada)
MAIN_ARMS = {
    "B2 (COCO)": {
        42:   R / "baselines/B2_coco/seed_42/train/weights/best.pt",
        123:  R / "baselines/B2_coco/seed_123/train/weights/best.pt",
        2024: R / "baselines/B2_coco/seed_2024/train/weights/best.pt",
    },
    "A' joint balanced": {
        42:   R / "braco_balanced/seed_0042/weights/best.pt",
        123:  R / "braco_balanced/seed_0123/weights/best.pt",
        2024: R / "braco_balanced/seed_2024/weights/best.pt",
    },
    "A' frozen backbone": {
        42:   R / "braco_frozen/seed_0042_finetune/weights/best.pt",
        123:  R / "braco_frozen/seed_0123_finetune/weights/best.pt",
        2024: R / "braco_frozen/seed_2024_finetune/weights/best.pt",
    },
    "A' sequential 100ep": {
        42:   R / "braco_a_sintetico_v4/seed_0042_finetune/weights/best.pt",
        123:  R / "braco_a_sintetico_v4/seed_0123_finetune/weights/best.pt",
        2024: R / "braco_a_sintetico_v4/seed_2024_finetune/weights/best.pt",
    },
    "A (curated direct)": {
        42:   R / "braco_a/seed_0042/finetune/weights/best.pt",
        123:  R / "braco_a/seed_0123/finetune/weights/best.pt",
        2024: R / "braco_a/seed_2024/finetune/weights/best.pt",
    },
    "B (random direct)": {
        42:   R / "braco_b/seed_0042_finetune/weights/best.pt",
        123:  R / "braco_b/seed_0123_finetune/weights/best.pt",
        2024: R / "braco_b/seed_2024_finetune/weights/best.pt",
    },
    "B1 (random init)": {
        42:   R / "baselines/B1_random/seed_42/train/weights/best.pt",
        123:  R / "baselines/B1_random/seed_123/train/weights/best.pt",
        2024: R / "baselines/B1_random/seed_2024/train/weights/best.pt",
    },
}

def main():
    print("=" * 100)
    print("  MÉTRICAS DETALHADAS — REVISÃO")
    print("=" * 100)

    # Verifica CITRA local
    citra_test = Path("/content/data/CITRA-3D-Real/test/images")
    n_test = len(list(citra_test.glob("*"))) if citra_test.exists() else 0
    if n_test == 0:
        print("✗ CITRA-3D-Real test não encontrado! Copie antes.")
        return
    print(f"✓ Test set: {n_test} imagens\n")

    all_results = {}

    for arm_name, paths in MAIN_ARMS.items():
        print(f"\n{'─'*80}")
        print(f"  {arm_name}")
        
        seed_data = {}
        for seed, pt in paths.items():
            if not pt.exists():
                print(f"    Seed {seed}: ✗ não encontrado")
                continue
            
            model = YOLO(str(pt))
            m = model.val(data=CITRA_YAML, split="test", device=0, verbose=False)
            
            # Métricas básicas
            p = float(m.box.p.mean()) if hasattr(m.box.p, 'mean') else float(m.box.p)
            r = float(m.box.r.mean()) if hasattr(m.box.r, 'mean') else float(m.box.r)
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
            
            result = {
                "mAP50": float(m.box.map50),
                "mAP50-95": float(m.box.map),
                "Precision": p,
                "Recall": r,
                "F1": f1,
            }
            
            # AP por tamanho (se disponível)
            try:
                # Ultralytics armazena em m.box.maps (lista por classe)
                # Para AP por tamanho, precisamos do COCO evaluator
                if hasattr(m, 'results_dict'):
                    rd = m.results_dict
                    for k, v in rd.items():
                        if 'small' in k.lower() or 'medium' in k.lower() or 'large' in k.lower():
                            result[k] = float(v)
            except:
                pass
            
            seed_data[seed] = result
            print(f"    Seed {seed}: mAP50={result['mAP50']:.4f} "
                  f"P={result['Precision']:.4f} R={result['Recall']:.4f} F1={result['F1']:.4f}")
        
        # Calcula estatísticas
        if seed_data:
            stats = {}
            for metric in ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]:
                vals = [d[metric] for d in seed_data.values()]
                stats[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "per_seed": {str(s): d[metric] for s, d in seed_data.items()}
                }
            
            all_results[arm_name] = stats
            
            print(f"\n    Resumo ({len(seed_data)} seeds):")
            print(f"    mAP50:     {stats['mAP50']['mean']:.4f} ± {stats['mAP50']['std']:.4f}")
            print(f"    mAP50-95:  {stats['mAP50-95']['mean']:.4f} ± {stats['mAP50-95']['std']:.4f}")
            print(f"    Precision: {stats['Precision']['mean']:.4f} ± {stats['Precision']['std']:.4f}")
            print(f"    Recall:    {stats['Recall']['mean']:.4f} ± {stats['Recall']['std']:.4f}")
            print(f"    F1:        {stats['F1']['mean']:.4f} ± {stats['F1']['std']:.4f}")

    # ══════════════════════════════════════════════════════════
    # PARTE 2: Steps por época
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f"  COMPARABILIDADE: STEPS POR ÉPOCA")
    print(f"{'='*80}")
    
    datasets = {
        "B2 (CITRA only)": 1348,
        "A' joint balanced": 35048,  # 17524 real + 17524 synth
        "A' sequential pretrain": 17524,
        "A' sequential finetune": 1348,
    }
    batch_size = 16
    
    for name, n_imgs in datasets.items():
        steps = (n_imgs + batch_size - 1) // batch_size
        print(f"  {name}: {n_imgs:,} imgs → {steps} steps/epoch")
    
    b2_steps_300 = ((1348 + 15) // 16) * 300
    joint_steps_300 = ((35048 + 15) // 16) * 300
    print(f"\n  B2 total updates (300ep): {b2_steps_300:,}")
    print(f"  A' joint total updates (300ep): {joint_steps_300:,}")
    print(f"  Ratio: A' joint vê {joint_steps_300/b2_steps_300:.1f}× mais updates")
    print(f"\n  Para igualar: B2 precisaria de {int(300 * joint_steps_300/b2_steps_300)} épocas")

    # ══════════════════════════════════════════════════════════
    # PARTE 3: Threshold do Ultralytics
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f"  THRESHOLD DE P/R/F1")
    print(f"{'='*80}")
    print(f"  Ultralytics calcula P, R em sweep de confidence thresholds.")
    print(f"  Os valores reportados por .val() são o melhor F1 no sweep.")
    print(f"  IoU threshold padrão: 0.5 (para mAP50) e 0.5:0.95 (para mAP50-95)")
    print(f"  Confidence threshold: selecionado automaticamente para maximizar F1")

    # ══════════════════════════════════════════════════════════
    # SALVA
    # ══════════════════════════════════════════════════════════
    def native(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, dict): return {k: native(v) for k, v in obj.items()}
        if isinstance(obj, list): return [native(v) for v in obj]
        return obj

    out = R / "metricas_detalhadas.json"
    with open(out, "w") as f:
        json.dump(native(all_results), f, indent=2)
    print(f"\n✓ Salvo: {out}")


if __name__ == "__main__":
    main()
