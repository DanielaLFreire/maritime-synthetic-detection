"""
treinar_b2_long_v2.py

Treina B2-long (CITRA-3D-Real oversampled 13×, sem sintéticas) para
seeds adicionais (123 e 2024), com 3 melhorias sobre o script original:

  1. NÃO recria o dataset se já existe (skip de ~5 min)
  2. SALVA o resultado da avaliação em JSON (era só print no v1)
  3. Suporta múltiplas seeds em uma execução

A' joint: 35.048 imgs/ep × 300 ep / 16 batch = 657.150 updates
B2-long:  17.524 imgs/ep × 300 ep / 16 batch = 328.575 updates
(B2-long iguala A' joint no número de imagens reais; o restante do
A' joint vem de sintéticas.)

Uso típico (3 execuções consecutivas em Colab, ou paralelo se possível):

  # Execução 1: terminar seed 42 que já existe (skip se best.pt presente)
  # Execução 2: seed 123
  # Execução 3: seed 2024

  Edite SEEDS abaixo ou execute uma seed por vez.
"""

from pathlib import Path
from ultralytics import YOLO
import shutil
import json
import time

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — edite conforme necessário
# ═══════════════════════════════════════════════════════════════════

# Seeds a rodar nesta execução. Para uma seed por vez, deixa só uma.
# Recomendação: rode uma de cada vez para garantir checkpoint
SEEDS = [123, 2024]            # seed 42 já está pronto

# Paths
DRIVE_BASE = Path("/content/drive/MyDrive/PROJETO_MARINHA")
LOCAL_CITRA = Path("/content/data/CITRA-3D-Real")
B2_LONG_DATA = Path("/content/data/b2_long")
RUNS_DIR = DRIVE_BASE / "Experimento_Dataset_Similar/runs/b2_long"
RESULTS_DIR = DRIVE_BASE / "Experimento_Dataset_Similar/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparams (idênticos ao B2 padrão e ao A' joint)
HYPERPARAMS = dict(
    epochs=300, patience=30,
    optimizer="AdamW", lr0=0.001, lrf=0.01,
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3, cos_lr=True,
    imgsz=640, batch=16, device=0,
    save=True, save_period=20,    # checkpoint a cada 20 epochs (proteção)
    exist_ok=True,
)


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Garantir dataset B2-long (CITRA 13× oversampled)
# ═══════════════════════════════════════════════════════════════════

def ensure_dataset():
    """Cria o dataset oversampled apenas se ainda não existir."""
    expected_train = B2_LONG_DATA / "train" / "images"
    expected_val   = B2_LONG_DATA / "val" / "images"

    if expected_train.exists() and expected_val.exists():
        n_train = len(list(expected_train.glob("*")))
        n_val   = len(list(expected_val.glob("*")))
        if n_train >= 17000 and n_val >= 4000:
            print(f"✓ Dataset já existe: {n_train:,} train, {n_val:,} val")
            return

    print("Criando dataset B2-long (CITRA 13× oversampled)...")
    if B2_LONG_DATA.exists():
        shutil.rmtree(B2_LONG_DATA)

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            dst = B2_LONG_DATA / split / sub
            dst.mkdir(parents=True, exist_ok=True)
            src = LOCAL_CITRA / split / sub
            assert src.exists(), f"Source missing: {src}"
            for rep in range(13):
                for f in src.glob("*"):
                    shutil.copy2(f, dst / f"r{rep:02d}_{f.name}")
            print(f"  {split}/{sub}: {len(list(dst.glob('*'))):,}")

    yaml_txt = f"""path: {B2_LONG_DATA}
train: train/images
val: val/images
nc: 1
names:
  - embarcacao
"""
    (B2_LONG_DATA / "data_single_class.yaml").write_text(yaml_txt)
    print("✓ Dataset B2-long pronto\n")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Treinar e avaliar uma seed
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate(seed: int) -> dict:
    """Treina B2-long para uma seed e re-avalia no test set."""

    seed_dir = RUNS_DIR / f"seed_{seed:04d}"
    best_pt = seed_dir / "weights" / "best.pt"

    # SKIP se já existe e best.pt está completo
    if best_pt.exists():
        print(f"✓ seed {seed}: best.pt já existe em {seed_pt}")
        print(f"  Pulando treino, indo direto para avaliação")
    else:
        print(f"\n{'='*60}")
        print(f"  TREINANDO seed {seed}")
        print(f"{'='*60}")
        t0 = time.time()

        model = YOLO("yolo11m.pt")
        model.train(
            data=str(B2_LONG_DATA / "data_single_class.yaml"),
            seed=seed,
            project=str(RUNS_DIR),
            name=f"seed_{seed:04d}",
            **HYPERPARAMS,
        )

        elapsed = time.time() - t0
        print(f"\n✓ Treino concluído em {elapsed/60:.1f} min")

    # AVALIA no test set CITRA-3D-Real
    print(f"\nAvaliando seed {seed} no test set...")
    test_yaml = LOCAL_CITRA / "data_single_class.yaml"
    assert test_yaml.exists(), f"Test yaml not found: {test_yaml}"
    assert best_pt.exists(),   f"best.pt not found: {best_pt}"

    eval_model = YOLO(str(best_pt))
    m = eval_model.val(
        data=str(test_yaml),
        split="test",
        device=0,
        verbose=False,
    )

    # Computa F1 manualmente
    P = float(m.box.mp)
    R = float(m.box.mr)
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    result = {
        "arm": "B2-long",
        "seed": seed,
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "P": P,
        "R": R,
        "F1": F1,
        "best_weights": str(best_pt),
        "test_yaml": str(test_yaml),
        "description": "COCO → CITRA-3D-Real oversampled 13× (volume control)",
    }

    print(f"\n  Seed {seed} RESULT:")
    print(f"    mAP50:    {result['mAP50']:.4f}")
    print(f"    mAP50-95: {result['mAP50_95']:.4f}")
    print(f"    P:        {result['P']:.4f}")
    print(f"    R:        {result['R']:.4f}")
    print(f"    F1:       {result['F1']:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_dataset()

    all_results = []

    # Sempre incluir seed 42 (re-avalia se best.pt existe)
    print("\n" + "="*60)
    print("  RE-AVALIAÇÃO seed 42 (já treinado)")
    print("="*60)
    try:
        r42 = train_and_evaluate(seed=42)
        all_results.append(r42)
    except Exception as e:
        print(f"⚠️ seed 42 falhou: {e}")

    # Rodar seeds adicionais
    for seed in SEEDS:
        if seed == 42:
            continue
        try:
            r = train_and_evaluate(seed=seed)
            all_results.append(r)
        except Exception as e:
            print(f"⚠️ seed {seed} falhou: {e}")
            continue

    # ═══════════════════════════════════════════════════════════════
    # Salvar consolidado em JSON
    # ═══════════════════════════════════════════════════════════════
    import statistics as stats

    out_json = RESULTS_DIR / "results_b2_long_n3.json"
    by_metric = {
        m: [r[m] for r in all_results]
        for m in ["mAP50", "mAP50_95", "P", "R", "F1"]
    }

    consolidated = {
        "arm": "B2-long",
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "per_seed": all_results,
        "summary": {
            m: {
                "mean": stats.mean(values),
                "std":  stats.stdev(values) if len(values) >= 2 else 0.0,
                "n": len(values),
                "values": values,
            }
            for m, values in by_metric.items()
        },
        "description": "COCO → CITRA-3D-Real oversampled 13× (volume control)",
    }

    out_json.write_text(json.dumps(consolidated, indent=2))
    print(f"\n\n✓ Consolidado salvo em {out_json}")

    # ═══════════════════════════════════════════════════════════════
    # Resumo final
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  RESUMO FINAL — B2-long (n=3)")
    print("="*60)
    for metric, agg in consolidated["summary"].items():
        if agg["n"] >= 2:
            print(f"  {metric}: {agg['mean']:.4f} ± {agg['std']:.4f}   "
                  f"(n={agg['n']})")
        else:
            print(f"  {metric}: {agg['mean']:.4f}   (n=1)")

    print("\n  Para comparação:")
    print(f"  B2 (padrão):    mAP50 = 0.8351 ± 0.0020")
    print(f"  A' joint:       mAP50 = 0.8451 ± 0.0033")
    print(f"  A' joint-rand:  mAP50 = 0.8457 ± 0.0058")
    print("="*60)
