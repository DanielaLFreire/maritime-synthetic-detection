"""
ablation_epocas_pretreino.py

Ablation diagnóstica: varia o número de épocas do pré-treino no
dataset_25k_v2 para investigar se o resultado inferior do braço A
(mAP50 = 0.7936 vs B2 = 0.8351) é causado por catastrophic forgetting
(excesso de treino no InaTechShips) ou por incompatibilidade
fundamental dos dados.

DESENHO

  Para cada N_EPOCHS em [10, 20, 50]:
    1. Pré-treino: COCO → dataset_25k_v2 (N_EPOCHS épocas, seed 42)
    2. Fine-tuning: best.pt → CITRA-3D-Real (300 épocas, patience 30)
    3. Avaliação no test set

  Comparação com resultados existentes:
    - B2 baseline (sem pré-treino): mAP50 = 0.8351
    - A-100ep (100 épocas de pré-treino, seed 42): mAP50 = 0.8006

  Se mAP50 melhora com menos épocas → catastrophic forgetting confirmado.
  Se mAP50 permanece baixo mesmo com 10 épocas → incompatibilidade.

PROTOCOLO

  Idêntico ao treinar_braco_a.py, exceto:
    - Seed fixa: 42 (diagnóstico, não precisa de variância entre seeds)
    - Pré-treino: patience=10 (menor, para não truncar runs curtas)
    - Fine-tuning: 300 épocas, patience 30 (inalterado)

USO

  python ablation_epocas_pretreino.py
  python ablation_epocas_pretreino.py --epochs 10 20     # só dois pontos
  python ablation_epocas_pretreino.py --dry-run           # só mostra o plano

SAÍDA

  /content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs/ablation_epochs/
  ├── ep_010/
  │   ├── pretrain/
  │   └── finetune/
  ├── ep_020/
  ├── ep_050/
  └── ablation_epochs_summary.json

TEMPO ESTIMADO

  ~2-3h no A100 (3 configurações × ~40-60 min cada)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

DRIVE_BASE = Path("/content/drive/MyDrive")
DATASET_25K_V2 = DRIVE_BASE / "InaTechShips" / "dataset_25k_v2"
CITRA3D_DRIVE = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "CITRA-3D-Real"
RUNS_DIR = DRIVE_BASE / "PROJETO_MARINHA" / "Experimento_Dataset_Similar" / "runs" / "ablation_epochs"
BRACO_A_DIR = DRIVE_BASE / "PROJETO_MARINHA" / "Experimento_Dataset_Similar" / "runs" / "braco_a"

LOCAL_DATA = Path("/content/data")
LOCAL_DATASET_25K = LOCAL_DATA / "dataset_25k_v2"
LOCAL_CITRA3D = LOCAL_DATA / "CITRA-3D-Real"

MODEL_NAME = "yolo11m.pt"
SEED = 42
EPOCH_VARIANTS = [10, 20, 50]

# Hyperparams (idênticos a B2 / braço A)
COMMON_HYPERPARAMS = dict(
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    imgsz=640,
    batch=16,
    workers=4,
    verbose=True,
    exist_ok=True,
)

PRETRAIN_PATIENCE = 10   # menor que no braço A (20), porque runs são curtas
FINETUNE_EPOCHS = 300
FINETUNE_PATIENCE = 30


# ═══════════════════════════════════════════════════════════════════
# Helpers (reutilizados do treinar_braco_a.py)
# ═══════════════════════════════════════════════════════════════════

def check_prerequisites() -> bool:
    ok = True

    if not DRIVE_BASE.exists():
        print("✗ Drive não montado")
        ok = False
    else:
        print("✓ Drive montado")

    yaml_25k = DATASET_25K_V2 / "data_single_class.yaml"
    if not yaml_25k.exists():
        print(f"✗ dataset_25k_v2 não encontrado")
        ok = False
    else:
        print(f"✓ dataset_25k_v2")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ GPU indisponível")
            ok = False
    except ImportError:
        print("✗ PyTorch não instalado")
        ok = False

    try:
        import ultralytics
        print(f"✓ ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("✗ ultralytics não instalado")
        ok = False

    return ok


def prepare_local_data(dataset_name: str, drive_src: Path, local_dst: Path) -> Path:
    local_yaml = local_dst / "data_single_class.yaml"

    if local_dst.exists() and local_yaml.exists():
        train_imgs = local_dst / "train" / "images"
        if train_imgs.exists() and any(train_imgs.iterdir()):
            n = sum(1 for _ in train_imgs.iterdir())
            print(f"  ✓ {dataset_name} já preparado ({n} imgs train)")
            return local_yaml

    print(f"  Copiando {dataset_name} do Drive...")
    t0 = time.time()

    if local_dst.exists():
        shutil.rmtree(local_dst)

    for split in ("train", "val", "test"):
        for subfolder in ("images", "labels_single_class"):
            src_dir = drive_src / split / subfolder
            if not src_dir.exists():
                continue
            dst_name = "labels" if subfolder == "labels_single_class" else subfolder
            dst_dir = local_dst / split / dst_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            n = 0
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)
                    n += 1
            print(f"    {split}/{dst_name}: {n:,}")

    yaml_content = f"""path: {local_dst}
train: train/images
val: val/images
test: test/images
nc: 1
names:
  - embarcacao
"""
    local_yaml.write_text(yaml_content)
    print(f"  ✓ {dataset_name} preparado em {time.time()-t0:.0f}s")
    return local_yaml


def run_one_variant(n_epochs: int, yaml_25k: Path, yaml_citra: Path,
                    run_dir: Path) -> dict | None:
    """Roda pré-treino (n_epochs) + fine-tuning (300 épocas) para uma variante."""
    from ultralytics import YOLO

    variant_dir = run_dir / f"ep_{n_epochs:03d}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    # ── Fase 1: pré-treino ──
    print(f"\n{'='*72}")
    print(f"  FASE 1 — PRÉ-TREINO {n_epochs} épocas (seed={SEED})")
    print(f"{'='*72}")

    pretrain_best = variant_dir / "pretrain" / "weights" / "best.pt"

    if pretrain_best.exists():
        print(f"  ✓ Pré-treino já existe: {pretrain_best}")
    else:
        model = YOLO(MODEL_NAME)
        t0 = time.time()

        try:
            model.train(
                data=str(yaml_25k),
                epochs=n_epochs,
                patience=PRETRAIN_PATIENCE,
                seed=SEED,
                project=str(variant_dir),
                name="pretrain",
                device=0,
                save=True,
                plots=True,
                **COMMON_HYPERPARAMS,
            )
        except Exception as exc:
            print(f"\n✗ Erro no pré-treino {n_epochs}ep: {exc}")
            return None

        elapsed = time.time() - t0
        print(f"\n✓ Pré-treino {n_epochs}ep concluído em {elapsed/60:.1f} min")

        if not pretrain_best.exists():
            print(f"✗ best.pt não encontrado")
            return None

    # ── Fase 2: fine-tuning ──
    print(f"\n{'='*72}")
    print(f"  FASE 2 — FINE-TUNING CITRA-3D-Real (pré-treino={n_epochs}ep)")
    print(f"{'='*72}")

    finetune_best = variant_dir / "finetune" / "weights" / "best.pt"

    if finetune_best.exists():
        print(f"  ✓ Fine-tuning já existe: {finetune_best}")
    else:
        model = YOLO(str(pretrain_best))
        t0 = time.time()

        try:
            model.train(
                data=str(yaml_citra),
                epochs=FINETUNE_EPOCHS,
                patience=FINETUNE_PATIENCE,
                seed=SEED,
                project=str(variant_dir),
                name="finetune",
                device=0,
                save=True,
                plots=True,
                **COMMON_HYPERPARAMS,
            )
        except Exception as exc:
            print(f"\n✗ Erro no fine-tuning ({n_epochs}ep pretrain): {exc}")
            return None

        elapsed = time.time() - t0
        print(f"\n✓ Fine-tuning concluído em {elapsed/60:.1f} min")

        if not finetune_best.exists():
            print(f"✗ best.pt não encontrado")
            return None

    # ── Avaliação no test set ──
    print(f"\n>> Avaliando no test set...")
    model_eval = YOLO(str(finetune_best))
    metrics = model_eval.val(data=str(yaml_citra), split="test", device=0, verbose=False)

    result = {
        "pretrain_epochs": n_epochs,
        "seed": SEED,
        "pretrain_best": str(pretrain_best),
        "finetune_best": str(finetune_best),
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "fitness": 0.1 * float(metrics.box.map50) + 0.9 * float(metrics.box.map),
        },
    }

    m = result["metrics"]
    print(f"\n  Resultado ({n_epochs}ep pré-treino → fine-tuning):")
    print(f"    mAP50:    {m['mAP50']:.4f}")
    print(f"    mAP50-95: {m['mAP50-95']:.4f}")
    print(f"    fitness:  {m['fitness']:.4f}")

    # Salva resultado individual
    result_file = variant_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: épocas de pré-treino")
    parser.add_argument("--epochs", type=int, nargs="+", default=EPOCH_VARIANTS,
                        help="Variantes de épocas (default: 10 20 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Só mostra o plano, não treina")
    args = parser.parse_args()

    print("=" * 72)
    print("  ABLATION — Épocas de pré-treino no dataset_25k_v2")
    print("=" * 72)
    print(f"  Variantes:      {args.epochs}")
    print(f"  Seed:           {SEED}")
    print(f"  Fine-tuning:    {FINETUNE_EPOCHS} épocas, patience {FINETUNE_PATIENCE}")
    print(f"  Runs dir:       {RUNS_DIR}")
    print(f"  Dry-run:        {args.dry_run}")
    print("=" * 72)

    # ── Referências existentes ──
    print(f"\n>> Referências existentes:")
    print(f"  B2 baseline (COCO → CITRA-3D):          mAP50 = 0.8351")
    print(f"  B1 baseline (Random → CITRA-3D):        mAP50 = 0.8008")
    print(f"  A-100ep (COCO → 100ep → CITRA-3D, s42): mAP50 = 0.8006")

    # Tenta ler resultado da seed 42 do braço A como referência
    braco_a_42 = BRACO_A_DIR / "seed_0042" / "finetune_result.json"
    if braco_a_42.exists():
        with open(braco_a_42) as f:
            a42 = json.load(f)
        ref_100ep = a42["metrics"]["mAP50"]
        print(f"  (lido de {braco_a_42.name}: mAP50={ref_100ep:.4f})")
    else:
        ref_100ep = 0.8006
        print(f"  (usando valor fixo)")

    if args.dry_run:
        print(f"\n>> Plano de execução:")
        for n_ep in args.epochs:
            est = n_ep * 0.6 + 30  # estimativa grosseira: 0.6 min/época pretrain + 30 min finetune
            print(f"  {n_ep:>3} épocas pré-treino → fine-tuning 300ep → ~{est:.0f} min")
        print(f"\n  Total estimado: ~{sum(n * 0.6 + 30 for n in args.epochs)/60:.1f}h")
        print(f"\n>> --dry-run: nada executado")
        return

    # ── Pré-checagens ──
    print(f"\n>> Pré-checagens")
    if not check_prerequisites():
        sys.exit(1)

    # ── Preparar dados locais ──
    print(f"\n>> Preparando dados locais...")
    yaml_25k = prepare_local_data("dataset_25k_v2", DATASET_25K_V2, LOCAL_DATASET_25K)

    citra_src = CITRA3D_DRIVE
    if not citra_src.exists():
        citra_src = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "CITRA-3D-Real"
    if citra_src.exists():
        yaml_citra = prepare_local_data("CITRA-3D-Real", citra_src, LOCAL_CITRA3D)
    else:
        yaml_citra = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "configs" / "citra3d_single_class.yaml"
        if not yaml_citra.exists():
            print(f"✗ CITRA-3D-Real não encontrado")
            sys.exit(1)
        if not (LOCAL_CITRA3D / "train" / "images").exists():
            print(f"✗ Dados locais do CITRA-3D não existem. Rode preparar_dados_locais.py primeiro.")
            sys.exit(1)

    # ── Loop de variantes ──
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    t0_global = time.time()

    for n_ep in args.epochs:
        result = run_one_variant(n_ep, yaml_25k, yaml_citra, RUNS_DIR)
        if result is not None:
            all_results.append(result)

    elapsed_total = time.time() - t0_global

    # ── Relatório consolidado ──
    if not all_results:
        print(f"\n✗ Nenhuma variante completou.")
        return

    print(f"\n{'='*72}")
    print(f"  RESULTADOS DA ABLATION DE ÉPOCAS")
    print(f"{'='*72}")
    print(f"  Variantes concluídas: {len(all_results)}/{len(args.epochs)}")
    print(f"  Tempo total: {elapsed_total/3600:.1f}h")

    # Tabela de resultados
    print(f"\n  {'épocas':>8}{'mAP50':>10}{'mAP50-95':>12}{'fitness':>10}{'Δ vs B2':>10}")
    print(f"  {'-'*50}")

    # Referências
    print(f"  {'B2 (ref)':>8}{'0.8351':>10}{'0.5055':>12}{'0.5385':>10}{'—':>10}")
    print(f"  {'B1':>8}{'0.8008':>10}{'0.4742':>12}{'0.4869':>10}{'-0.0343':>10}")

    for r in sorted(all_results, key=lambda x: x["pretrain_epochs"]):
        ep = r["pretrain_epochs"]
        m = r["metrics"]
        delta = m["mAP50"] - 0.8351
        print(f"  {ep:>8}{m['mAP50']:>10.4f}{m['mAP50-95']:>12.4f}"
              f"{m['fitness']:>10.4f}{delta:>+10.4f}")

    # Adiciona A-100ep como referência
    print(f"  {'100':>8}{ref_100ep:>10.4f}{'0.4680':>12}{'0.5013':>10}"
          f"{ref_100ep - 0.8351:>+10.4f}")

    # Diagnóstico automático
    print(f"\n>> Diagnóstico:")
    best_variant = max(all_results, key=lambda r: r["metrics"]["mAP50"])
    best_ep = best_variant["pretrain_epochs"]
    best_map50 = best_variant["metrics"]["mAP50"]

    if best_map50 > 0.8351:
        print(f"  ✓ ACHADO: pré-treino com {best_ep} épocas SUPERA B2!")
        print(f"    mAP50 = {best_map50:.4f} vs B2 = 0.8351 (Δ = {best_map50-0.8351:+.4f})")
        print(f"    → Pré-treino no InaTechShips é benéfico com dose controlada.")
    elif best_map50 > ref_100ep + 0.005:
        print(f"  ⚠ ACHADO: {best_ep} épocas é melhor que 100 épocas,")
        print(f"    mas ainda não supera B2.")
        print(f"    mAP50: {best_map50:.4f} (vs 100ep={ref_100ep:.4f}, B2=0.8351)")
        print(f"    → Catastrophic forgetting confirmado parcialmente.")
        print(f"    → O pré-treino intermediário prejudica mesmo em dose menor,")
        print(f"      mas menos que em dose alta.")
    elif best_map50 > 0.8008:
        print(f"  ⚠ ACHADO: {best_ep} épocas fica entre B1 e B2.")
        print(f"    mAP50: {best_map50:.4f} (B1=0.8008, B2=0.8351)")
        print(f"    → O pré-treino intermediário não é catastrófico com poucos epochs,")
        print(f"      mas também não ajuda vs COCO puro.")
    else:
        print(f"  ✗ ACHADO: mesmo com {best_ep} épocas, resultado ≤ B1.")
        print(f"    mAP50: {best_map50:.4f} (B1=0.8008)")
        print(f"    → Incompatibilidade fundamental: o dataset InaTechShips")
        print(f"      prejudica transfer learning para CITRA-3D independente")
        print(f"      da duração do pré-treino.")

    # Salva JSON
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "ablation_pretrain_epochs",
        "seed": SEED,
        "variants": all_results,
        "reference_b2_mAP50": 0.8351,
        "reference_b1_mAP50": 0.8008,
        "reference_a100ep_mAP50": ref_100ep,
        "best_variant_epochs": best_ep,
        "best_variant_mAP50": best_map50,
        "total_time_seconds": elapsed_total,
        "total_time_human": f"{elapsed_total/3600:.1f}h",
    }

    summary_file = RUNS_DIR / "ablation_epochs_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Relatório: {summary_file}")

    print(f"\n{'='*72}")
    print(f"  ABLATION FINALIZADA — {elapsed_total/3600:.1f}h")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
