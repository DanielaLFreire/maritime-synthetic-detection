"""
treinar_braco_a.py

Treina o braço A do experimento principal: pré-treino no dataset_25k_v2
(curado por similaridade CLIP) seguido de fine-tuning no CITRA-3D-Real.

DESENHO EXPERIMENTAL

  COCO (yolo11m.pt) → dataset_25k_v2 (pré-treino) → CITRA-3D-Real (fine-tuning)

  Comparação direta com:
    B2: COCO → CITRA-3D-Real (sem etapa intermediária)
    B1: Random init → CITRA-3D-Real (sem COCO nem intermediário)
    B (futuro): COCO → random_pool_v2 → CITRA-3D-Real (curado vs aleatório)

  Variável isolada: presença de pré-treino em dados marítimos (dataset_25k_v2)
  antes do fine-tuning no domínio operacional.

PROTOCOLO

  Fase 1 — Pré-treino no dataset_25k_v2:
    - Modelo: YOLOv11m, pesos iniciais: yolo11m.pt (COCO)
    - Dataset: dataset_25k_v2 single-class ("embarcacao"), 16.677 train / 5.558 val
    - Épocas: 100, patience: 20
    - Hyperparams: AdamW, lr0=0.001, lrf=0.01, momentum=0.937, wd=0.0005,
      warmup_epochs=3, cos_lr=True, imgsz=640, batch=16
    - Saída: best.pt do pré-treino

  Fase 2 — Fine-tuning no CITRA-3D-Real:
    - Modelo: YOLOv11m, pesos iniciais: best.pt da Fase 1
    - Dataset: CITRA-3D-Real single-class, 1.348 train / 332 val / 401 test
    - Épocas: 300, patience: 30
    - Hyperparams: idênticos (AdamW, lr0=0.001, lrf=0.01, etc.)
    - Saída: best.pt do fine-tuning + métricas no test set

  Repetição: 3 seeds (42, 123, 2024) para estimativa de variância.

PRÉ-REQUISITOS

  - Google Colab com GPU A100 (recomendado)
  - Drive montado em /content/drive
  - dataset_25k_v2 em /content/drive/MyDrive/InaTechShips/dataset_25k_v2/
    com data_single_class.yaml
  - CITRA-3D-Real preparado pelo preparar_dados_locais.py (ou disponível no Drive)
  - ultralytics >= 8.4

USO

  python treinar_braco_a.py                    # roda tudo (3 seeds × 2 fases)
  python treinar_braco_a.py --seeds 42         # só seed 42
  python treinar_braco_a.py --phase pretrain   # só Fase 1 (pré-treino)
  python treinar_braco_a.py --phase finetune   # só Fase 2 (fine-tuning, requer best.pt da Fase 1)

SAÍDA

  /content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs/braco_a/
  ├── seed_042/
  │   ├── pretrain/     (run do pré-treino no dataset_25k_v2)
  │   └── finetune/     (run do fine-tuning no CITRA-3D-Real)
  ├── seed_123/
  ├── seed_2024/
  └── braco_a_summary.json   (resumo + comparação com baselines)

TEMPO ESTIMADO

  ~6h total no A100 (3 seeds × ~1h pré-treino + ~30min fine-tuning)
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

# Paths no Drive
DRIVE_BASE = Path("/content/drive/MyDrive")
DATASET_25K_V2 = DRIVE_BASE / "InaTechShips" / "dataset_25k_v2"
CITRA3D_DRIVE = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "CITRA-3D-Real"
RUNS_DIR = DRIVE_BASE / "PROJETO_MARINHA" / "Experimento_Dataset_Similar" / "runs" / "braco_a"
BASELINES_DIR = DRIVE_BASE / "PROJETO_MARINHA" / "Experimento_Dataset_Similar" / "runs" / "baselines"

# Paths locais (disco rápido do Colab)
LOCAL_DATA = Path("/content/data")
LOCAL_DATASET_25K = LOCAL_DATA / "dataset_25k_v2"
LOCAL_CITRA3D = LOCAL_DATA / "CITRA-3D-Real"

# Modelo
MODEL_NAME = "yolo11m.pt"

# Seeds
ALL_SEEDS = [42, 123, 2024]

# Hyperparams (idênticos a B2, validados pelo HPO)
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

# Fase 1 — Pré-treino
PRETRAIN_EPOCHS = 100
PRETRAIN_PATIENCE = 20

# Fase 2 — Fine-tuning
FINETUNE_EPOCHS = 300
FINETUNE_PATIENCE = 30


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def check_prerequisites() -> bool:
    """Verifica que tudo está pronto antes de começar."""
    ok = True

    # Drive montado?
    if not DRIVE_BASE.exists():
        print("✗ Drive não montado. Rode: from google.colab import drive; drive.mount('/content/drive')")
        ok = False
    else:
        print("✓ Drive montado")

    # dataset_25k_v2 existe?
    yaml_25k = DATASET_25K_V2 / "data_single_class.yaml"
    if not yaml_25k.exists():
        print(f"✗ data_single_class.yaml não encontrado em {DATASET_25K_V2}")
        ok = False
    else:
        print(f"✓ dataset_25k_v2: {yaml_25k}")

    # CITRA-3D-Real no Drive?
    citra_yaml = CITRA3D_DRIVE / "data_single_class.yaml"
    # Tenta caminho alternativo
    citra_yaml_alt = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "configs" / "citra3d_single_class.yaml"
    if citra_yaml.exists():
        print(f"✓ CITRA-3D-Real yaml: {citra_yaml}")
    elif citra_yaml_alt.exists():
        print(f"✓ CITRA-3D-Real yaml: {citra_yaml_alt}")
    else:
        print(f"⚠ CITRA-3D yaml não encontrado nos caminhos esperados")
        print(f"  Tentei: {citra_yaml}")
        print(f"  Tentei: {citra_yaml_alt}")
        print(f"  O script tentará usar preparar_dados_locais.py")

    # GPU?
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU: {gpu_name}")
        else:
            print("✗ GPU não disponível")
            ok = False
    except ImportError:
        print("✗ PyTorch não instalado")
        ok = False

    # Ultralytics?
    try:
        import ultralytics
        print(f"✓ ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("✗ ultralytics não instalado")
        ok = False

    return ok


def prepare_local_data(dataset_name: str, drive_src: Path, local_dst: Path,
                       yaml_name: str = "data_single_class.yaml") -> Path:
    """
    Copia dataset do Drive para disco local do Colab (mais rápido para treino).

    Retorna o path do data.yaml local.
    """
    local_yaml = local_dst / yaml_name

    if local_dst.exists() and local_yaml.exists():
        # Verifica se já tem imagens (sessão pode ter sido resetada)
        train_imgs = local_dst / "train" / "images"
        if train_imgs.exists() and any(train_imgs.iterdir()):
            n = sum(1 for _ in train_imgs.iterdir())
            print(f"  ✓ {dataset_name} já preparado localmente ({n} imgs train)")
            return local_yaml

    print(f"  Copiando {dataset_name} do Drive para {local_dst}...")
    t0 = time.time()

    # Limpa destino
    if local_dst.exists():
        shutil.rmtree(local_dst)

    # Copia estrutura completa
    for split in ("train", "val", "test"):
        for subfolder in ("images", "labels_single_class"):
            src_dir = drive_src / split / subfolder
            if not src_dir.exists():
                continue

            # Para labels_single_class, copia como "labels" no destino
            # (Ultralytics espera /images/ e /labels/ no mesmo nível)
            dst_name = "labels" if subfolder == "labels_single_class" else subfolder
            dst_dir = local_dst / split / dst_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            n_copied = 0
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)
                    n_copied += 1

            print(f"    {split}/{dst_name}: {n_copied:,} arquivos")

    # Gera data.yaml local
    yaml_content = f"""# Auto-gerado por treinar_braco_a.py
path: {local_dst}
train: train/images
val: val/images
test: test/images
nc: 1
names:
  - embarcacao
"""
    local_yaml.write_text(yaml_content)

    elapsed = time.time() - t0
    print(f"  ✓ {dataset_name} preparado em {elapsed:.0f}s")

    return local_yaml


def run_pretrain(seed: int, data_yaml: Path, run_dir: Path) -> Path | None:
    """
    Fase 1: pré-treino no dataset_25k_v2.

    Retorna o path do best.pt, ou None se falhar.
    """
    from ultralytics import YOLO

    print(f"\n{'='*72}")
    print(f"  FASE 1 — PRÉ-TREINO no dataset_25k_v2 (seed={seed})")
    print(f"{'='*72}")

    project = str(run_dir)
    name = "pretrain"

    model = YOLO(MODEL_NAME)
    t0 = time.time()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=PRETRAIN_EPOCHS,
            patience=PRETRAIN_PATIENCE,
            seed=seed,
            project=project,
            name=name,
            device=0,
            save=True,
            plots=True,
            **COMMON_HYPERPARAMS,
        )
    except Exception as exc:
        print(f"\n✗ Erro no pré-treino seed {seed}: {exc}")
        return None

    elapsed = time.time() - t0
    best_pt = Path(project) / name / "weights" / "best.pt"

    if best_pt.exists():
        print(f"\n✓ Pré-treino seed {seed} concluído em {elapsed/60:.1f} min")
        print(f"  best.pt: {best_pt}")
        return best_pt
    else:
        print(f"\n✗ best.pt não encontrado após pré-treino seed {seed}")
        return None


def run_finetune(seed: int, pretrained_weights: Path, data_yaml: Path,
                 run_dir: Path) -> dict | None:
    """
    Fase 2: fine-tuning no CITRA-3D-Real a partir dos pesos pré-treinados.

    Retorna dict com métricas do test set, ou None se falhar.
    """
    from ultralytics import YOLO

    print(f"\n{'='*72}")
    print(f"  FASE 2 — FINE-TUNING no CITRA-3D-Real (seed={seed})")
    print(f"{'='*72}")
    print(f"  Pesos iniciais: {pretrained_weights}")

    project = str(run_dir)
    name = "finetune"

    model = YOLO(str(pretrained_weights))
    t0 = time.time()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=FINETUNE_EPOCHS,
            patience=FINETUNE_PATIENCE,
            seed=seed,
            project=project,
            name=name,
            device=0,
            save=True,
            plots=True,
            **COMMON_HYPERPARAMS,
        )
    except Exception as exc:
        print(f"\n✗ Erro no fine-tuning seed {seed}: {exc}")
        return None

    elapsed = time.time() - t0

    # Avalia no test set
    print(f"\n>> Avaliando no test set...")
    best_pt = Path(project) / name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"✗ best.pt não encontrado")
        return None

    model_eval = YOLO(str(best_pt))
    metrics = model_eval.val(data=str(data_yaml), split="test", device=0, verbose=False)

    result = {
        "seed": seed,
        "pretrained_weights": str(pretrained_weights),
        "best_weights": str(best_pt),
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{elapsed/60:.1f} min",
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
    }

    fitness = 0.1 * result["metrics"]["mAP50"] + 0.9 * result["metrics"]["mAP50-95"]
    result["metrics"]["fitness"] = fitness

    print(f"\n✓ Fine-tuning seed {seed} concluído em {elapsed/60:.1f} min")
    print(f"  mAP50:    {result['metrics']['mAP50']:.4f}")
    print(f"  mAP50-95: {result['metrics']['mAP50-95']:.4f}")
    print(f"  fitness:  {fitness:.4f}")

    return result


def load_baselines() -> dict | None:
    """Carrega resultados dos baselines B1/B2 para comparação."""
    summary_file = BASELINES_DIR / "all_runs_summary.json"
    if not summary_file.exists():
        print(f"  ⚠ Baselines summary não encontrado: {summary_file}")
        return None

    with open(summary_file) as f:
        return json.load(f)


def generate_summary(all_results: list[dict], baselines: dict | None) -> dict:
    """Gera relatório consolidado do braço A com comparação aos baselines."""
    import statistics

    # Médias do braço A
    mAP50s = [r["metrics"]["mAP50"] for r in all_results]
    mAP50_95s = [r["metrics"]["mAP50-95"] for r in all_results]
    fitnesses = [r["metrics"]["fitness"] for r in all_results]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "braco_a",
        "description": "COCO → dataset_25k_v2 (pré-treino) → CITRA-3D-Real (fine-tuning)",
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "protocol": {
            "pretrain": {
                "dataset": "dataset_25k_v2 (single-class, 27.796 IDs únicos)",
                "initial_weights": MODEL_NAME,
                "epochs": PRETRAIN_EPOCHS,
                "patience": PRETRAIN_PATIENCE,
            },
            "finetune": {
                "dataset": "CITRA-3D-Real (single-class, 2.081 imagens)",
                "epochs": FINETUNE_EPOCHS,
                "patience": FINETUNE_PATIENCE,
            },
            "hyperparams": {k: v for k, v in COMMON_HYPERPARAMS.items()
                          if k not in ("verbose", "exist_ok", "workers")},
        },
        "results_per_seed": all_results,
        "aggregate": {
            "mAP50_mean": statistics.mean(mAP50s),
            "mAP50_std": statistics.stdev(mAP50s) if len(mAP50s) > 1 else 0,
            "mAP50-95_mean": statistics.mean(mAP50_95s),
            "mAP50-95_std": statistics.stdev(mAP50_95s) if len(mAP50_95s) > 1 else 0,
            "fitness_mean": statistics.mean(fitnesses),
            "fitness_std": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0,
        },
        "total_time_seconds": sum(r["elapsed_seconds"] for r in all_results),
        "total_time_human": f"{sum(r['elapsed_seconds'] for r in all_results)/3600:.1f}h",
    }

    # Comparação com baselines
    if baselines:
        comparison = {}

        # Extrai métricas dos baselines
        for baseline_name in ("B1", "B2"):
            bl_key = baseline_name.lower()
            if bl_key in baselines:
                bl = baselines[bl_key]
                bl_map50 = bl.get("mAP50_mean", bl.get("test_mAP50_mean"))
                bl_map50_95 = bl.get("mAP50-95_mean", bl.get("test_mAP50-95_mean"))
                bl_map50_std = bl.get("mAP50_std", bl.get("test_mAP50_std", 0))

                if bl_map50 is not None:
                    delta_map50 = summary["aggregate"]["mAP50_mean"] - bl_map50
                    delta_map50_95 = summary["aggregate"]["mAP50-95_mean"] - bl_map50_95 if bl_map50_95 else None

                    comparison[baseline_name] = {
                        "baseline_mAP50": bl_map50,
                        "baseline_mAP50_std": bl_map50_std,
                        "baseline_mAP50-95": bl_map50_95,
                        "braco_a_mAP50": summary["aggregate"]["mAP50_mean"],
                        "braco_a_mAP50-95": summary["aggregate"]["mAP50-95_mean"],
                        "delta_mAP50": delta_map50,
                        "delta_mAP50-95": delta_map50_95,
                        "delta_mAP50_pct": f"{delta_map50/bl_map50*100:+.2f}%" if bl_map50 else None,
                    }

        summary["comparison_vs_baselines"] = comparison

    return summary


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar braço A do experimento")
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS,
                        help="Seeds a executar (default: 42 123 2024)")
    parser.add_argument("--phase", choices=["all", "pretrain", "finetune"],
                        default="all", help="Fase a executar")
    args = parser.parse_args()

    print("=" * 72)
    print("  BRAÇO A — Pré-treino dataset_25k_v2 + Fine-tuning CITRA-3D-Real")
    print("=" * 72)
    print(f"  Seeds:          {args.seeds}")
    print(f"  Fase:           {args.phase}")
    print(f"  Modelo:         {MODEL_NAME}")
    print(f"  Pré-treino:     {PRETRAIN_EPOCHS} épocas, patience {PRETRAIN_PATIENCE}")
    print(f"  Fine-tuning:    {FINETUNE_EPOCHS} épocas, patience {FINETUNE_PATIENCE}")
    print(f"  Runs dir:       {RUNS_DIR}")
    print("=" * 72)

    # ---- Pré-checagens ----
    print("\n>> Pré-checagens")
    if not check_prerequisites():
        print("\n✗ Pré-checagens falharam. Corrija os problemas acima.")
        sys.exit(1)

    # ---- Preparar dados locais ----
    print("\n>> Preparando dados locais...")

    # dataset_25k_v2
    if args.phase in ("all", "pretrain"):
        yaml_25k = prepare_local_data(
            "dataset_25k_v2",
            DATASET_25K_V2,
            LOCAL_DATASET_25K,
        )
    else:
        yaml_25k = None

    # CITRA-3D-Real
    if args.phase in ("all", "finetune"):
        # Tenta encontrar o CITRA-3D-Real no Drive
        citra_src = CITRA3D_DRIVE
        if not citra_src.exists():
            # Tenta caminho alternativo
            citra_src = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "CITRA-3D-Real"

        if citra_src.exists():
            yaml_citra = prepare_local_data(
                "CITRA-3D-Real",
                citra_src,
                LOCAL_CITRA3D,
            )
        else:
            # Tenta usar yaml existente de configs
            yaml_citra = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "configs" / "citra3d_single_class.yaml"
            if yaml_citra.exists():
                print(f"  ✓ Usando yaml existente: {yaml_citra}")
                # Verifica se dados locais existem
                if not (LOCAL_CITRA3D / "train" / "images").exists():
                    print(f"  ⚠ Dados locais não encontrados em {LOCAL_CITRA3D}")
                    print(f"    Rode preparar_dados_locais.py --only citra primeiro!")
                    sys.exit(1)
            else:
                print(f"✗ CITRA-3D-Real não encontrado")
                sys.exit(1)
    else:
        yaml_citra = None

    # ---- Estimativa de tempo ----
    n_seeds = len(args.seeds)
    if args.phase == "all":
        est_min = n_seeds * (60 + 30)  # ~60min pretrain + 30min finetune por seed
    elif args.phase == "pretrain":
        est_min = n_seeds * 60
    else:
        est_min = n_seeds * 30

    print(f"\n>> Estimativa: {n_seeds} seed(s) × ~{est_min//n_seeds} min/seed = ~{est_min/60:.1f}h total")

    # ---- Loop principal ----
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    t0_global = time.time()

    for seed in args.seeds:
        seed_dir = RUNS_DIR / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        pretrain_best = None

        # Fase 1: pré-treino
        if args.phase in ("all", "pretrain"):
            pretrain_best = run_pretrain(seed, yaml_25k, seed_dir)

            if pretrain_best is None:
                print(f"\n✗ Pré-treino seed {seed} falhou. Pulando para próxima seed.")
                continue

            # Copia best.pt para o Drive (segurança)
            drive_best = seed_dir / "pretrain_best.pt"
            shutil.copy2(pretrain_best, drive_best)
            print(f"  Backup: {drive_best}")

        # Fase 2: fine-tuning
        if args.phase in ("all", "finetune"):
            # Se só rodando finetune, precisa encontrar o best.pt do pretrain
            if pretrain_best is None:
                pretrain_best = seed_dir / "pretrain" / "weights" / "best.pt"
                if not pretrain_best.exists():
                    pretrain_best = seed_dir / "pretrain_best.pt"
                if not pretrain_best.exists():
                    print(f"\n✗ best.pt do pré-treino não encontrado para seed {seed}")
                    print(f"  Rode --phase pretrain primeiro.")
                    continue

            result = run_finetune(seed, pretrain_best, yaml_citra, seed_dir)

            if result is not None:
                all_results.append(result)

                # Salva resultado individual
                result_file = seed_dir / "finetune_result.json"
                result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- Relatório consolidado ----
    elapsed_total = time.time() - t0_global

    if all_results:
        print(f"\n{'='*72}")
        print(f"  RESULTADOS DO BRAÇO A")
        print(f"{'='*72}")
        print(f"  Seeds concluídas: {len(all_results)}/{n_seeds}")
        print(f"  Tempo total:      {elapsed_total/3600:.1f}h")

        # Carrega baselines para comparação
        baselines = load_baselines()
        summary = generate_summary(all_results, baselines)

        # Mostra métricas
        agg = summary["aggregate"]
        print(f"\n  Braço A (média ± DP):")
        print(f"    mAP50:    {agg['mAP50_mean']:.4f} ± {agg['mAP50_std']:.4f}")
        print(f"    mAP50-95: {agg['mAP50-95_mean']:.4f} ± {agg['mAP50-95_std']:.4f}")
        print(f"    fitness:  {agg['fitness_mean']:.4f} ± {agg['fitness_std']:.4f}")

        # Comparação com baselines
        if "comparison_vs_baselines" in summary:
            print(f"\n  Comparação com baselines:")
            print(f"  {'braço':<12}{'mAP50':>10}{'mAP50-95':>12}{'Δ mAP50':>12}")
            print(f"  {'-'*46}")

            a_map50 = agg["mAP50_mean"]
            a_map50_95 = agg["mAP50-95_mean"]
            print(f"  {'A (curado)':<12}{a_map50:>10.4f}{a_map50_95:>12.4f}{'ref':>12}")

            for bl_name, comp in summary["comparison_vs_baselines"].items():
                bl_map50 = comp["baseline_mAP50"]
                bl_map50_95 = comp.get("baseline_mAP50-95", "?")
                delta = comp["delta_mAP50"]
                delta_str = f"{delta:+.4f}"
                print(f"  {bl_name:<12}{bl_map50:>10.4f}{bl_map50_95:>12.4f}{delta_str:>12}")

        # Salva
        summary_file = RUNS_DIR / "braco_a_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n  Relatório: {summary_file}")

    else:
        print(f"\n✗ Nenhuma seed completou com sucesso.")

    print(f"\n{'='*72}")
    print(f"  BRAÇO A FINALIZADO")
    print(f"  Tempo total: {elapsed_total/3600:.1f}h")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
