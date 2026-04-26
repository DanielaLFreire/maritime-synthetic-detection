"""
treinar_baselines.py

Treina os baselines B1 (random init) e B2 (COCO pretrain) do experimento,
com YOLOv11m no CITRA-3D-Real classe-única.

DESENHO

  B1 = YOLOv11m random init       → CITRA-3D-Real
  B2 = YOLOv11m COCO pretrain     → CITRA-3D-Real

  Cada baseline é treinado com 3 seeds (42, 123, 2024), totalizando
  6 corridas independentes.

CONFIGURAÇÃO CONSOLIDADA (decidida em 10/04/2026)

  Modelo:    YOLOv11m
  Épocas:    300 com early stopping
  Patience:  30
  imgsz:     640
  Batch:     16
  Optimizer: auto (Ultralytics decide; HPO refinará na tarefa A7)
  Dataset:   CITRA-3D-Real classe-única (via citra3d_single_class.yaml)
  Seeds:     42, 123, 2024

USO

  # Uma corrida específica
  python treinar_baselines.py --baseline B1 --seed 42
  python treinar_baselines.py --baseline B2 --seed 123

  # Todas as 6 corridas em sequência (deixar rodando à noite)
  python treinar_baselines.py --all

  # Listar o que seria executado sem rodar
  python treinar_baselines.py --all --dry-run

PRÉ-REQUISITOS

  Antes de rodar este script, na sessão atual do Colab:
    1. Drive montado
    2. `pip install ultralytics`
    3. `python gerar_data_yaml.py` (cria os links em /content/data/)

ESTRUTURA DE SAÍDA

  runs/baselines/B1_random/seed_42/
  ├── train/                       # diretório padrão do Ultralytics
  │   ├── weights/{best,last}.pt
  │   ├── results.csv
  │   ├── confusion_matrix.png
  │   └── ...
  ├── test_metrics.json            # avaliação no test set (estruturado)
  ├── run_summary.json             # config + duração + metadados
  └── log.txt                      # log textual do treino

  O caminho raiz tem espaço: "Experimento Dataset_Similar". Tratado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Yaml do dataset (gerado por gerar_data_yaml.py)
CITRA_YAML = Path(
    "/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/citra3d_single_class.yaml"
)

# Raiz dos resultados (atenção: tem espaço no nome da pasta)
RESULTS_ROOT = Path(
    "/content/drive/MyDrive/PROJETO_MARINHA/Experimento Dataset_Similar/runs/baselines"
)

# Modelo
MODEL_SIZE = "yolo11m"  # → yolo11m.pt para B2, yolo11m.yaml para B1

# ---------------------------------------------------------------------------
# Hiperparâmetros DIFERENCIADOS por baseline
#
# Justificativa: B1 (random init) e B2 (COCO pretrain) precisam de regimes
# de treino DIFERENTES, e isso é prática padrão na literatura.
#
# B1: random init exige LR maior, SGD, warmup curto. Treina do zero.
# B2: fine-tuning de COCO exige LR pequeno, AdamW, cosine decay controlado.
#     Caso contrário, os primeiros passos de gradiente destroem os pesos
#     COCO úteis. Foi exatamente isso que aconteceu no piloto inicial:
#     LR rampou de 0.00004 → 0.00175 ao longo de 33 épocas, mAP atingiu
#     pico em ~0.30 na época 3 e despencou para 0 com val_loss=NaN.
#
# Configurações abaixo são CONSERVADORAS e PADRÃO. Não são ainda o
# resultado de HPO (essa é a tarefa A7 do cronograma), mas garantem
# que cada baseline rode em regime adequado ao seu cenário.
# ---------------------------------------------------------------------------

# Parâmetros COMUNS aos dois baselines
COMMON_HYPERPARAMS = {
    "epochs": 300,
    "patience": 30,
    "imgsz": 640,
    "batch": 16,
    "save": True,
    "save_period": 20,
    "cache": True,
    "workers": 8,
    "verbose": True,
    "plots": True,
    "device": 0,
}

# Parâmetros ESPECÍFICOS de B1 (random init)
B1_HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    "optimizer": "SGD",
    "lr0": 0.01,           # LR inicial padrão para SGD do zero
    "lrf": 0.01,           # LR final = lr0 * lrf = 0.0001
    "momentum": 0.937,     # momentum padrão SGD do Ultralytics
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,        # cosine annealing explícito
}

# Parâmetros ESPECÍFICOS de B2 (COCO pretrain — fine-tuning)
B2_HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    "optimizer": "AdamW",
    "lr0": 0.001,          # ~10x menor que B1 — crítico para fine-tuning
    "lrf": 0.01,           # LR final = 0.00001
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.01,  # bias_lr menor para fine-tuning
    "cos_lr": True,
}

SEEDS = [42, 123, 2024]
BASELINES = ["B1", "B2"]

# Mapeamento de baselines → config + metadados
BASELINE_INFO = {
    "B1": {
        "dirname": "B1_random",
        "description": "YOLOv11m random init → CITRA-3D-Real",
        "weights": None,  # carregar arquitetura sem pesos
        "hyperparams": B1_HYPERPARAMS,
    },
    "B2": {
        "dirname": "B2_coco",
        "description": "YOLOv11m COCO pretrain → CITRA-3D-Real",
        "weights": "coco",  # carregar pesos COCO da Ultralytics
        "hyperparams": B2_HYPERPARAMS,
    },
}


# ---------------------------------------------------------------------------
# Pré-checagens
# ---------------------------------------------------------------------------

def precheck() -> bool:
    """Verifica que os pré-requisitos estão satisfeitos."""
    ok = True

    if not CITRA_YAML.exists():
        print(f"ERRO: yaml não encontrado: {CITRA_YAML}")
        print("      Rode `python gerar_data_yaml.py` primeiro.")
        ok = False
    else:
        print(f"  ✓ yaml encontrado: {CITRA_YAML}")

    # Validação leve do yaml
    if CITRA_YAML.exists():
        try:
            import yaml
            data = yaml.safe_load(CITRA_YAML.read_text())
            base = Path(data.get("path", ""))
            if not base.exists():
                print(f"ERRO: path no yaml não existe: {base}")
                print("      Rode `python gerar_data_yaml.py` para recriar links.")
                ok = False
            else:
                print(f"  ✓ path no yaml acessível: {base}")
                for split in ("train", "val", "test"):
                    split_dir = base / data.get(split, "")
                    n = sum(1 for _ in split_dir.iterdir()) if split_dir.is_dir() else -1
                    status = "✓" if n > 0 else "✗"
                    print(f"  {status} {split}: {n} arquivos em {split_dir}")
                    if n <= 0:
                        ok = False
        except Exception as exc:
            print(f"ERRO ao validar yaml: {exc}")
            ok = False

    # Ultralytics
    try:
        import ultralytics  # noqa: F401
        print(f"  ✓ ultralytics instalado: {ultralytics.__version__}")
    except ImportError:
        print("ERRO: ultralytics não instalado. Rode `pip install ultralytics`.")
        ok = False

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA disponível: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA NÃO disponível — treino vai ser lentíssimo na CPU.")
    except ImportError:
        print("ERRO: torch não instalado.")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------

def train_one(baseline: str, seed: int) -> dict:
    """Executa uma corrida de treino e avaliação."""
    info = BASELINE_INFO[baseline]
    hyperparams = info["hyperparams"]  # config específica B1 ou B2
    run_dir = RESULTS_ROOT / info["dirname"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"INICIANDO: {baseline} (seed={seed})")
    print(f"  descrição: {info['description']}")
    print(f"  saída:     {run_dir}")
    print(f"  optimizer: {hyperparams['optimizer']}, lr0={hyperparams['lr0']}, "
          f"lrf={hyperparams['lrf']}, cos_lr={hyperparams['cos_lr']}")
    print("=" * 78)

    summary = {
        "baseline": baseline,
        "description": info["description"],
        "seed": seed,
        "model_size": MODEL_SIZE,
        "weights_init": info["weights"],
        "data_yaml": str(CITRA_YAML),
        "hyperparams": dict(hyperparams),
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }

    # Salva summary parcial (caso o treino crashe, ainda há registro)
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    t0 = time.time()

    try:
        from ultralytics import YOLO

        # Carrega modelo
        if info["weights"] == "coco":
            model = YOLO(f"{MODEL_SIZE}.pt")  # baixa pesos COCO se necessário
            print(f">> Modelo carregado: {MODEL_SIZE}.pt (COCO pretrain)")
        else:
            model = YOLO(f"{MODEL_SIZE}.yaml")  # só arquitetura, sem pesos
            print(f">> Modelo carregado: {MODEL_SIZE}.yaml (random init)")

        # Treino
        # `project` define a pasta-mãe; `name` define a subpasta dessa corrida.
        # Definimos project = run_dir e name = "train" para ficar:
        #   {run_dir}/train/weights/best.pt etc.
        train_results = model.train(
            data=str(CITRA_YAML),
            project=str(run_dir),
            name="train",
            exist_ok=True,
            seed=seed,
            **hyperparams,
        )

        train_dir = run_dir / "train"
        best_pt = train_dir / "weights" / "best.pt"
        print(f">> Treino concluído. Best weights: {best_pt}")

        # Avaliação no test set usando os melhores pesos
        print()
        print(">> Avaliando no test set...")
        best_model = YOLO(str(best_pt))
        test_results = best_model.val(
            data=str(CITRA_YAML),
            split="test",
            project=str(run_dir),
            name="test_eval",
            exist_ok=True,
            verbose=True,
        )

        # Extrai métricas do test
        test_metrics = {
            "mAP50": float(test_results.box.map50),
            "mAP50-95": float(test_results.box.map),
            "precision": float(test_results.box.mp),
            "recall": float(test_results.box.mr),
            "fitness": float(test_results.fitness),
            "n_test_images": len(test_results.box.maps) if hasattr(test_results.box, "maps") else None,
        }

        # Tenta extrair breakdown por escala se disponível
        try:
            if hasattr(test_results, "box") and hasattr(test_results.box, "all_ap"):
                ap_per_class = test_results.box.all_ap
                if ap_per_class is not None:
                    test_metrics["ap_per_class_shape"] = list(ap_per_class.shape)
        except Exception:
            pass

        # Salva test metrics estruturado
        test_metrics_path = run_dir / "test_metrics.json"
        test_metrics_path.write_text(
            json.dumps(test_metrics, indent=2, ensure_ascii=False)
        )

        elapsed = time.time() - t0

        # Atualiza summary final
        summary.update({
            "finished_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_human": f"{elapsed/60:.1f} min",
            "status": "ok",
            "test_metrics": test_metrics,
            "best_weights": str(best_pt),
        })
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        print()
        print("=" * 78)
        print(f"OK: {baseline} seed={seed} concluído em {elapsed/60:.1f} min")
        print(f"   mAP50:    {test_metrics['mAP50']:.4f}")
        print(f"   mAP50-95: {test_metrics['mAP50-95']:.4f}")
        print(f"   precision: {test_metrics['precision']:.4f}")
        print(f"   recall:   {test_metrics['recall']:.4f}")
        print("=" * 78)

        return summary

    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print()
        print("!" * 78)
        print(f"FALHA: {baseline} seed={seed} após {elapsed/60:.1f} min")
        print(tb)
        print("!" * 78)

        summary.update({
            "finished_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "status": "error",
            "error": str(exc),
            "traceback": tb,
        })
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary


# ---------------------------------------------------------------------------
# Agregação de múltiplas runs
# ---------------------------------------------------------------------------

def aggregate_runs(results: list) -> None:
    """Imprime tabela consolidada das corridas concluídas."""
    print()
    print("=" * 78)
    print("RESUMO CONSOLIDADO DE TODAS AS CORRIDAS")
    print("=" * 78)
    print()
    print(f"{'baseline':<10}{'seed':<8}{'status':<10}{'mAP50':<10}{'mAP50-95':<12}{'tempo':<12}")
    print("-" * 78)
    for r in results:
        baseline = r.get("baseline", "?")
        seed = r.get("seed", "?")
        status = r.get("status", "?")
        elapsed = r.get("elapsed_human", "?")
        m50 = r.get("test_metrics", {}).get("mAP50")
        m95 = r.get("test_metrics", {}).get("mAP50-95")
        m50_str = f"{m50:.4f}" if m50 is not None else "—"
        m95_str = f"{m95:.4f}" if m95 is not None else "—"
        print(f"{baseline:<10}{seed:<8}{status:<10}{m50_str:<10}{m95_str:<12}{elapsed:<12}")

    # Salva agregado
    agg_path = RESULTS_ROOT / "all_runs_summary.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "n_runs": len(results),
        "runs": results,
    }, indent=2, ensure_ascii=False))
    print()
    print(f">> Agregado salvo em: {agg_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar baselines B1 e B2")
    parser.add_argument(
        "--baseline", choices=BASELINES,
        help="Qual baseline rodar (B1 ou B2). Ignorado se --all for usado.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Qual seed usar (42, 123 ou 2024). Ignorado se --all for usado.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Rodar todas as 6 combinações (B1+B2 × 3 seeds).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Apenas listar o que seria executado, sem treinar.",
    )
    args = parser.parse_args()

    print(">> treinar_baselines.py")
    print(f"   data.yaml:      {CITRA_YAML}")
    print(f"   results root:   {RESULTS_ROOT}")
    print(f"   modelo:         {MODEL_SIZE}")
    print(f"   épocas:         {COMMON_HYPERPARAMS['epochs']} (patience={COMMON_HYPERPARAMS['patience']})")
    print(f"   imgsz:          {COMMON_HYPERPARAMS['imgsz']}")
    print(f"   batch:          {COMMON_HYPERPARAMS['batch']}")
    print()
    print(f"   B1: optimizer={B1_HYPERPARAMS['optimizer']}, lr0={B1_HYPERPARAMS['lr0']}, "
          f"lrf={B1_HYPERPARAMS['lrf']}, cos_lr={B1_HYPERPARAMS['cos_lr']}")
    print(f"   B2: optimizer={B2_HYPERPARAMS['optimizer']}, lr0={B2_HYPERPARAMS['lr0']}, "
          f"lrf={B2_HYPERPARAMS['lrf']}, cos_lr={B2_HYPERPARAMS['cos_lr']}")
    print()

    print(">> Pré-checagens")
    if not precheck():
        print()
        print("PRÉ-CHECAGENS FALHARAM. Resolva os problemas acima antes de continuar.")
        sys.exit(1)

    # Determinar a fila de execução
    if args.all:
        queue = [(b, s) for b in BASELINES for s in SEEDS]
    elif args.baseline and args.seed is not None:
        if args.seed not in SEEDS:
            print(f"AVISO: seed {args.seed} não é uma das padrão {SEEDS}")
        queue = [(args.baseline, args.seed)]
    else:
        parser.error("Forneça --baseline e --seed, ou use --all")
        return

    print()
    print(f">> Fila de execução: {len(queue)} corrida(s)")
    for b, s in queue:
        print(f"   - {b} seed={s}")

    if args.dry_run:
        print()
        print(">> --dry-run: nada será executado.")
        return

    # Executar
    results = []
    for i, (baseline, seed) in enumerate(queue, start=1):
        print()
        print(f"##### Corrida {i}/{len(queue)} #####")
        result = train_one(baseline, seed)
        results.append(result)

    # Agregar
    if len(results) > 1:
        aggregate_runs(results)


if __name__ == "__main__":
    main()
