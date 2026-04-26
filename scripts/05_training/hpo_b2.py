"""
hpo_b2.py

Hyperparameter Optimization para o baseline B2 (YOLOv11m + COCO pretrain
+ fine-tuning CITRA-3D-Real), em duas fases:

FASE 1 — Exploração (Optuna TPE, 30 trials × 100 épocas)
  Busca num espaço de 5 dimensões em torno da configuração conservadora
  atual de B2. Cada trial roda 100 épocas com patience=15, seed=42.
  Métrica: best val fitness atingido durante o trial, onde fitness é
  o padrão do Ultralytics (0.1 * mAP50 + 0.9 * mAP50-95).

FASE 2 — Validação (1 treino completo)
  Pega o TOP-1 da Fase 1 e roda um treino completo (300 épocas, patience=30,
  seed=42, igual aos baselines). Avalia no test set. Compara com B2 baseline.
  Decisão automática:
    - Se ganho > 3 × DP ≈ 0.0072 em mAP50 vs B2 baseline: REFAZER 3 seeds
    - Caso contrário: MANTER config atual

Justificativa do threshold (3 × DP):
  Variância observada em B2 entre seeds = 0.0024 em mAP50. Um ganho menor
  que 3 × essa variância pode ser ruído de seed única. Esta formulação é
  derivada empiricamente, não arbitrária.

Espaço de busca (5 dimensões):
  lr0           : log-uniform [0.0003, 0.003]
  lrf           : log-uniform [0.005,  0.05 ]
  momentum      : uniform     [0.85,   0.95 ]
  weight_decay  : log-uniform [0.0001, 0.001]
  warmup_epochs : int uniform [1,      5    ]

Fixados (não tuned, controle experimental):
  optimizer        = AdamW
  cos_lr           = True
  warmup_bias_lr   = 0.01
  warmup_momentum  = 0.8
  imgsz            = 640
  batch            = 16
  device           = 0

USO

  # Fase 1 (exploração) — começa do zero
  python hpo_b2.py --phase 1 --n-trials 30

  # Fase 1 — retomar de onde parou (Optuna salva estudo persistente)
  python hpo_b2.py --phase 1 --n-trials 30 --resume

  # Fase 2 (validação do TOP-1)
  python hpo_b2.py --phase 2

  # Fase 1 + Fase 2 em sequência (deixar rodando à noite)
  python hpo_b2.py --phase all --n-trials 30

  # Dry-run: imprime config sem rodar
  python hpo_b2.py --phase all --dry-run

PRÉ-REQUISITOS

  No início da sessão Colab:
    1. Drive montado
    2. !pip install ultralytics optuna
    3. !python preparar_dados_locais.py --only citra
       (recopia CITRA-3D-Real para /content/data)

ESTRUTURA DE SAÍDA

  Drive/.../Experimento_Dataset_Similar/hpo/B2/
  ├── optuna_study.db              # estudo Optuna persistente (SQLite)
  ├── trials/
  │   ├── trial_000/train/...      # output Ultralytics de cada trial
  │   ├── trial_001/...
  │   └── ...
  ├── trial_summary.json           # log estruturado de todos os trials
  ├── phase2_validation/
  │   ├── train/                   # treino completo do TOP-1
  │   ├── test_eval/
  │   └── result.json              # comparação vs B2 baseline + decisão
  └── hpo_report.json              # relatório final consolidado
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CITRA_YAML = Path(
    "/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/citra3d_single_class.yaml"
)

HPO_ROOT = Path(
    "/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/hpo/B2"
)

# Estudo Optuna persistente (SQLite no Drive)
OPTUNA_STORAGE = f"sqlite:///{HPO_ROOT}/optuna_study.db"
OPTUNA_STUDY_NAME = "B2_hpo"

MODEL_SIZE = "yolo11m"
SEED = 42

# Fase 1 — exploração
PHASE1_EPOCHS = 100
PHASE1_PATIENCE = 15

# Fase 2 — validação completa (igual aos baselines)
PHASE2_EPOCHS = 300
PHASE2_PATIENCE = 30

# Baseline B2 atual (de all_runs_summary.json) — referência para comparação
B2_BASELINE_MAP50_MEAN = 0.8351
B2_BASELINE_MAP50_DP = 0.0024
B2_BASELINE_MAP5095_MEAN = 0.5055
B2_BASELINE_MAP5095_DP = 0.0027
# fitness baseline = 0.1 * mAP50 + 0.9 * mAP50-95
B2_BASELINE_FITNESS = 0.1 * B2_BASELINE_MAP50_MEAN + 0.9 * B2_BASELINE_MAP5095_MEAN

# Threshold para refazer baselines: 3 × DP em mAP50
DECISION_THRESHOLD_PTS = 3 * B2_BASELINE_MAP50_DP  # ≈ 0.0072

# Hyperparams comuns (fixos)
COMMON_HYPERPARAMS = {
    "imgsz": 640,
    "batch": 16,
    "save": True,
    "save_period": 0,           # não salvar checkpoints intermediários (economia)
    "cache": True,
    "workers": 8,
    "verbose": False,           # silenciar Ultralytics em trials
    "plots": False,             # não gerar plots em cada trial
    "device": 0,
    "optimizer": "AdamW",       # fixo
    "cos_lr": True,             # fixo
    "warmup_bias_lr": 0.01,     # fixo
    "warmup_momentum": 0.8,     # fixo
}


# ---------------------------------------------------------------------------
# Pré-checagens
# ---------------------------------------------------------------------------

def precheck() -> bool:
    ok = True

    if not CITRA_YAML.exists():
        print(f"ERRO: yaml não encontrado: {CITRA_YAML}")
        print("      Rode `python preparar_dados_locais.py --only citra` primeiro.")
        ok = False
    else:
        print(f"  ✓ yaml: {CITRA_YAML}")

    try:
        import ultralytics
        print(f"  ✓ ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("ERRO: ultralytics não instalado. `pip install ultralytics`")
        ok = False

    try:
        import optuna
        print(f"  ✓ optuna: {optuna.__version__}")
    except ImportError:
        print("ERRO: optuna não instalado. `pip install optuna`")
        ok = False

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA indisponível — HPO vai ser MUITO lento na CPU.")
    except ImportError:
        ok = False

    HPO_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ HPO root: {HPO_ROOT}")

    return ok


# ---------------------------------------------------------------------------
# Função objetivo (Optuna)
# ---------------------------------------------------------------------------

def make_objective(trials_dir: Path, summary_log: list):
    """
    Cria a função de objetivo do Optuna. Closures sobre trials_dir e
    summary_log para evitar variáveis globais.
    """

    def objective(trial) -> float:
        from ultralytics import YOLO

        # ---- Espaço de busca ----
        lr0 = trial.suggest_float("lr0", 0.0003, 0.003, log=True)
        lrf = trial.suggest_float("lrf", 0.005, 0.05, log=True)
        momentum = trial.suggest_float("momentum", 0.85, 0.95)
        weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.001, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)

        hp = {
            **COMMON_HYPERPARAMS,
            "epochs": PHASE1_EPOCHS,
            "patience": PHASE1_PATIENCE,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": float(warmup_epochs),
        }

        trial_dir = trials_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f">>>> Trial {trial.number:03d}")
        print(f"     lr0={lr0:.5f} lrf={lrf:.5f} momentum={momentum:.4f} "
              f"wd={weight_decay:.5f} warmup={warmup_epochs}")

        t0 = time.time()
        record = {
            "trial_number": trial.number,
            "started_at": datetime.now().isoformat(),
            "params": {
                "lr0": lr0, "lrf": lrf, "momentum": momentum,
                "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
            },
            "status": "running",
        }

        try:
            model = YOLO(f"{MODEL_SIZE}.pt")
            results = model.train(
                data=str(CITRA_YAML),
                project=str(trial_dir),
                name="train",
                exist_ok=True,
                seed=SEED,
                **hp,
            )

            # Extrai best fitness do treino (a métrica usada para early stop
            # e seleção de best.pt já é o fitness do Ultralytics)
            train_dir = trial_dir / "train"
            results_csv = train_dir / "results.csv"

            best_fitness = None
            best_map50 = None
            best_map5095 = None

            if results_csv.exists():
                # Lê results.csv e extrai best fitness atingido
                import pandas as pd
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                # Computa fitness explicitamente (1.0 padrão do Ultralytics =
                # 0.1 * mAP50 + 0.9 * mAP50-95)
                if "metrics/mAP50(B)" in df.columns and "metrics/mAP50-95(B)" in df.columns:
                    df["fitness"] = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
                    best_idx = df["fitness"].idxmax()
                    best_fitness = float(df.loc[best_idx, "fitness"])
                    best_map50 = float(df.loc[best_idx, "metrics/mAP50(B)"])
                    best_map5095 = float(df.loc[best_idx, "metrics/mAP50-95(B)"])
                    n_epochs = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else None
                else:
                    print("     ⚠ colunas esperadas não encontradas em results.csv")
                    best_fitness = 0.0
                    n_epochs = None
            else:
                print("     ⚠ results.csv não encontrado")
                best_fitness = 0.0
                n_epochs = None

            elapsed = time.time() - t0
            record.update({
                "finished_at": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "elapsed_human": f"{elapsed/60:.1f} min",
                "status": "ok",
                "best_fitness": best_fitness,
                "best_val_mAP50": best_map50,
                "best_val_mAP50_95": best_map5095,
                "best_epoch": n_epochs,
            })

            print(f"     ✓ trial {trial.number:03d} concluído em {elapsed/60:.1f} min")
            print(f"       best fitness = {best_fitness:.4f} "
                  f"(mAP50={best_map50:.4f}, mAP50-95={best_map5095:.4f})")

            summary_log.append(record)
            _save_summary(trials_dir.parent, summary_log)

            return best_fitness

        except Exception as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            print(f"     ✗ trial {trial.number:03d} FALHOU após {elapsed/60:.1f} min")
            print(tb)
            record.update({
                "finished_at": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "status": "error",
                "error": str(exc),
                "traceback": tb,
            })
            summary_log.append(record)
            _save_summary(trials_dir.parent, summary_log)
            # Retorna valor baixo para Optuna entender que esse trial é ruim
            return 0.0

    return objective


def _save_summary(hpo_dir: Path, summary_log: list) -> None:
    """Salva log estruturado de trials."""
    path = hpo_dir / "trial_summary.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "n_trials": len(summary_log),
        "trials": summary_log,
    }, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Fase 1: exploração com Optuna
# ---------------------------------------------------------------------------

def run_phase1(n_trials: int, resume: bool) -> dict:
    """Executa busca Optuna TPE."""
    import optuna
    from optuna.samplers import TPESampler

    print()
    print("=" * 78)
    print(f"FASE 1 — Exploração com Optuna TPE ({n_trials} trials × {PHASE1_EPOCHS} épocas)")
    print("=" * 78)

    trials_dir = HPO_ROOT / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    summary_log: list = []
    summary_path = HPO_ROOT / "trial_summary.json"
    if resume and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            summary_log = existing.get("trials", [])
            print(f">> Retomando: {len(summary_log)} trials anteriores no log")
        except Exception:
            pass

    # Cria estudo persistente
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = len(study.trials)
    if n_done > 0:
        print(f">> Estudo Optuna existente: {n_done} trials já completos")
        print(f"   Best fitness até agora: {study.best_value:.4f}")

    n_remaining = max(0, n_trials - n_done)
    if n_remaining == 0:
        print(f">> Já temos {n_done} >= {n_trials} trials. Pulando para análise.")
    else:
        print(f">> Vão rodar {n_remaining} novos trials")
        objective = make_objective(trials_dir, summary_log)
        study.optimize(objective, n_trials=n_remaining)

    # ---- Análise final da Fase 1 ----
    print()
    print("=" * 78)
    print("FASE 1 — RESULTADO")
    print("=" * 78)

    completed_trials = [t for t in study.trials if t.value is not None and t.value > 0]
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    if not completed_trials:
        print("ERRO: nenhum trial completou com sucesso.")
        return {"status": "error", "message": "no completed trials"}

    print(f"\n>> Trials completos: {len(completed_trials)}/{len(study.trials)}")
    print(f"\n>> TOP 5:")
    print(f"   {'rank':<6}{'trial':<8}{'fitness':<12}{'lr0':<10}{'lrf':<10}"
          f"{'momentum':<11}{'wd':<10}{'warmup':<8}")
    print("   " + "-" * 75)
    for rank, t in enumerate(completed_trials[:5], start=1):
        p = t.params
        print(
            f"   #{rank:<5}{t.number:<8}{t.value:<12.5f}"
            f"{p['lr0']:<10.5f}{p['lrf']:<10.5f}"
            f"{p['momentum']:<11.4f}{p['weight_decay']:<10.5f}"
            f"{p['warmup_epochs']:<8}"
        )

    best_trial = completed_trials[0]
    print(f"\n>> Best trial: #{best_trial.number}")
    print(f"   fitness = {best_trial.value:.5f}")
    print(f"   B2 baseline fitness ≈ {B2_BASELINE_FITNESS:.5f}")
    delta = best_trial.value - B2_BASELINE_FITNESS
    print(f"   Δ fitness vs baseline = {delta:+.5f}")

    return {
        "status": "ok",
        "n_trials_completed": len(completed_trials),
        "best_trial_number": best_trial.number,
        "best_fitness": best_trial.value,
        "best_params": dict(best_trial.params),
        "baseline_fitness": B2_BASELINE_FITNESS,
        "delta_fitness": delta,
        "top5": [
            {
                "rank": rank,
                "trial_number": t.number,
                "fitness": t.value,
                "params": dict(t.params),
            }
            for rank, t in enumerate(completed_trials[:5], start=1)
        ],
    }


# ---------------------------------------------------------------------------
# Fase 2: validação do TOP-1 com treino completo
# ---------------------------------------------------------------------------

def run_phase2(phase1_result: dict) -> dict:
    """Roda treino completo (300 épocas) com a config TOP-1 da Fase 1."""
    if phase1_result.get("status") != "ok":
        print("ERRO: Fase 1 falhou, não posso rodar Fase 2.")
        return {"status": "skipped", "reason": "phase1 failed"}

    best_params = phase1_result["best_params"]
    best_trial_num = phase1_result["best_trial_number"]

    print()
    print("=" * 78)
    print(f"FASE 2 — Validação do TOP-1 (trial #{best_trial_num})")
    print(f"        Treino completo: {PHASE2_EPOCHS} épocas, patience={PHASE2_PATIENCE}, seed={SEED}")
    print("=" * 78)
    print(f">> Config a validar:")
    for k, v in best_params.items():
        print(f"   {k} = {v}")

    from ultralytics import YOLO

    val_dir = HPO_ROOT / "phase2_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    hp = {
        **COMMON_HYPERPARAMS,
        "epochs": PHASE2_EPOCHS,
        "patience": PHASE2_PATIENCE,
        "lr0": best_params["lr0"],
        "lrf": best_params["lrf"],
        "momentum": best_params["momentum"],
        "weight_decay": best_params["weight_decay"],
        "warmup_epochs": float(best_params["warmup_epochs"]),
        # Reativa plots e verbose para o treino de validação (queremos curvas)
        "verbose": True,
        "plots": True,
    }

    t0 = time.time()
    try:
        model = YOLO(f"{MODEL_SIZE}.pt")
        train_results = model.train(
            data=str(CITRA_YAML),
            project=str(val_dir),
            name="train",
            exist_ok=True,
            seed=SEED,
            **hp,
        )

        train_dir = val_dir / "train"
        best_pt = train_dir / "weights" / "best.pt"
        print(f"\n>> Treino concluído. Best weights: {best_pt}")

        # Avalia no TEST set (igual aos baselines, não no val)
        print(">> Avaliando no test set...")
        best_model = YOLO(str(best_pt))
        test_results = best_model.val(
            data=str(CITRA_YAML),
            split="test",
            project=str(val_dir),
            name="test_eval",
            exist_ok=True,
            verbose=True,
        )

        test_map50 = float(test_results.box.map50)
        test_map5095 = float(test_results.box.map)
        test_precision = float(test_results.box.mp)
        test_recall = float(test_results.box.mr)
        test_fitness = 0.1 * test_map50 + 0.9 * test_map5095

        elapsed = time.time() - t0

        # ---- Decisão automática ----
        delta_map50 = test_map50 - B2_BASELINE_MAP50_MEAN
        delta_map5095 = test_map5095 - B2_BASELINE_MAP5095_MEAN
        delta_fitness = test_fitness - B2_BASELINE_FITNESS

        # Threshold: 3 × DP em mAP50
        threshold_in_dps = delta_map50 / B2_BASELINE_MAP50_DP if B2_BASELINE_MAP50_DP > 0 else 0
        meets_threshold = delta_map50 > DECISION_THRESHOLD_PTS

        decision = "REFAZER_BASELINES" if meets_threshold else "MANTER_CONFIG"

        result = {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "elapsed_human": f"{elapsed/60:.1f} min",
            "config_used": dict(best_params),
            "test_metrics": {
                "mAP50": test_map50,
                "mAP50-95": test_map5095,
                "precision": test_precision,
                "recall": test_recall,
                "fitness": test_fitness,
            },
            "baseline_b2": {
                "mAP50_mean": B2_BASELINE_MAP50_MEAN,
                "mAP50_dp": B2_BASELINE_MAP50_DP,
                "mAP50_95_mean": B2_BASELINE_MAP5095_MEAN,
                "mAP50_95_dp": B2_BASELINE_MAP5095_DP,
                "fitness_mean": B2_BASELINE_FITNESS,
            },
            "comparison": {
                "delta_mAP50": delta_map50,
                "delta_mAP50_95": delta_map5095,
                "delta_fitness": delta_fitness,
                "delta_in_dps_mAP50": threshold_in_dps,
                "threshold_dps": 3.0,
                "threshold_pts_mAP50": DECISION_THRESHOLD_PTS,
                "meets_threshold": meets_threshold,
            },
            "decision": decision,
            "best_weights": str(best_pt),
        }

        # Salva resultado
        result_path = val_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        # Resumo legível
        print()
        print("=" * 78)
        print("FASE 2 — RESULTADO")
        print("=" * 78)
        print(f"  Tempo de treino: {elapsed/60:.1f} min")
        print()
        print(f"  {'métrica':<15}{'HPO TOP-1':<14}{'B2 baseline':<14}{'Δ':<14}")
        print(f"  {'-' * 55}")
        print(f"  {'mAP50':<15}{test_map50:<14.4f}{B2_BASELINE_MAP50_MEAN:<14.4f}{delta_map50:+.4f}")
        print(f"  {'mAP50-95':<15}{test_map5095:<14.4f}{B2_BASELINE_MAP5095_MEAN:<14.4f}{delta_map5095:+.4f}")
        print(f"  {'fitness':<15}{test_fitness:<14.4f}{B2_BASELINE_FITNESS:<14.4f}{delta_fitness:+.4f}")
        print()
        print(f"  Threshold para refazer baselines: 3 × DP = {DECISION_THRESHOLD_PTS:.4f} pts mAP50")
        print(f"  Δ observado em mAP50:             {delta_map50:+.4f} ({threshold_in_dps:+.2f} × DP)")
        print()
        if meets_threshold:
            print(f"  >>>> DECISÃO: REFAZER BASELINES <<<<")
            print(f"  O ganho ({delta_map50:+.4f}) supera o threshold ({DECISION_THRESHOLD_PTS:.4f}).")
            print(f"  Recomendação: rodar `treinar_baselines.py --baseline B2 --seed 42`,")
            print(f"  depois 123 e 2024, com a nova config (vide config_used em result.json).")
        else:
            print(f"  >>>> DECISÃO: MANTER CONFIG ATUAL <<<<")
            print(f"  O ganho ({delta_map50:+.4f}) NÃO supera o threshold ({DECISION_THRESHOLD_PTS:.4f}).")
            print(f"  A config atual foi validada via HPO formal e está dentro de 3 DPs do ótimo encontrado.")
        print("=" * 78)

        return result

    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"\n!!!! FASE 2 FALHOU após {elapsed/60:.1f} min")
        print(tb)
        return {
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(exc),
            "traceback": tb,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HPO para baseline B2")
    parser.add_argument(
        "--phase", choices=["1", "2", "all"], default="all",
        help="Qual fase rodar (default: all)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="Número de trials Optuna na Fase 1 (default: 30)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Retomar estudo Optuna existente em vez de começar do zero",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Apenas imprime configuração sem rodar",
    )
    args = parser.parse_args()

    print(">> hpo_b2.py")
    print(f"   data.yaml:     {CITRA_YAML}")
    print(f"   HPO root:      {HPO_ROOT}")
    print(f"   modelo:        {MODEL_SIZE}")
    print(f"   seed:          {SEED}")
    print(f"   fase:          {args.phase}")
    if args.phase in ("1", "all"):
        print(f"   Fase 1:        {args.n_trials} trials × {PHASE1_EPOCHS} épocas (patience {PHASE1_PATIENCE})")
    if args.phase in ("2", "all"):
        print(f"   Fase 2:        1 treino completo × {PHASE2_EPOCHS} épocas (patience {PHASE2_PATIENCE})")
    print(f"   Threshold:     ganho > {DECISION_THRESHOLD_PTS:.4f} mAP50 ({3} × DP) → refazer baselines")
    print()

    print(">> Pré-checagens")
    if not precheck():
        print()
        print("PRÉ-CHECAGENS FALHARAM. Resolva antes de continuar.")
        sys.exit(1)

    if args.dry_run:
        print("\n>> --dry-run: nada será executado.")
        return

    # Estimativa de custo
    if args.phase in ("1", "all"):
        # ~3 min/época × 100 épocas = ~5h por trial em pior caso, mas trials
        # convergem cedo via early stopping → estimar ~30-50 min/trial
        est_min_per_trial = 40
        est_phase1 = est_min_per_trial * args.n_trials / 60
        print(f">> Estimativa Fase 1: ~{est_phase1:.1f}h ({est_min_per_trial} min/trial × {args.n_trials} trials)")
    if args.phase in ("2", "all"):
        print(f">> Estimativa Fase 2: ~50-60 min")

    overall = {
        "started_at": datetime.now().isoformat(),
        "phase": args.phase,
        "n_trials_requested": args.n_trials,
    }
    t_start = time.time()

    # ---- Fase 1 ----
    phase1_result = None
    if args.phase in ("1", "all"):
        phase1_result = run_phase1(args.n_trials, args.resume)
        overall["phase1"] = phase1_result

    # Carrega Fase 1 do disco se phase=2 sozinha (pega TOP-1 do estudo persistente)
    if args.phase == "2":
        import optuna
        try:
            study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE)
            completed = [t for t in study.trials if t.value is not None and t.value > 0]
            completed.sort(key=lambda t: t.value, reverse=True)
            if not completed:
                print("ERRO: estudo Optuna existe mas não tem trials válidos.")
                sys.exit(1)
            best = completed[0]
            phase1_result = {
                "status": "ok",
                "best_trial_number": best.number,
                "best_fitness": best.value,
                "best_params": dict(best.params),
            }
            print(f">> Fase 2 isolada: usando TOP-1 do estudo existente (trial #{best.number}, "
                  f"fitness={best.value:.5f})")
        except Exception as exc:
            print(f"ERRO carregando estudo existente: {exc}")
            sys.exit(1)

    # ---- Fase 2 ----
    if args.phase in ("2", "all") and phase1_result:
        phase2_result = run_phase2(phase1_result)
        overall["phase2"] = phase2_result

    # ---- Relatório final ----
    overall["finished_at"] = datetime.now().isoformat()
    overall["total_elapsed_seconds"] = time.time() - t_start
    overall["total_elapsed_human"] = f"{(time.time() - t_start)/3600:.1f}h"

    report_path = HPO_ROOT / "hpo_report.json"
    report_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False))

    print()
    print("=" * 78)
    print("HPO COMPLETO")
    print("=" * 78)
    print(f"  Tempo total:    {overall['total_elapsed_human']}")
    print(f"  Relatório:      {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
