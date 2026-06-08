"""
treinar_random_copypaste_joint_v3.py

Treina o baseline "context-aware random copy-paste" (arm A' joint-rand)
com o MESMO regime joint balanced do A' joint.

═══════════════════════════════════════════════════════════════════
NOVIDADE v3 (vs v2):
  - Separa SEEDS (seeds a TREINAR nesta execução) de ALL_SEEDS
    (todas as seeds do experimento).
  - O results.json final SEMPRE inclui todas as seeds completas (não
    só as treinadas nesta execução), evitando sobrescrever resultados
    anteriores quando você roda uma seed por vez.
  - Faz backup automático do results.json anterior antes de sobrescrever.

NOVIDADES v2:
  - Aponta para dataset_random_copypaste_v4
  - Resumability: pula seeds que já têm best.pt
  - verify_paths() pré-treino
  - JSON results pronto para Tabela 6 do paper
═══════════════════════════════════════════════════════════════════

Como usar:
  - Primeira execução: SEEDS = [42],   ALL_SEEDS = [42, 123, 2024]
  - Segunda execução:  SEEDS = [123],  ALL_SEEDS = [42, 123, 2024]
  - Terceira execução: SEEDS = [2024], ALL_SEEDS = [42, 123, 2024]
  - Em todas, ALL_SEEDS é fixo. O results.json final tem n=3 sempre.

Uso:
    python treinar_random_copypaste_joint_v3.py
"""

from pathlib import Path
import shutil
import time
import json
import os
from datetime import datetime

# ═══ CONFIGURAÇÃO ═══
# Paths principais
CITRA_ROOT   = Path("/content/data/CITRA-3D-Real")
RCP_ROOT     = Path("/content/drive/MyDrive/InaTechShips/dataset_random_copypaste_v4")
COMBINED_DIR = Path("/content/data/combined_random_copypaste_v4")
RUNS_DIR     = Path("/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs/braco_random_copypaste_v4")
TEST_YAML    = CITRA_ROOT / "data_single_class.yaml"

# Subdiretório das labels reais (single-class, vessel only)
LABEL_SUBDIR_REAL = "labels_single_class"

# Experimento
# SEEDS    = seeds a TREINAR/AVALIAR nesta execução (mude entre sessões)
# ALL_SEEDS = todas as seeds previstas no experimento (fixo)
# O results.json final SEMPRE inclui ALL_SEEDS, não só SEEDS.
SEEDS      = [42]
ALL_SEEDS  = [42, 123, 2024]
OVERSAMPLE = 13   # real × 13 para balancear com 13 variações sintéticas

# Hiperparâmetros (idênticos ao A' joint do paper)
HYPER = dict(
    epochs        = 300,
    patience      = 30,
    optimizer     = "AdamW",
    lr0           = 0.001,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,
    cos_lr        = True,
    imgsz         = 640,
    batch         = 16,
    device        = 0,
    amp           = True,
    save          = True,
    exist_ok      = True,
)


# ═══ FUNÇÕES ═══

def verify_paths():
    """Confirma que paths necessários existem antes de começar."""
    print("Verificando paths...")
    errors = []

    # CITRA local
    for split in ["train", "val", "test"]:
        imgs_dir = CITRA_ROOT / split / "images"
        lbls_dir = CITRA_ROOT / split / LABEL_SUBDIR_REAL
        if not imgs_dir.exists():
            errors.append(f"  ✗ {imgs_dir} não existe")
        else:
            n = len(list(imgs_dir.glob("*")))
            print(f"  ✓ {imgs_dir.relative_to(CITRA_ROOT.parent)}: {n} imagens")
        if not lbls_dir.exists():
            errors.append(f"  ✗ {lbls_dir} não existe (labels single-class)")
        else:
            n = len(list(lbls_dir.glob("*.txt")))
            print(f"  ✓ {lbls_dir.relative_to(CITRA_ROOT.parent)}: {n} labels")

    # RCP v4 (random copy-paste gerado)
    for split in ["train", "val"]:
        imgs_dir = RCP_ROOT / split / "images"
        lbls_dir = RCP_ROOT / split / "labels"
        if not imgs_dir.exists():
            errors.append(f"  ✗ {imgs_dir} não existe (gerar primeiro com gerar_baseline_random_copypaste_v4.py)")
        else:
            n = len(list(imgs_dir.glob("*")))
            print(f"  ✓ RCP/{split}/images: {n} imagens")
        if not lbls_dir.exists():
            errors.append(f"  ✗ {lbls_dir} não existe")
        else:
            n = len(list(lbls_dir.glob("*.txt")))
            print(f"  ✓ RCP/{split}/labels: {n} labels")

    # YAML do test set
    if not TEST_YAML.exists():
        errors.append(f"  ✗ {TEST_YAML} não existe")
    else:
        print(f"  ✓ {TEST_YAML.relative_to(CITRA_ROOT.parent)}")

    if errors:
        print("\n❌ Erros encontrados:")
        for e in errors:
            print(e)
        raise FileNotFoundError("Paths inválidos. Corrija antes de prosseguir.")
    print("✓ Todos os paths OK.\n")


def create_combined_dataset(force=False):
    """Combina CITRA-3D-Real (13×) + random copy-paste v4 em
    /content/data/combined_random_copypaste_v4/ (local, rápido).

    Se o dataset já existe e está completo, pula a criação
    (a menos que force=True)."""

    meta_file = COMBINED_DIR / "combined_metadata.json"
    if meta_file.exists() and not force:
        meta = json.loads(meta_file.read_text())
        print(f"⏩ Dataset combinado já existe (criado em {meta['created_at']}):")
        for split, counts in meta["splits"].items():
            print(f"   {split}: {counts['real']} reais + {counts['synth']} synth = {counts['total']} total")
        return

    print("Criando dataset combinado...")
    t0 = time.time()
    splits_stats = {}

    for split in ["train", "val"]:
        out_imgs = COMBINED_DIR / split / "images"
        out_lbls = COMBINED_DIR / split / "labels"
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)

        # ── Imagens reais × OVERSAMPLE ──
        real_imgs_dir = CITRA_ROOT / split / "images"
        real_lbls_dir = CITRA_ROOT / split / LABEL_SUBDIR_REAL
        real_imgs = sorted(real_imgs_dir.glob("*"))
        count_real = 0
        for rep in range(OVERSAMPLE):
            for img in real_imgs:
                name = f"real_r{rep:02d}_{img.name}"
                shutil.copy2(img, out_imgs / name)
                lbl_src = real_lbls_dir / f"{img.stem}.txt"
                if lbl_src.exists():
                    shutil.copy2(lbl_src, out_lbls / f"real_r{rep:02d}_{img.stem}.txt")
                count_real += 1

        # ── Random copy-paste v4 ──
        rcp_imgs_dir = RCP_ROOT / split / "images"
        rcp_lbls_dir = RCP_ROOT / split / "labels"
        count_synth = 0
        for img in sorted(rcp_imgs_dir.glob("*")):
            shutil.copy2(img, out_imgs / img.name)
            lbl_src = rcp_lbls_dir / f"{img.stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_lbls / f"{img.stem}.txt")
            count_synth += 1

        total = count_real + count_synth
        print(f"  {split}: {count_real} reais + {count_synth} synth = {total} total")
        splits_stats[split] = {
            "real": count_real, "synth": count_synth, "total": total
        }

    # ── data.yaml para Ultralytics ──
    (COMBINED_DIR / "data.yaml").write_text(
        f"path: {COMBINED_DIR}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names:\n"
        f"  - embarcacao\n"
    )

    # ── Metadata ──
    metadata = {
        "created_at":  datetime.now().isoformat(),
        "elapsed_s":   round(time.time() - t0, 1),
        "oversample":  OVERSAMPLE,
        "sources": {
            "real_dataset":  str(CITRA_ROOT),
            "synth_dataset": str(RCP_ROOT),
        },
        "splits": splits_stats,
    }
    meta_file.write_text(json.dumps(metadata, indent=2))

    print(f"✓ Dataset combinado criado em {COMBINED_DIR} ({metadata['elapsed_s']}s)")
    print(f"  Metadata: {meta_file}")


def train_seed(seed):
    """Treina uma seed. Pula se já existe best.pt (resumability)."""
    from ultralytics import YOLO

    run_name = f"seed_{seed:04d}"
    best_pt  = RUNS_DIR / run_name / "weights" / "best.pt"

    if best_pt.exists():
        print(f"⏩ Seed {seed}: best.pt já existe, pulando treino.")
        # Apenas avaliar
        model = YOLO(str(best_pt))
    else:
        print(f"\n{'='*60}")
        print(f"  TREINO — A' joint-rand (v4) — seed {seed}")
        print(f"{'='*60}")
        model = YOLO("yolo11m.pt")
        model.train(
            data    = str(COMBINED_DIR / "data.yaml"),
            seed    = seed,
            project = str(RUNS_DIR),
            name    = run_name,
            **HYPER,
        )
        model = YOLO(str(best_pt))

    # Avaliação no test set real
    print(f"\nAvaliando seed {seed} no CITRA-3D-Real test set...")
    metrics = model.val(
        data   = str(TEST_YAML),
        split  = "test",
        device = HYPER["device"],
    )

    p   = float(metrics.box.p.mean())
    r   = float(metrics.box.r.mean())
    f1  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out = {
        "seed":      seed,
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": p,
        "recall":    r,
        "f1":        f1,
    }
    print(f"  ✓ Seed {seed}: mAP50={out['mAP50']:.4f}  "
          f"mAP50-95={out['mAP50_95']:.4f}  "
          f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    return out


def save_results(per_seed_results):
    """Salva results.json com per-seed + agregação mean ± std,
    formato pronto para a Tabela 6 do paper."""
    import statistics as st

    def agg(metric):
        vals = [r[metric] for r in per_seed_results]
        return {"mean": st.mean(vals),
                "std":  st.stdev(vals) if len(vals) > 1 else 0.0}

    results = {
        "arm":         "A' joint-rand",
        "description": "COCO → (CITRA-3D-Real × 13 + random-copypaste v4) balanced",
        "n_seeds":     len(per_seed_results),
        "seeds":       [r["seed"] for r in per_seed_results],
        "completed_at": datetime.now().isoformat(),
        "per_seed":    per_seed_results,
        "aggregate": {
            "mAP50":     agg("mAP50"),
            "mAP50_95":  agg("mAP50_95"),
            "precision": agg("precision"),
            "recall":    agg("recall"),
            "f1":        agg("f1"),
        }
    }

    out_file = RUNS_DIR / "results.json"
    out_file.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"  RESULTADOS AGREGADOS (n={len(per_seed_results)} seeds)")
    print(f"{'='*60}")
    a = results["aggregate"]
    print(f"  mAP50:     {a['mAP50']['mean']:.4f} ± {a['mAP50']['std']:.4f}")
    print(f"  mAP50-95:  {a['mAP50_95']['mean']:.4f} ± {a['mAP50_95']['std']:.4f}")
    print(f"  Precision: {a['precision']['mean']:.4f} ± {a['precision']['std']:.4f}")
    print(f"  Recall:    {a['recall']['mean']:.4f} ± {a['recall']['std']:.4f}")
    print(f"  F1:        {a['f1']['mean']:.4f} ± {a['f1']['std']:.4f}")
    print(f"\n  Resultados salvos em: {out_file}")


def main():
    t0 = time.time()

    # ── 1. Verificações pré-treino ──
    verify_paths()

    # ── 2. Dataset combinado ──
    create_combined_dataset()

    # ── 3. Backup do results.json anterior (se existir) ──
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RUNS_DIR / "results.json"
    if results_file.exists():
        from datetime import datetime as _dt
        bk = RUNS_DIR / f"results_backup_{_dt.now().strftime('%Y%m%d_%H%M%S')}.json"
        bk.write_text(results_file.read_text())
        print(f"📦 Backup do results.json anterior: {bk.name}")

    # ── 4. Processar TODAS as seeds (treinar as faltantes, avaliar as completas) ──
    print(f"\nIniciando processamento.")
    print(f"  Seeds previstas (ALL_SEEDS): {ALL_SEEDS}")
    print(f"  Seeds a treinar agora (SEEDS): {SEEDS}")
    print(f"  Demais seeds: apenas re-avaliadas se best.pt existir,")
    print(f"                puladas se não existir.")

    per_seed_results = []
    for seed in ALL_SEEDS:
        run_name = f"seed_{seed:04d}"
        best_pt  = RUNS_DIR / run_name / "weights" / "best.pt"

        if best_pt.exists():
            # Seed já tem pesos — re-avalia para coletar métricas
            print(f"\n→ Seed {seed}: best.pt encontrado, re-avaliando.")
            r = train_seed(seed)   # train_seed já pula treino se best.pt existir
            per_seed_results.append(r)
        elif seed in SEEDS:
            # Seed a treinar agora
            print(f"\n→ Seed {seed}: ainda não treinada, iniciando treino.")
            r = train_seed(seed)
            per_seed_results.append(r)
        else:
            # Seed prevista mas ainda não treinada e não na lista atual — pula
            print(f"\n→ Seed {seed}: ainda não treinada e fora da lista SEEDS desta execução. Pulando.")

        # Persistência incremental (caso quebre antes do fim)
        (RUNS_DIR / "_partial_results.json").write_text(
            json.dumps(per_seed_results, indent=2))

    # ── 5. Salvar resultados consolidados ──
    if per_seed_results:
        save_results(per_seed_results)
        # Limpa partial após sucesso
        partial = RUNS_DIR / "_partial_results.json"
        if partial.exists():
            partial.unlink()
    else:
        print("\n⚠️  Nenhum resultado coletado — verifique SEEDS e ALL_SEEDS.")

    print(f"\n✓ CONCLUÍDO em {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
