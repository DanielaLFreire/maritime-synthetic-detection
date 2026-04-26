"""
reconstruir_dataset_25k.py

Reconstrói o dataset_25k a partir dos 27.796 IDs únicos, com splits
disjuntos e estratificação por classe.

CONTEXTO

  O dataset_25k atual no Drive (/content/drive/MyDrive/InaTechShips/
  dataset_25k/) tem 38.109 arquivos distribuídos em train/val/test, mas
  apenas 27.796 IDs únicos. Há sobreposição entre splits:

    train ∩ val:   5.253 imagens (idênticas, mesmo MD5)
    train ∩ test:  3.777
    val ∩ test:    1.283
    triplas:           0
    total duplicado: 10.313 (~37% do pool)

  Causa provável: o notebook de montagem (InaTechShips.ipynb, Seção 11)
  não limpa DATASET_PATH antes de cada execução, então execuções
  sucessivas com pools de metadados diferentes deixaram arquivos órfãos
  de runs antigos junto com novos.

  Esse bug invalida o uso do dataset_25k atual para train/val em
  qualquer experimento (modelo treina em imagens que vê na validação).

  Os baselines B1/B2 que já rodaram NÃO foram afetados — eles usam
  CITRA-3D-Real, não dataset_25k.

O QUE ESTE SCRIPT FAZ

  1. Lê o dataset_25k atual e coleta os 27.796 IDs únicos via
     união dos 3 splits.
  2. Para cada ID, lê o class_id da primeira linha do label
     correspondente em qualquer split onde ele apareça.
  3. Agrupa IDs por class_id (10 classes).
  4. Para cada classe: shuffle reprodutível (seed=42) + split
     60/20/20 estratificado.
  5. Copia imagens + labels (10-class) + labels_single_class (1-class)
     para a estrutura nova em dataset_25k_v2/{split}/{images,labels,
     labels_single_class}/.
  6. Gera data.yaml apontando para a nova estrutura.
  7. Gera relatório JSON validando: disjunção total entre splits,
     contagem por classe em cada split, tamanhos finais.

PRÉ-REQUISITOS

  - Roda no Colab (Drive montado).
  - Drive: /content/drive/MyDrive/InaTechShips/dataset_25k/ existe e
    contém train/val/test/{images,labels,labels_single_class}/.

USO

  python reconstruir_dataset_25k.py
  python reconstruir_dataset_25k.py --dry-run    # só calcula, não copia
  python reconstruir_dataset_25k.py --src /caminho/diferente   # opcional

SAÍDA

  /content/drive/MyDrive/InaTechShips/dataset_25k_v2/
  ├── train/
  │   ├── images/           (~16.673 .jpg)
  │   ├── labels/           (~16.673 .txt — 10 classes)
  │   └── labels_single_class/  (~16.673 .txt — 1 classe)
  ├── val/                  (~5.557)
  ├── test/                 (~5.566)
  ├── data.yaml             (config Ultralytics, nc=10)
  ├── data_single_class.yaml (config single-class, nc=1)
  └── reconstrucao_report.json (validação completa)

REPRODUTIBILIDADE

  Seed 42 (mesma do notebook original). Para o mesmo dataset_25k de
  entrada, sempre produz o mesmo split.

TEMPO ESPERADO

  ~30-60 min no Drive (gargalo de I/O do Drive). Cada arquivo é uma
  chamada de rede para o Google. Total ~83.388 cópias (27.796 × 3:
  imagens, labels 10-class, labels single-class).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

SRC_ROOT = Path("/content/drive/MyDrive/InaTechShips/dataset_25k")
DST_ROOT = Path("/content/drive/MyDrive/InaTechShips/dataset_25k_v2")
REPORT_FILE = DST_ROOT / "reconstrucao_report.json"

SEED = 42
SPLITS = ("train", "val", "test")
SPLIT_PROPORTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}

CLASS_NAMES = [
    "BULK CARRIER",
    "RO-RO/PASSENGER SHIP",
    "VEHICLES CARRIER",
    "PASSENGERS SHIP",
    "RO-RO CARGO",
    "CONTAINER SHIP",
    "OIL PRODUCTS TANKER",
    "GENERAL CARGO",
    "TUG",
    "OIL/CHEMICAL TANKER",
]
N_CLASSES = len(CLASS_NAMES)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def read_class_id(label_file: Path) -> int | None:
    """Lê o class_id da primeira linha do arquivo de label."""
    try:
        with open(label_file) as f:
            line = f.readline().strip()
        if not line:
            return None
        return int(line.split()[0])
    except (OSError, ValueError, IndexError):
        return None


def collect_ids_with_classes(src_root: Path) -> dict[int, int]:
    """
    Coleta todos os IDs únicos do dataset_25k atual e seus class_ids.

    Retorna: {photo_id: class_id}.

    Para IDs duplicados entre splits (mesmo arquivo), pega o class_id
    do primeiro split onde encontrar (todos os duplicados têm o mesmo
    class_id porque os arquivos são idênticos — verificado por MD5).
    """
    id_to_class: dict[int, int] = {}
    files_seen = 0

    for split in SPLITS:
        labels_dir = src_root / split / "labels"
        if not labels_dir.exists():
            print(f"   ⚠ {labels_dir} não existe, pulando")
            continue

        for label_file in labels_dir.iterdir():
            if label_file.suffix != ".txt":
                continue
            files_seen += 1
            try:
                photo_id = int(label_file.stem)
            except ValueError:
                continue

            if photo_id in id_to_class:
                continue  # já temos esse ID de outro split

            class_id = read_class_id(label_file)
            if class_id is None:
                continue

            id_to_class[photo_id] = class_id

    return id_to_class


def stratified_split_per_class(
    ids_by_class: dict[int, list[int]],
    proportions: dict[str, float],
    seed: int,
) -> dict[str, list[int]]:
    """
    Para cada classe, faz split estratificado nas proporções dadas.

    Retorna: {split_name: [photo_id, ...]}.

    Usa arredondamento equilibrado: train = int(p_train * N), val = round(p_val * N),
    test = N - train - val. Para classes com muitos IDs (>1000), as proporções
    finais batem quase exatamente (60/20/20). Para classes pequenas pode haver
    pequenas variações que se distribuem entre train/val/test sem perder IDs.

    O split é determinístico para uma seed dada. A ordem de iteração sobre as
    classes é fixa (sorted) para reprodutibilidade.
    """
    rng = random.Random(seed)
    split_ids = {split: [] for split in proportions}

    for class_id in sorted(ids_by_class.keys()):
        class_ids = list(ids_by_class[class_id])
        rng.shuffle(class_ids)

        n_total = len(class_ids)
        # Arredondamento equilibrado: train = floor, val = round, test = resto
        # Garante n_train + n_val + n_test == n_total sem perder IDs
        n_train = int(round(proportions["train"] * n_total))
        n_val = int(round(proportions["val"] * n_total))
        # Ajuste se arredondamento causou overflow
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        n_test = n_total - n_train - n_val

        # Garantia mínima: cada split tem pelo menos 1 ID se total >= 3
        if n_total >= 3 and (n_val == 0 or n_test == 0):
            # Caso degenerado para classes minúsculas, redistribui
            if n_test == 0 and n_train > 1:
                n_train -= 1
                n_test = 1
            if n_val == 0 and n_train > 1:
                n_train -= 1
                n_val = 1

        split_ids["train"].extend(class_ids[:n_train])
        split_ids["val"].extend(class_ids[n_train:n_train + n_val])
        split_ids["test"].extend(class_ids[n_train + n_val:])

    return split_ids


def find_source_file(src_root: Path, photo_id: int, subfolder: str, ext: str) -> Path | None:
    """
    Procura o arquivo de um photo_id em qualquer dos 3 splits do src.

    subfolder: "images", "labels", ou "labels_single_class"
    ext: ".jpg" ou ".txt"

    Retorna o primeiro caminho encontrado, ou None.
    """
    for split in SPLITS:
        candidate = src_root / split / subfolder / f"{photo_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_split_files(
    photo_ids: list[int],
    split: str,
    src_root: Path,
    dst_root: Path,
    dry_run: bool,
) -> dict:
    """
    Copia imagens + labels + labels_single_class de uma lista de IDs
    para a estrutura de destino.

    Retorna estatísticas (n_copied, n_failed, missing_files).
    """
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    dst_labels_sc = dst_root / split / "labels_single_class"

    if not dry_run:
        for d in (dst_images, dst_labels, dst_labels_sc):
            d.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    missing: list[dict] = []

    print(f"\n   Copiando split {split}: {len(photo_ids):,} IDs")
    print_every = max(500, len(photo_ids) // 20)

    for i, photo_id in enumerate(photo_ids, 1):
        # Encontra arquivos no source
        src_img = find_source_file(src_root, photo_id, "images", ".jpg")
        src_lbl = find_source_file(src_root, photo_id, "labels", ".txt")
        src_lbl_sc = find_source_file(src_root, photo_id, "labels_single_class", ".txt")

        local_missing = []
        if src_img is None:
            local_missing.append("image")
        if src_lbl is None:
            local_missing.append("label")
        if src_lbl_sc is None:
            local_missing.append("label_single_class")

        if local_missing:
            missing.append({"photo_id": photo_id, "missing": local_missing})
            continue

        if not dry_run:
            shutil.copy2(src_img, dst_images / f"{photo_id}.jpg")
            shutil.copy2(src_lbl, dst_labels / f"{photo_id}.txt")
            shutil.copy2(src_lbl_sc, dst_labels_sc / f"{photo_id}.txt")
        n_copied += 1

        if i % print_every == 0:
            print(f"     [{i:>6,}/{len(photo_ids):,}] copiados, faltam {len(photo_ids)-i:,}")

    print(f"   ✓ {split}: {n_copied:,} copiados, {len(missing)} faltando")

    return {
        "n_copied": n_copied,
        "n_missing": len(missing),
        "missing_files": missing[:10],  # só os primeiros 10 pra não inflar o relatório
    }


def write_data_yamls(dst_root: Path, dry_run: bool) -> None:
    """Gera data.yaml (10 classes) e data_single_class.yaml (1 classe)."""
    if dry_run:
        return

    # Multi-class
    yaml_multi = f"""# Auto-gerado por reconstruir_dataset_25k.py
path: {dst_root}
train: train/images
val: val/images
test: test/images
nc: {N_CLASSES}
names:
"""
    for name in CLASS_NAMES:
        yaml_multi += f"  - {name}\n"
    (dst_root / "data.yaml").write_text(yaml_multi)

    # Single-class (aponta para labels_single_class)
    yaml_sc = f"""# Auto-gerado por reconstruir_dataset_25k.py
# Single-class (todas as 10 classes colapsadas em 'embarcacao')
path: {dst_root}
train: train/images
val: val/images
test: test/images
nc: 1
names:
  - embarcacao
"""
    (dst_root / "data_single_class.yaml").write_text(yaml_sc)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstrói dataset_25k sem duplicações")
    parser.add_argument("--src", type=Path, default=SRC_ROOT, help="dataset_25k de entrada")
    parser.add_argument("--dst", type=Path, default=DST_ROOT, help="dataset_25k_v2 de saída")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed do split")
    parser.add_argument("--dry-run", action="store_true",
                        help="Só calcula split, não copia arquivos")
    args = parser.parse_args()

    print("=" * 72)
    print("  Reconstrução do dataset_25k → dataset_25k_v2")
    print("=" * 72)
    print(f"  Source:    {args.src}")
    print(f"  Dest:      {args.dst}")
    print(f"  Seed:      {args.seed}")
    print(f"  Dry-run:   {args.dry_run}")
    print(f"  Proporção: train={SPLIT_PROPORTIONS['train']:.0%}, "
          f"val={SPLIT_PROPORTIONS['val']:.0%}, test={SPLIT_PROPORTIONS['test']:.0%}")
    print("=" * 72)

    if not args.src.exists():
        print(f"ERRO: source {args.src} não existe")
        sys.exit(1)

    if args.dst.exists() and not args.dry_run:
        print(f"\n⚠ ATENÇÃO: {args.dst} já existe!")
        response = input("   Sobrescrever? [y/N]: ").strip().lower()
        if response != "y":
            print("   Abortado.")
            sys.exit(0)

    # ---- Passo 1: Coleta IDs e classes ----
    print("\n>> Coletando IDs únicos do dataset_25k atual...")
    id_to_class = collect_ids_with_classes(args.src)
    print(f"   Total de IDs únicos: {len(id_to_class):,}")

    # Conta por classe
    ids_by_class: dict[int, list[int]] = defaultdict(list)
    for photo_id, class_id in id_to_class.items():
        ids_by_class[class_id].append(photo_id)

    print(f"\n   Distribuição por classe:")
    print(f"   {'class_id':<10}{'name':<25}{'IDs':>8}")
    print(f"   {'-' * 45}")
    for class_id in sorted(ids_by_class.keys()):
        name = CLASS_NAMES[class_id] if class_id < N_CLASSES else f"class_{class_id}"
        n = len(ids_by_class[class_id])
        print(f"   {class_id:<10}{name:<25}{n:>8,}")

    # ---- Passo 2: Split estratificado ----
    print(f"\n>> Fazendo split estratificado (seed={args.seed})...")
    split_ids = stratified_split_per_class(ids_by_class, SPLIT_PROPORTIONS, args.seed)

    # Sanity: disjunção
    train_set = set(split_ids["train"])
    val_set = set(split_ids["val"])
    test_set = set(split_ids["test"])

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    print(f"\n   Tamanhos resultantes:")
    print(f"   train: {len(split_ids['train']):,} ({len(split_ids['train'])/len(id_to_class):.1%})")
    print(f"   val:   {len(split_ids['val']):,} ({len(split_ids['val'])/len(id_to_class):.1%})")
    print(f"   test:  {len(split_ids['test']):,} ({len(split_ids['test'])/len(id_to_class):.1%})")
    print(f"   total: {sum(len(v) for v in split_ids.values()):,}")

    print(f"\n   Disjunção entre splits (deve ser 0):")
    print(f"   train ∩ val:  {len(overlap_tv)}")
    print(f"   train ∩ test: {len(overlap_tt)}")
    print(f"   val ∩ test:   {len(overlap_vt)}")

    if overlap_tv or overlap_tt or overlap_vt:
        print("\n   ✗ ERRO: splits não são disjuntos. Bug no script de split!")
        sys.exit(1)
    else:
        print("   ✓ Splits são disjuntos")

    # ---- Passo 3: Distribuição por classe em cada split ----
    print(f"\n   Distribuição classe × split:")
    print(f"   {'class':<5}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
    print(f"   {'-' * 40}")
    class_counts_per_split = {split: defaultdict(int) for split in SPLITS}
    for split in SPLITS:
        for photo_id in split_ids[split]:
            class_counts_per_split[split][id_to_class[photo_id]] += 1
    for class_id in range(N_CLASSES):
        n_train = class_counts_per_split["train"][class_id]
        n_val = class_counts_per_split["val"][class_id]
        n_test = class_counts_per_split["test"][class_id]
        total = n_train + n_val + n_test
        print(f"   {class_id:<5}{n_train:>8,}{n_val:>8,}{n_test:>8,}{total:>8,}")

    # ---- Passo 4: Copia arquivos ----
    if args.dry_run:
        print("\n>> --dry-run: não copia arquivos")
        copy_stats = {split: None for split in SPLITS}
    else:
        print(f"\n>> Copiando arquivos para {args.dst}...")
        copy_stats = {}
        for split in SPLITS:
            copy_stats[split] = copy_split_files(
                split_ids[split], split, args.src, args.dst, dry_run=False
            )

    # ---- Passo 5: data.yaml ----
    if not args.dry_run:
        print(f"\n>> Escrevendo data.yaml e data_single_class.yaml...")
        write_data_yamls(args.dst, dry_run=False)
        print(f"   ✓ {args.dst / 'data.yaml'}")
        print(f"   ✓ {args.dst / 'data_single_class.yaml'}")

    # ---- Relatório final ----
    report = {
        "generated_at": datetime.now().isoformat(),
        "src": str(args.src),
        "dst": str(args.dst),
        "seed": args.seed,
        "proportions": SPLIT_PROPORTIONS,
        "n_unique_ids_input": len(id_to_class),
        "n_classes": N_CLASSES,
        "ids_per_class_input": {
            CLASS_NAMES[c]: len(ids_by_class[c])
            for c in sorted(ids_by_class.keys())
        },
        "split_sizes": {split: len(split_ids[split]) for split in SPLITS},
        "split_proportions_actual": {
            split: len(split_ids[split]) / len(id_to_class) for split in SPLITS
        },
        "disjoint_check": {
            "train_intersect_val": len(overlap_tv),
            "train_intersect_test": len(overlap_tt),
            "val_intersect_test": len(overlap_vt),
            "all_disjoint": len(overlap_tv) == len(overlap_tt) == len(overlap_vt) == 0,
        },
        "class_distribution_per_split": {
            split: {
                CLASS_NAMES[c]: class_counts_per_split[split][c]
                for c in range(N_CLASSES)
            }
            for split in SPLITS
        },
        "copy_stats": copy_stats,
    }

    if not args.dry_run:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n   Relatório: {REPORT_FILE}")

    print()
    print("=" * 72)
    if args.dry_run:
        print("  DRY-RUN COMPLETO — nada foi copiado")
    else:
        print("  RECONSTRUÇÃO COMPLETA")
        print(f"  Próximo passo: rodar recalcular_distribuicao_decis_v2.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
