"""
recalcular_distribuicao_decis_v2.py

Recalcula a distribuição por decis empíricos para o dataset_25k_v2
(reconstruído sem duplicações entre splits).

CONTEXTO

  O distribuicao_decis.json original foi calculado com base nos splits
  contaminados do dataset_25k antigo (38.109 arquivos com 27.796 únicos).
  Os bins empíricos podem mudar quando recalculados sobre os 27.796
  únicos com novo split 60/20/20, e os curated_counts_per_split mudam
  necessariamente.

  Este script gera o JSON novo, que vai ser usado pelo
  downsample_random_pool_v2.py para calcular quantas imagens manter
  por (split × decil) no random_pool_v2.

O QUE ESTE SCRIPT FAZ

  1. Lê todos os IDs de cada split do dataset_25k_v2.
  2. Une os IDs e calcula 10 bins por decil empírico (mesma lógica do
     gerar_ids_aleatorios.py original).
  3. Para cada split, conta quantos IDs do dataset_25k_v2 estão em cada
     decil (curated_counts_per_split).
  4. Salva tudo em distribuicao_decis_v2.json no Drive.

PRÉ-REQUISITOS

  - Roda no Colab (Drive montado) ou em qualquer lugar com acesso ao
    dataset_25k_v2.
  - dataset_25k_v2/{train,val,test}/images/ deve existir.

USO

  python recalcular_distribuicao_decis_v2.py
  python recalcular_distribuicao_decis_v2.py --src /caminho/diferente

SAÍDA

  distribuicao_decis_v2.json no mesmo diretório do random_pool_v2:
  /content/drive/MyDrive/InaTechShips/random_pool_v2/distribuicao_decis_v2.json

REPRODUTIBILIDADE

  Determinístico: mesmo dataset_25k_v2 → mesmos bins e contagens.

TEMPO ESPERADO

  ~30 segundos no Drive (apenas lê nomes de arquivo, não conteúdo).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

DATASET_V2_ROOT = Path("/content/drive/MyDrive/InaTechShips/dataset_25k_v2")
RANDOM_POOL_ROOT = Path("/content/drive/MyDrive/InaTechShips/random_pool_v2")
OUTPUT_FILE = RANDOM_POOL_ROOT / "distribuicao_decis_v2.json"

NUM_BINS = 10
SPLITS = ("train", "val", "test")


def collect_ids_per_split(root: Path) -> dict[str, list[int]]:
    """Coleta IDs de imagens em cada split."""
    ids_per_split: dict[str, list[int]] = {}
    for split in SPLITS:
        images_dir = root / split / "images"
        if not images_dir.exists():
            print(f"   ⚠ {images_dir} não existe")
            ids_per_split[split] = []
            continue
        ids = []
        for f in images_dir.iterdir():
            if f.suffix.lower() == ".jpg":
                try:
                    ids.append(int(f.stem))
                except ValueError:
                    pass
        ids_per_split[split] = ids
    return ids_per_split


def compute_decile_bins(all_ids_sorted: list[int], num_bins: int) -> list[tuple[int, int]]:
    """
    Calcula bins por decil empírico: cada bin contém aproximadamente
    1/num_bins dos IDs ordenados.

    Retorna lista de (low, high) inclusivos por bin.
    """
    n = len(all_ids_sorted)
    bins = []
    for i in range(num_bins):
        start_idx = (i * n) // num_bins
        end_idx = ((i + 1) * n) // num_bins - 1 if i < num_bins - 1 else n - 1
        low = all_ids_sorted[start_idx]
        high = all_ids_sorted[end_idx]
        bins.append((low, high))
    return bins


def assign_to_bin(photo_id: int, bins: list[tuple[int, int]]) -> int:
    """Retorna o índice do bin ao qual o ID pertence, ou -1."""
    for idx, (low, high) in enumerate(bins):
        if low <= photo_id <= high:
            return idx
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalcula decis para dataset_25k_v2")
    parser.add_argument("--src", type=Path, default=DATASET_V2_ROOT,
                        help="dataset_25k_v2 (default: caminho padrão no Drive)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Caminho do JSON de saída")
    parser.add_argument("--num-bins", type=int, default=NUM_BINS, help="Número de decis")
    args = parser.parse_args()

    print("=" * 72)
    print("  Recalcular distribuição por decis (dataset_25k_v2)")
    print("=" * 72)
    print(f"  Source:  {args.src}")
    print(f"  Output:  {args.output}")
    print(f"  Bins:    {args.num_bins}")
    print("=" * 72)

    if not args.src.exists():
        print(f"\nERRO: {args.src} não existe")
        print("Rode reconstruir_dataset_25k.py primeiro.")
        sys.exit(1)

    # ---- Coleta IDs ----
    print("\n>> Coletando IDs por split...")
    ids_per_split = collect_ids_per_split(args.src)
    for split in SPLITS:
        print(f"   {split}: {len(ids_per_split[split]):,} IDs")

    all_ids = sorted(set().union(*[set(ids) for ids in ids_per_split.values()]))
    n_total = len(all_ids)
    n_sum = sum(len(v) for v in ids_per_split.values())
    print(f"   total único: {n_total:,}  |  soma splits: {n_sum:,}")

    if n_total != n_sum:
        print(f"\n   ⚠ AVISO: total único != soma — splits NÃO são disjuntos!")
        print(f"   Diferença: {n_sum - n_total} IDs duplicados")
        print(f"   Rode o reconstruir_dataset_25k.py se isso não for esperado.")
    else:
        print(f"   ✓ Splits são disjuntos")

    # ---- Calcula bins ----
    print(f"\n>> Calculando {args.num_bins} bins por decil empírico...")
    bins = compute_decile_bins(all_ids, args.num_bins)

    print(f"\n   {'idx':<5}{'low':>10}{'high':>12}{'range_size':>14}")
    print(f"   {'-' * 41}")
    for i, (low, high) in enumerate(bins):
        print(f"   {i:<5}{low:>10,}{high:>12,}{(high - low + 1):>14,}")

    # ---- Conta por (split × decil) ----
    print(f"\n>> Contando IDs por (split × decil)...")
    counts: dict[str, list[int]] = {split: [0] * args.num_bins for split in SPLITS}
    for split in SPLITS:
        for photo_id in ids_per_split[split]:
            bin_idx = assign_to_bin(photo_id, bins)
            if bin_idx >= 0:
                counts[split][bin_idx] += 1

    print(f"\n   {'decil':<6}{'train':>9}{'val':>9}{'test':>9}{'total':>9}")
    print(f"   {'-' * 42}")
    for i in range(args.num_bins):
        n_train = counts["train"][i]
        n_val = counts["val"][i]
        n_test = counts["test"][i]
        total = n_train + n_val + n_test
        print(f"   {i:<6}{n_train:>9,}{n_val:>9,}{n_test:>9,}{total:>9,}")

    # ---- Salva ----
    output = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "DATASET_V2_ROOT": str(args.src),
            "NUM_BINS": args.num_bins,
        },
        "n_total_ids": n_total,
        "bins": [
            {"idx": i, "low": low, "high": high, "range_size": high - low + 1}
            for i, (low, high) in enumerate(bins)
        ],
        "curated_counts_per_split": counts,
        "split_sizes": {split: len(ids_per_split[split]) for split in SPLITS},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n   ✓ Salvo em: {args.output}")
    print()
    print("=" * 72)
    print("  Próximo passo: rodar downsample_random_pool_v2.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
