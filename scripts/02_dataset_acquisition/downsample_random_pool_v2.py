"""
downsample_random_pool_v2.py

Reduz o random_pool_v2 para igualar exatamente os tamanhos por
(split × decil) do dataset_25k_v2 reconstruído.

CONTEXTO

  O random_pool_v2 atual tem 39.628 imagens válidas:
    train: 22.899
    val:    9.515
    test:   7.214

  O dataset_25k_v2 reconstruído (sem duplicações) terá ~27.796 imagens
  totais distribuídas em ~16.677 / 5.557 / 5.566 (60/20/20).

  Para o experimento principal, o random_pool_v2 deve ter:
    - Mesmo tamanho que dataset_25k_v2 por split (controle experimental)
    - Estratificação por decis empíricos espelhando o dataset_25k_v2
      (mantém o pareamento metodológico do design original)

  Este script remove o excedente preservando essa estratificação.

O QUE ESTE SCRIPT FAZ

  1. Lê distribuicao_decis_v2.json (gerado pelo
     recalcular_distribuicao_decis_v2.py).
  2. Para cada split do random_pool_v2, agrupa as imagens por decil.
  3. Para cada (split × decil), seleciona aleatoriamente (seed=42) o
     número exato de IDs que o dataset_25k_v2 tem naquele decil.
  4. Move as imagens excedentes para random_pool_v2/_excedente/{split}/
     (preserva, não deleta).
  5. Gera relatório JSON com tudo o que foi feito.

PRÉ-REQUISITOS

  - dataset_25k_v2 reconstruído.
  - distribuicao_decis_v2.json gerado.
  - random_pool_v2/{train,val,test}/images/ com 39.628 imagens válidas.

ROTA DE EXECUÇÃO RECOMENDADA

  Roda no Colab (Drive montado), apontando para o random_pool_v2 do
  Drive em /content/drive/MyDrive/InaTechShips/random_pool_v2/. Evita
  re-upload de ~17GB depois.

  ATENÇÃO: operação shutil.move no Drive é lenta (~1-2 arquivos/s).
  Tempo estimado: ~30-45 min para mover ~11.832 arquivos excedentes.

  Alternativa local mais rápida: rodar com --src apontando para
  ~/PROJETO_MARINHA/random_pool_v2/ (1-2 minutos), mas requer re-sync
  para o Drive depois.

USO

  python downsample_random_pool_v2.py
  python downsample_random_pool_v2.py --dry-run         # só calcula
  python downsample_random_pool_v2.py --src /caminho   # outro local

REPRODUTIBILIDADE

  Seed 42. Para o mesmo random_pool_v2 + distribuicao_decis_v2.json,
  sempre seleciona as mesmas imagens.

REVERSIBILIDADE

  Imagens excedentes são MOVIDAS para _excedente/, não deletadas. Para
  reverter: mover todos os arquivos de _excedente/{split}/ de volta
  para {split}/images/.
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

RANDOM_POOL_ROOT = Path("/content/drive/MyDrive/InaTechShips/random_pool_v2")
DECIS_V2_FILE = RANDOM_POOL_ROOT / "distribuicao_decis_v2.json"
EXCEDENTE_DIR = RANDOM_POOL_ROOT / "_excedente"
REPORT_FILE = RANDOM_POOL_ROOT / "downsample_report.json"

SEED = 42
SPLITS = ("train", "val", "test")


def list_jpg_ids(directory: Path) -> set[int]:
    """Lista IDs de arquivos .jpg num diretório."""
    if not directory.exists():
        return set()
    out: set[int] = set()
    for p in directory.iterdir():
        if p.suffix.lower() == ".jpg":
            try:
                out.add(int(p.stem))
            except ValueError:
                pass
    return out


def assign_to_bin(photo_id: int, bins: list[tuple[int, int]]) -> int:
    """Retorna o índice do bin ao qual o ID pertence, ou -1."""
    for idx, (low, high) in enumerate(bins):
        if low <= photo_id <= high:
            return idx
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample random_pool_v2 para v2")
    parser.add_argument("--src", type=Path, default=RANDOM_POOL_ROOT,
                        help="random_pool_v2 root (default: caminho padrão no Drive)")
    parser.add_argument("--decis-file", type=Path, default=DECIS_V2_FILE,
                        help="distribuicao_decis_v2.json")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed da seleção")
    parser.add_argument("--dry-run", action="store_true",
                        help="Só calcula, não move arquivos")
    args = parser.parse_args()

    # Ajusta paths derivados se --src foi customizado
    excedente_dir = args.src / "_excedente"
    report_file = args.src / "downsample_report.json"

    print("=" * 72)
    print("  Downsample random_pool_v2 → tamanhos do dataset_25k_v2")
    print("=" * 72)
    print(f"  Source:        {args.src}")
    print(f"  Decis JSON:    {args.decis_file}")
    print(f"  Excedente:     {excedente_dir}")
    print(f"  Seed:          {args.seed}")
    print(f"  Dry-run:       {args.dry_run}")
    print("=" * 72)

    # ---- Carrega decis v2 ----
    if not args.decis_file.exists():
        print(f"\nERRO: {args.decis_file} não existe")
        print("Rode recalcular_distribuicao_decis_v2.py primeiro.")
        sys.exit(1)

    with open(args.decis_file) as f:
        decis = json.load(f)

    bins = [(b["low"], b["high"]) for b in decis["bins"]]
    targets_per_split = decis["curated_counts_per_split"]
    n_bins = len(bins)

    print(f"\n>> Decis v2 carregados ({n_bins} bins)")
    print(f"   Alvos do dataset_25k_v2:")
    for split in SPLITS:
        n_target = sum(targets_per_split[split])
        print(f"     {split}: {n_target:,}")

    # ---- Coleta estado atual do random_pool_v2 ----
    print(f"\n>> Coletando IDs atuais do random_pool_v2...")
    current_ids: dict[str, set[int]] = {}
    for split in SPLITS:
        images_dir = args.src / split / "images"
        ids = list_jpg_ids(images_dir)
        current_ids[split] = ids
        print(f"   {split}: {len(ids):,} imagens")

    # ---- Para cada (split × decil), seleciona aleatoriamente ----
    print(f"\n>> Calculando seleção por (split × decil)...")
    rng = random.Random(args.seed)

    keep_per_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    move_per_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    cells_log: dict[str, list[dict]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        # Agrupa IDs do split por decil
        ids_per_decil: dict[int, list[int]] = defaultdict(list)
        for pid in current_ids[split]:
            decil = assign_to_bin(pid, bins)
            if decil >= 0:
                ids_per_decil[decil].append(pid)

        print(f"\n   {split}:")
        print(f"     {'decil':<7}{'atual':>8}{'alvo':>8}{'manter':>8}{'mover':>8}")

        for decil in range(n_bins):
            available = sorted(ids_per_decil.get(decil, []))
            n_target = targets_per_split[split][decil]
            n_available = len(available)

            if n_available <= n_target:
                # Sem excedente neste decil — mantém tudo
                kept = available
                moved = []
                if n_available < n_target:
                    print(f"     {decil:<7}{n_available:>8,}{n_target:>8,}"
                          f"{n_available:>8,}{0:>8,} ⚠ déficit {n_target - n_available}")
                else:
                    print(f"     {decil:<7}{n_available:>8,}{n_target:>8,}"
                          f"{n_available:>8,}{0:>8,}")
            else:
                # Há excedente — seleciona aleatoriamente n_target para manter
                shuffled = list(available)
                rng.shuffle(shuffled)
                kept = sorted(shuffled[:n_target])
                moved = sorted(shuffled[n_target:])
                print(f"     {decil:<7}{n_available:>8,}{n_target:>8,}"
                      f"{n_target:>8,}{len(moved):>8,}")

            keep_per_split[split].update(kept)
            move_per_split[split].update(moved)
            cells_log[split].append({
                "decil": decil,
                "n_available": n_available,
                "n_target": n_target,
                "n_kept": len(kept),
                "n_moved": len(moved),
                "deficit": max(0, n_target - n_available),
            })

    # ---- Resumo do plano ----
    print(f"\n>> Plano de downsample:")
    print(f"   {'split':<8}{'atual':>8}{'manter':>8}{'mover':>8}{'déficit':>9}")
    for split in SPLITS:
        n_atual = len(current_ids[split])
        n_keep = len(keep_per_split[split])
        n_move = len(move_per_split[split])
        n_target_total = sum(targets_per_split[split])
        deficit = max(0, n_target_total - n_keep)
        marker = " ✓" if deficit == 0 else f" ⚠"
        print(f"   {split:<8}{n_atual:>8,}{n_keep:>8,}{n_move:>8,}{deficit:>9,}{marker}")

    total_to_move = sum(len(v) for v in move_per_split.values())
    print(f"\n   Total de imagens a mover: {total_to_move:,}")

    # ---- Movimentação ----
    if args.dry_run:
        print(f"\n>> --dry-run: nada será movido")
        move_stats = {split: {"n_moved": 0, "n_failed": 0} for split in SPLITS}
    else:
        print(f"\n>> Movendo arquivos excedentes para {excedente_dir}/...")
        move_stats = {}

        for split in SPLITS:
            split_excedente = excedente_dir / split
            split_excedente.mkdir(parents=True, exist_ok=True)

            n_moved = 0
            n_failed = 0
            print_every = max(100, len(move_per_split[split]) // 20)

            ids_to_move = sorted(move_per_split[split])
            for i, photo_id in enumerate(ids_to_move, 1):
                src_file = args.src / split / "images" / f"{photo_id}.jpg"
                dst_file = split_excedente / f"{photo_id}.jpg"
                try:
                    if src_file.exists():
                        shutil.move(str(src_file), str(dst_file))
                        n_moved += 1
                    else:
                        n_failed += 1
                except OSError as exc:
                    print(f"     ⚠ Falha movendo {photo_id}.jpg: {exc}")
                    n_failed += 1

                if i % print_every == 0:
                    print(f"     [{split}] {i:>6,}/{len(ids_to_move):,} movidos")

            move_stats[split] = {"n_moved": n_moved, "n_failed": n_failed}
            print(f"   ✓ {split}: {n_moved:,} movidos, {n_failed} falhas")

    # ---- Verificação pós-movimento ----
    if not args.dry_run:
        print(f"\n>> Verificando estado pós-movimento...")
        for split in SPLITS:
            n_now = len(list_jpg_ids(args.src / split / "images"))
            n_target = sum(targets_per_split[split])
            n_excedente = len(list_jpg_ids(excedente_dir / split))
            marker = "✓" if n_now == n_target else "⚠"
            print(f"   {split}: {n_now:,} (alvo {n_target:,}) "
                  f"+ {n_excedente:,} no _excedente {marker}")

    # ---- Relatório ----
    report = {
        "generated_at": datetime.now().isoformat(),
        "src": str(args.src),
        "decis_file": str(args.decis_file),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "current_state_before": {
            split: len(current_ids[split]) for split in SPLITS
        },
        "targets_dataset_25k_v2": {
            split: sum(targets_per_split[split]) for split in SPLITS
        },
        "kept_after_downsample": {
            split: len(keep_per_split[split]) for split in SPLITS
        },
        "moved_to_excedente": {
            split: len(move_per_split[split]) for split in SPLITS
        },
        "move_stats": move_stats,
        "cells_detail": cells_log,
    }

    if not args.dry_run:
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n   Relatório: {report_file}")

    print()
    print("=" * 72)
    if args.dry_run:
        print("  DRY-RUN COMPLETO — nenhum arquivo movido")
    else:
        print("  DOWNSAMPLE COMPLETO")
        print(f"  random_pool_v2 agora tem mesmos tamanhos que dataset_25k_v2.")
        print(f"  Excedentes preservados em {excedente_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
