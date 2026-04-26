"""
gerar_ids_rodada2.py

Sorteio complementar adaptativo para a segunda rodada do random_pool_v2.

Contexto:
  A primeira rodada (gerar_ids_aleatorios.py + baixar_random_pool_v2.py)
  baixou 49.422 arquivos com aparente 99,7% de sucesso, mas validação
  posterior revelou que 30,8% (15.236) eram HTMLs de erro disfarçados
  como JPEG — o shipspotting CDN retorna HTTP 200 + página HTML em vez
  de HTTP 404 para IDs inexistentes. Isso deixou todos os splits abaixo
  do alvo do dataset_25k:

    train: 19.617 / 22.064 (déficit 2.447, 11.1%)
    val:    8.362 /  9.147 (déficit   785,  8.6%)
    test:   6.207 /  6.898 (déficit   691, 10.0%)

  Surpresa: déficits NÃO concentrados no decil 9 (IDs altos) como
  previsto, mas espalhados por todos os decis com pico nos decis 0-3.
  Isso refuta a hipótese inicial de que IDs antigos teriam sido
  preservados no shipspotting.

PRÉ-REQUISITO: arquivo de IDs do dataset_25k em local file
  O dataset_25k está no Google Drive (não localmente), então este script
  não pode varrê-lo diretamente. Em vez disso, lê os IDs de um arquivo
  texto pré-extraído:

      ~/PROJETO_MARINHA/random_pool_v2/dataset_25k_ids.txt

  Para gerar esse arquivo, rode o seguinte snippet no Colab (onde o
  Drive está montado):

      from pathlib import Path
      candidates = [
          Path("/content/drive/MyDrive/InaTechShips/dataset_25k"),
          Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/dataset_25k"),
      ]
      root = next((c for c in candidates if c.exists()), None)
      assert root, "dataset_25k não encontrado"

      ids = set()
      for p in root.rglob("*.jpg"):
          try:
              ids.add(int(p.stem))
          except ValueError:
              pass

      out = Path("/content/drive/MyDrive/dataset_25k_ids.txt")
      with open(out, "w") as f:
          for pid in sorted(ids):
              f.write(f"{pid}\\n")
      print(f"{len(ids):,} IDs salvos em {out}")

  Depois, baixe o arquivo do Drive e salve em:
      ~/PROJETO_MARINHA/random_pool_v2/dataset_25k_ids.txt

O que este script faz:
  1. Lê os IDs do dataset_25k do arquivo local (pré-extraído).
  2. Lê o estado real atual: imagens válidas em random_pool_v2/{split}/images/
     + imagens corrompidas em random_pool_v2/_corrompidas/{split}/
     + originais sorteadas em ids_random_{split}.txt
  3. Lê os bins de decis do distribuicao_decis.json original.
  4. Para cada (split × decil), calcula:
     - déficit = alvo (do dataset_25k) - válidas atuais
     - taxa de sucesso observada = válidas / IDs originalmente sorteados
     - margem adaptativa = (1 / taxa_sucesso) × fator_seguranca
     - n_sortear = max(0, déficit × margem_adaptativa)
  5. Sorteia novos IDs em cada (split × decil) excluindo:
     - IDs já sorteados na rodada 1 (válidos OU corrompidos)
     - IDs do dataset_25k (subset curado, deve permanecer disjunto)
     - IDs já sorteados em outros splits da rodada 2
  6. Salva listas em ids_random_rodada2_{split}.txt
  7. Gera rodada2_sampling_report.json com justificativa de cada cela

Reprodutibilidade:
  Seed 4242 (não 42 como na rodada 1, para evitar resamples idênticos).

Uso:
  python gerar_ids_rodada2.py --dry-run                # conferir números
  python gerar_ids_rodada2.py                          # escreve os arquivos
  python gerar_ids_rodada2.py --margin-safety 1.5      # fator de segurança
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = Path.home() / "PROJETO_MARINHA" / "random_pool_v2"
QUARANTINE_DIR = BASE_DIR / "_corrompidas"
DECIS_JSON = BASE_DIR / "distribuicao_decis.json"

# Lista de IDs do dataset_25k (pré-extraída no Colab e baixada para o disco local)
# Veja instruções no final do docstring.
DATASET_25K_IDS_FILE = BASE_DIR / "dataset_25k_ids.txt"

REPORT_FILE = BASE_DIR / "rodada2_sampling_report.json"
SEED = 4242
SPLITS = ("train", "val", "test")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def read_ids_from_txt(path: Path) -> set[int]:
    """Lê um arquivo .txt com um ID por linha e retorna set."""
    if not path.exists():
        return set()
    out: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(int(line))
                except ValueError:
                    pass
    return out


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


def find_dataset_25k_ids() -> set[int]:
    """
    Lê os IDs do dataset_25k a partir do arquivo local DATASET_25K_IDS_FILE.
    Este arquivo deve ter sido previamente extraído no Colab (onde o Drive
    está montado) e baixado para ~/PROJETO_MARINHA/random_pool_v2/.

    Retorna set vazio + aviso se o arquivo não existir.
    """
    if not DATASET_25K_IDS_FILE.exists():
        return set()
    return read_ids_from_txt(DATASET_25K_IDS_FILE)


def assign_to_bin(photo_id: int, bins: list[tuple[int, int]]) -> int:
    """Retorna o índice do decil ao qual o ID pertence, ou -1."""
    for idx, (low, high) in enumerate(bins):
        if low <= photo_id <= high:
            return idx
    return -1


# ═══════════════════════════════════════════════════════════════════
# Núcleo do sorteio adaptativo
# ═══════════════════════════════════════════════════════════════════

def compute_deficit_per_cell(
    splits_data: dict,
    bins: list[tuple[int, int]],
    targets_per_split: dict,
) -> dict:
    """
    Para cada (split × decil), calcula:
      - n_target: alvo do dataset_25k naquela cela
      - n_valid: imagens válidas atuais naquela cela
      - n_originally_sampled: IDs originalmente sorteados (rodada 1)
      - n_corrupt: imagens corrompidas naquela cela
      - deficit: max(0, n_target - n_valid)
      - success_rate: n_valid / max(1, n_originally_sampled)
    """
    result = {}
    for split in SPLITS:
        valid_ids = splits_data[split]["valid"]
        original_ids = splits_data[split]["original"]
        corrupt_ids = splits_data[split]["corrupt"]
        targets = targets_per_split[split]

        # Conta por decil
        valid_per_decil: dict[int, int] = defaultdict(int)
        original_per_decil: dict[int, int] = defaultdict(int)
        corrupt_per_decil: dict[int, int] = defaultdict(int)

        for pid in valid_ids:
            valid_per_decil[assign_to_bin(pid, bins)] += 1
        for pid in original_ids:
            original_per_decil[assign_to_bin(pid, bins)] += 1
        for pid in corrupt_ids:
            corrupt_per_decil[assign_to_bin(pid, bins)] += 1

        cells = []
        for decil_idx in range(len(bins)):
            n_target = targets[decil_idx]
            n_valid = valid_per_decil.get(decil_idx, 0)
            n_orig = original_per_decil.get(decil_idx, 0)
            n_corrupt = corrupt_per_decil.get(decil_idx, 0)
            deficit = max(0, n_target - n_valid)
            success_rate = n_valid / n_orig if n_orig > 0 else 0.0
            cells.append({
                "decil": decil_idx,
                "bin_low": bins[decil_idx][0],
                "bin_high": bins[decil_idx][1],
                "n_target": n_target,
                "n_valid": n_valid,
                "n_originally_sampled": n_orig,
                "n_corrupt": n_corrupt,
                "deficit": deficit,
                "success_rate": success_rate,
            })
        result[split] = cells
    return result


def compute_n_to_sample(
    cells: list[dict],
    margin_safety: float,
) -> list[dict]:
    """
    Para cada célula com déficit, calcula n_to_sample com margem
    adaptativa baseada na taxa de sucesso observada.

    margem = max(1.2, (1 / success_rate) × margin_safety)
    n_sortear = ceil(deficit × margem)
    """
    import math
    out = []
    for cell in cells:
        if cell["deficit"] == 0:
            cell["margin"] = None
            cell["n_to_sample"] = 0
        else:
            sr = cell["success_rate"]
            if sr <= 0:
                margin = 3.0  # último recurso, ninguém sabe a taxa
            else:
                margin = max(1.2, (1.0 / sr) * margin_safety)
            cell["margin"] = round(margin, 3)
            cell["n_to_sample"] = math.ceil(cell["deficit"] * margin)
        out.append(cell)
    return out


def sample_new_ids_for_cell(
    decil_idx: int,
    bin_low: int,
    bin_high: int,
    n_to_sample: int,
    excluded: set[int],
    rng: random.Random,
    max_attempts_factor: int = 50,
) -> tuple[list[int], int]:
    """
    Sorteia n_to_sample IDs novos no range [bin_low, bin_high], excluindo
    qualquer ID já no set excluded. Retorna (lista_de_ids, n_rejeitados).
    """
    sampled: list[int] = []
    rejected = 0
    range_size = bin_high - bin_low + 1
    max_attempts = n_to_sample * max_attempts_factor

    if range_size < n_to_sample * 2:
        # Range muito apertado — itera o range inteiro
        candidates = [i for i in range(bin_low, bin_high + 1) if i not in excluded]
        rng.shuffle(candidates)
        sampled = candidates[:n_to_sample]
        for i in candidates[n_to_sample:]:
            rejected += 1
        return (sampled, rejected)

    attempts = 0
    seen_in_this_call: set[int] = set()
    while len(sampled) < n_to_sample and attempts < max_attempts:
        attempts += 1
        candidate = rng.randint(bin_low, bin_high)
        if candidate in excluded or candidate in seen_in_this_call:
            rejected += 1
            continue
        seen_in_this_call.add(candidate)
        sampled.append(candidate)

    return (sampled, rejected)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Sorteio rodada 2 random_pool_v2")
    parser.add_argument(
        "--margin-safety", type=float, default=1.3,
        help="Fator de segurança aplicado sobre 1/taxa_sucesso (default: 1.3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Não escreve arquivos, só calcula e imprime relatório",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  Rodada 2 — sorteio complementar adaptativo")
    print("=" * 72)
    print(f"  Base:           {BASE_DIR}")
    print(f"  Quarentena:     {QUARANTINE_DIR}")
    print(f"  Decis JSON:     {DECIS_JSON}")
    print(f"  Margin safety:  {args.margin_safety}")
    print(f"  Seed:           {SEED}")
    print(f"  Dry-run:        {args.dry_run}")
    print("=" * 72)

    # ---- Carrega bins e targets ----
    if not DECIS_JSON.exists():
        print(f"ERRO: {DECIS_JSON} não existe")
        sys.exit(1)

    with open(DECIS_JSON) as f:
        dd = json.load(f)
    bins = [(b["low"], b["high"]) for b in dd["bins"]]
    targets_per_split = dd["curated_counts_per_split"]

    # ---- Carrega IDs do dataset_25k (de arquivo local pré-extraído) ----
    print("\n>> Carregando IDs do dataset_25k para o set de exclusão...")
    d25_ids = find_dataset_25k_ids()
    if not d25_ids:
        print(f"   ⚠ AVISO CRÍTICO: arquivo {DATASET_25K_IDS_FILE} não existe ou está vazio.")
        print(f"   Você deve primeiro extrair os IDs no Colab (onde o Drive está montado)")
        print(f"   e baixar o arquivo para esse caminho. Veja as instruções.")
        print(f"   PROSSEGUINDO SEM EXCLUIR IDs DO DATASET_25K!")
        print(f"   Isso pode causar overlap entre random_pool_v2 e dataset_25k.")
        response = input("\n   Prosseguir mesmo assim? [y/N]: ").strip().lower()
        if response != "y":
            print("   Abortado.")
            sys.exit(1)
    else:
        print(f"   ✓ {len(d25_ids):,} IDs do dataset_25k lidos de {DATASET_25K_IDS_FILE}")
        # Sanity check: deveria ser ~38.109
        if not (37000 <= len(d25_ids) <= 39000):
            print(f"   ⚠ Contagem fora do esperado (~38.109). Verifique se o arquivo está correto.")

    # ---- Coleta estado atual por split ----
    print("\n>> Coletando estado atual...")
    splits_data = {}
    for split in SPLITS:
        valid = list_jpg_ids(BASE_DIR / split / "images")
        corrupt = list_jpg_ids(QUARANTINE_DIR / split)
        original = read_ids_from_txt(BASE_DIR / f"ids_random_{split}.txt")
        splits_data[split] = {
            "valid": valid,
            "corrupt": corrupt,
            "original": original,
        }
        print(f"   {split}: válidas={len(valid):,}, corrompidas={len(corrupt):,}, "
              f"originais sorteadas={len(original):,}")

    # ---- Calcula déficits e taxas de sucesso por cela ----
    print("\n>> Calculando déficits por (split × decil)...")
    cells_per_split = compute_deficit_per_cell(splits_data, bins, targets_per_split)
    for split in SPLITS:
        cells_per_split[split] = compute_n_to_sample(
            cells_per_split[split], args.margin_safety
        )

    # Impressão da tabela
    for split in SPLITS:
        cells = cells_per_split[split]
        total_deficit = sum(c["deficit"] for c in cells)
        total_sample = sum(c["n_to_sample"] for c in cells)
        print(f"\n   {split} — déficit total: {total_deficit:,}, "
              f"a sortear: {total_sample:,}")
        print(f"   {'decil':<7}{'alvo':<8}{'válidas':<10}{'sorteados1':<12}"
              f"{'sucesso':<10}{'déficit':<10}{'margem':<9}{'sortear2':<10}")
        for c in cells:
            sr_pct = f"{c['success_rate']*100:.1f}%"
            margin_str = f"{c['margin']:.2f}" if c['margin'] else "-"
            print(
                f"   {c['decil']:<7}{c['n_target']:<8}{c['n_valid']:<10}"
                f"{c['n_originally_sampled']:<12}{sr_pct:<10}"
                f"{c['deficit']:<10}{margin_str:<9}{c['n_to_sample']:<10}"
            )

    # ---- Sorteio ----
    print("\n>> Sorteando IDs novos por cela...")

    # Set de exclusão GLOBAL: dataset_25k + originais de todas as rodadas + corrompidas
    excluded_global: set[int] = set(d25_ids)
    for split in SPLITS:
        excluded_global |= splits_data[split]["original"]
        excluded_global |= splits_data[split]["valid"]
        excluded_global |= splits_data[split]["corrupt"]

    print(f"   Set de exclusão global: {len(excluded_global):,} IDs "
          f"(dataset_25k + originais + válidas + corrompidas)")

    rng = random.Random(SEED)
    new_ids_per_split: dict[str, list[int]] = {s: [] for s in SPLITS}
    sampling_log = {s: [] for s in SPLITS}

    # Set acumulativo entre splits (para garantir disjunção)
    excluded_running = set(excluded_global)

    for split in SPLITS:
        cells = cells_per_split[split]
        for cell in cells:
            n_target = cell["n_to_sample"]
            if n_target == 0:
                cell["sampled"] = 0
                cell["rejected_attempts"] = 0
                sampling_log[split].append(cell)
                continue

            sampled, rejected = sample_new_ids_for_cell(
                decil_idx=cell["decil"],
                bin_low=cell["bin_low"],
                bin_high=cell["bin_high"],
                n_to_sample=n_target,
                excluded=excluded_running,
                rng=rng,
            )
            new_ids_per_split[split].extend(sampled)
            excluded_running.update(sampled)
            cell["sampled"] = len(sampled)
            cell["rejected_attempts"] = rejected
            if len(sampled) < n_target:
                print(f"   ⚠ {split} decil {cell['decil']}: pediu {n_target}, "
                      f"conseguiu só {len(sampled)} (range esgotado?)")
            sampling_log[split].append(cell)

    # ---- Resumo do sorteio ----
    print("\n>> Sorteio concluído:")
    print(f"   {'split':<8}{'sortear':<12}{'sorteados':<12}")
    for split in SPLITS:
        target = sum(c["n_to_sample"] for c in cells_per_split[split])
        actual = len(new_ids_per_split[split])
        print(f"   {split:<8}{target:<12}{actual:<12}")

    total_new = sum(len(ids) for ids in new_ids_per_split.values())
    print(f"   TOTAL: {total_new:,} novos IDs a baixar na rodada 2")

    # ---- Escreve arquivos ----
    if args.dry_run:
        print("\n>> --dry-run: nenhum arquivo será escrito")
    else:
        print("\n>> Escrevendo arquivos...")
        for split in SPLITS:
            ids = sorted(new_ids_per_split[split])
            path = BASE_DIR / f"ids_random_rodada2_{split}.txt"
            with open(path, "w") as f:
                for pid in ids:
                    f.write(f"{pid}\n")
            print(f"   ✓ {path} ({len(ids):,} IDs)")

    # ---- Relatório ----
    report = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "margin_safety": args.margin_safety,
        "dataset_25k_ids_file": str(DATASET_25K_IDS_FILE),
        "n_dataset_25k_excluded": len(d25_ids),
        "summary_per_split": {
            split: {
                "n_valid_before": len(splits_data[split]["valid"]),
                "n_corrupt_before": len(splits_data[split]["corrupt"]),
                "n_original_sampled_round1": len(splits_data[split]["original"]),
                "n_target_dataset25k": sum(targets_per_split[split]),
                "n_total_deficit": sum(c["deficit"] for c in cells_per_split[split]),
                "n_to_sample_round2": sum(c["n_to_sample"] for c in cells_per_split[split]),
                "n_actually_sampled_round2": len(new_ids_per_split[split]),
            }
            for split in SPLITS
        },
        "cells_detail": sampling_log,
    }
    if not args.dry_run:
        REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n   Relatório: {REPORT_FILE}")

    print()
    print("=" * 72)
    print("Próximo passo: rodar baixar_random_pool_v2.py com --ids-suffix rodada2")
    print("=" * 72)


if __name__ == "__main__":
    main()
