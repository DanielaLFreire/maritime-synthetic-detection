"""
gerar_labels_single_class.py

Gera pastas labels_single_class/ paralelas aos labels existentes nos
datasets que vão ser usados nos treinos do experimento.

Datasets processados:

  1. CITRA-3D-Real (domínio operacional alvo)
     - Lê de:    {split}/labels_cleaned/  (após limpeza + quarentena)
     - Escreve:  {split}/labels_single_class/
     - 9 classes originais (0-8) → todas viram 0

  2. dataset_25k (subset curado por CLIP do InaTechShips)
     - Lê de:    {split}/labels/  (labels reanotados pela autora)
     - Escreve:  {split}/labels_single_class/
     - 10 classes originais (0-9) → todas viram 0

Operação:

  - Para cada arquivo .txt da pasta de origem, parseia as linhas válidas,
    troca o índice de classe por 0, e escreve no arquivo de destino com
    as mesmas coordenadas e formato consistente (precisão fixa).
  - Linhas inválidas no CITRA-3D-Real não devem existir (já passaram pela
    limpeza), mas o parser é defensivo e descarta qualquer lixo encontrado.
  - Arquivos vazios são preservados como vazios (não devem existir nesses
    datasets, mas o script lida com o caso por segurança).
  - Operação não-destrutiva: as pastas labels/ e labels_cleaned/ originais
    não são tocadas.

Paralelização:

  Threads para I/O do Drive (gargalo é latência de leitura/escrita,
  não CPU). 48 threads por dataset.

Saída:

  - Pastas labels_single_class/ em cada split de cada dataset
  - Relatório consolidado: single_class_generation_report.txt + .json
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Dataset 1: CITRA-3D-Real (lê de labels_cleaned/)
CITRA_ROOT = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real")
CITRA_LABELS_SOURCE_SUBDIR = "labels_cleaned"

# Dataset 2: dataset_25k (lê de labels/)
INATECH_ROOT = Path("/content/drive/MyDrive/InaTechShips/dataset_25k")
INATECH_LABELS_SOURCE_SUBDIR = "labels"

# Subdiretório de saída em ambos os datasets
SINGLE_CLASS_SUBDIR = "labels_single_class"

# Onde salvar o relatório
REPORT_DIR = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets")
REPORT_TXT = REPORT_DIR / "single_class_generation_report.txt"
REPORT_JSON = REPORT_DIR / "single_class_generation_report.json"

SPLITS = ("train", "val", "test")
MAX_WORKERS = 48
PROGRESS_EVERY = 2000


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------

@dataclass
class SplitGenStats:
    files_input: int = 0
    files_written: int = 0
    files_empty: int = 0
    bboxes_total: int = 0
    bboxes_invalid_skipped: int = 0
    original_class_counts: Counter = field(default_factory=Counter)


@dataclass
class DatasetGenStats:
    name: str
    source_root: Path
    source_subdir: str
    splits: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geração de label classe-única para um arquivo
# ---------------------------------------------------------------------------

def generate_single_class_label(src_path: Path, dst_path: Path) -> tuple[int, int, Counter]:
    """
    Lê src_path, troca o índice de classe de cada linha válida por 0,
    escreve em dst_path. Retorna (n_bboxes_validas, n_bboxes_invalidas,
    counter_classes_originais).
    """
    n_valid = 0
    n_invalid = 0
    original_classes: Counter = Counter()

    try:
        text = src_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0, original_classes

    new_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            n_invalid += 1
            continue
        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            n_invalid += 1
            continue
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
            n_invalid += 1
            continue

        # Linha válida → emite com classe 0
        new_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        n_valid += 1
        original_classes[cls] += 1

    # Escreve sempre, mesmo que vazio (mantém pareamento com a imagem)
    if new_lines:
        dst_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        dst_path.write_text("", encoding="utf-8")

    return n_valid, n_invalid, original_classes


# ---------------------------------------------------------------------------
# Processamento de um split
# ---------------------------------------------------------------------------

def process_split(
    name: str,
    source_dir: Path,
    output_dir: Path,
    label_for_progress: str,
) -> SplitGenStats:
    stats = SplitGenStats()

    if not source_dir.exists():
        print(f"  [{label_for_progress}] AVISO: source não existe: {source_dir}")
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(source_dir.glob("*.txt"))
    stats.files_input = len(label_files)

    print(
        f"  [{label_for_progress}] {len(label_files)} arquivos — "
        f"gerando com {MAX_WORKERS} threads...",
        flush=True,
    )

    if not label_files:
        return stats

    def _worker(src: Path):
        dst = output_dir / src.name
        n_valid, n_invalid, classes = generate_single_class_label(src, dst)
        return n_valid, n_invalid, classes, dst, n_valid > 0

    processed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_worker, lbl) for lbl in label_files]
        for fut in as_completed(futures):
            try:
                n_valid, n_invalid, classes, dst, has_content = fut.result()
            except Exception as exc:
                print(f"    erro: {exc}", flush=True)
                processed += 1
                continue

            stats.bboxes_total += n_valid
            stats.bboxes_invalid_skipped += n_invalid
            stats.original_class_counts.update(classes)
            stats.files_written += 1
            if not has_content:
                stats.files_empty += 1

            processed += 1
            if processed % PROGRESS_EVERY == 0:
                print(f"    progresso: {processed}/{len(label_files)}", flush=True)

    print(
        f"  [{label_for_progress}] OK: {stats.files_written}/{stats.files_input} "
        f"arquivos, {stats.bboxes_total} bboxes",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# Processamento de um dataset (todos os splits)
# ---------------------------------------------------------------------------

def process_dataset(
    name: str,
    root: Path,
    source_subdir: str,
) -> DatasetGenStats:
    ds = DatasetGenStats(name=name, source_root=root, source_subdir=source_subdir)

    if not root.exists():
        ds.notes.append(f"ROOT NOT FOUND: {root}")
        return ds

    print(f"\n>> Processando dataset: {name}")
    print(f"   root:           {root}")
    print(f"   source labels:  {source_subdir}/")
    print(f"   target labels:  {SINGLE_CLASS_SUBDIR}/")

    for split in SPLITS:
        source_dir = root / split / source_subdir
        output_dir = root / split / SINGLE_CLASS_SUBDIR

        if not source_dir.exists():
            ds.notes.append(f"split '{split}' missing source dir: {source_dir}")
            ds.splits[split] = SplitGenStats()
            continue

        ds.splits[split] = process_split(
            name, source_dir, output_dir, f"{name}/{split}"
        )

    return ds


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

# Mapeamentos de classes para o relatório (legibilidade)
CITRA_CLASS_NAMES = {
    0: "Militar", 1: "Barca", 2: "Mercante", 3: "Vela", 4: "Passageiro",
    5: "TUG", 6: "Lancha", 7: "Miuda", 8: "Navio",
}

INATECH_CLASS_NAMES = {
    0: "GENERAL CARGO",
    1: "CONTAINER SHIP",
    2: "BULK CARRIER",
    3: "PASSENGERS SHIP",
    4: "RO-RO/PASSENGER SHIP",
    5: "TUG",
    6: "OIL/CHEMICAL TANKER",
    7: "RO-RO CARGO",
    8: "VEHICLES CARRIER",
    9: "OIL PRODUCTS TANKER",
}

DATASET_CLASS_MAPS = {
    "CITRA-3D-Real": CITRA_CLASS_NAMES,
    "dataset_25k": INATECH_CLASS_NAMES,
}


def render_text_report(datasets: list[DatasetGenStats]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("RELATÓRIO DE GERAÇÃO DE LABELS CLASSE-ÚNICA")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Para cada dataset, todas as classes originais foram colapsadas para 0.")
    lines.append("Os labels originais NÃO foram tocados — labels_single_class/ é paralela.")
    lines.append("")

    for ds in datasets:
        lines.append("─" * 78)
        lines.append(f"### {ds.name}")
        lines.append(f"  source:  {ds.source_root}/{{split}}/{ds.source_subdir}/")
        lines.append(f"  target:  {ds.source_root}/{{split}}/{SINGLE_CLASS_SUBDIR}/")
        for note in ds.notes:
            lines.append(f"  note: {note}")

        class_map = DATASET_CLASS_MAPS.get(ds.name, {})

        total_in = 0
        total_out = 0
        total_bboxes = 0
        total_invalid = 0
        total_empty = 0
        grand_classes: Counter = Counter()

        for split in SPLITS:
            if split not in ds.splits:
                continue
            s = ds.splits[split]
            lines.append(f"  [{split}]")
            lines.append(f"    arquivos lidos:           {s.files_input}")
            lines.append(f"    arquivos escritos:        {s.files_written}")
            lines.append(f"    arquivos vazios:          {s.files_empty}")
            lines.append(f"    bboxes válidas:           {s.bboxes_total}")
            if s.bboxes_invalid_skipped > 0:
                lines.append(f"    bboxes inválidas (skip):  {s.bboxes_invalid_skipped}")

            total_in += s.files_input
            total_out += s.files_written
            total_bboxes += s.bboxes_total
            total_invalid += s.bboxes_invalid_skipped
            total_empty += s.files_empty
            grand_classes.update(s.original_class_counts)

        lines.append("")
        lines.append(f"  [CONSOLIDADO {ds.name}]")
        lines.append(f"    total arquivos lidos:     {total_in}")
        lines.append(f"    total arquivos escritos:  {total_out}")
        lines.append(f"    total arquivos vazios:    {total_empty}")
        lines.append(f"    total bboxes (todas → 0): {total_bboxes}")
        if total_invalid > 0:
            lines.append(f"    total bboxes inválidas:   {total_invalid}")
        lines.append("")

        # Distribuição original (preservada para referência histórica)
        if grand_classes:
            lines.append(f"  distribuição ORIGINAL (descartada no colapso, só para registro):")
            for cls_idx in sorted(grand_classes.keys()):
                name = class_map.get(cls_idx, f"classe_{cls_idx}")
                count = grand_classes[cls_idx]
                pct = 100.0 * count / total_bboxes if total_bboxes else 0.0
                lines.append(f"    {cls_idx} {name:<22} {count:>6}  ({pct:5.1f}%)")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


def dataset_to_dict(ds: DatasetGenStats) -> dict:
    out = {
        "name": ds.name,
        "source_root": str(ds.source_root),
        "source_subdir": ds.source_subdir,
        "target_subdir": SINGLE_CLASS_SUBDIR,
        "notes": list(ds.notes),
        "splits": {},
    }
    for split, s in ds.splits.items():
        out["splits"][split] = {
            "files_input": s.files_input,
            "files_written": s.files_written,
            "files_empty": s.files_empty,
            "bboxes_total": s.bboxes_total,
            "bboxes_invalid_skipped": s.bboxes_invalid_skipped,
            "original_class_counts": {
                str(int(k)): int(v) for k, v in s.original_class_counts.items()
            },
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f">> MAX_WORKERS = {MAX_WORKERS}")
    print(f">> SINGLE_CLASS_SUBDIR = {SINGLE_CLASS_SUBDIR}")
    print()

    datasets = []

    # Dataset 1: CITRA-3D-Real
    citra = process_dataset(
        "CITRA-3D-Real", CITRA_ROOT, CITRA_LABELS_SOURCE_SUBDIR
    )
    datasets.append(citra)

    # Dataset 2: dataset_25k (InaTechShips similar)
    inatech = process_dataset(
        "dataset_25k", INATECH_ROOT, INATECH_LABELS_SOURCE_SUBDIR
    )
    datasets.append(inatech)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    text_report = render_text_report(datasets)
    REPORT_TXT.write_text(text_report, encoding="utf-8")

    json_report = {
        "config": {
            "MAX_WORKERS": MAX_WORKERS,
            "SINGLE_CLASS_SUBDIR": SINGLE_CLASS_SUBDIR,
            "CITRA_ROOT": str(CITRA_ROOT),
            "INATECH_ROOT": str(INATECH_ROOT),
        },
        "datasets": [dataset_to_dict(d) for d in datasets],
    }
    REPORT_JSON.write_text(json.dumps(json_report, indent=2, ensure_ascii=False))

    print()
    print(text_report)
    print(f"\n>> Texto: {REPORT_TXT}")
    print(f">> JSON:  {REPORT_JSON}")
    print()
    print("=== Próximo passo ===")
    print("Geração de data.yaml classe-única (tarefa A4).")


if __name__ == "__main__":
    main()
