"""
preparar_citra3d.py

Inspeciona e extrai seletivamente o CITRA-3D.zip original, filtrando
apenas imagens reais (nomes começando por data DD.MM.YYYY-HH-MM-SS).

Duas fases independentes:

  Fase 1 (MODE='inspect'): lê o ZIP principal e os ZIPs aninhados sem
    extrair nada, gera relatório detalhado do conteúdo. Rápido.

  Fase 2 (MODE='extract'): depois de validar a Fase 1, extrai apenas
    imagens reais + labels pareados para /content/CITRA-3D-Extracted/.

Uso no Colab:
    # Primeiro rode com MODE='inspect' e veja o relatório:
    MODE = 'inspect'
    !python preparar_citra3d.py

    # Só depois, se estiver tudo OK, rode com MODE='extract':
    # (edite o arquivo ou use env var)
    MODE = 'extract'
    !python preparar_citra3d.py

Autora: Daniela L. Freire (ICMC/USP) — projeto Marinha do Brasil
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Entrada
CITRA_ZIP = "/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D.zip"
DATA_YAML = "/content/drive/MyDrive/PROJETO_MARINHA/Datasets/data.yaml"

# Saída (disco local do Colab — rápido, depois você sincroniza pro Drive)
OUTPUT_ROOT = Path("/content/CITRA-3D-Extracted")
REPORT_DIR = Path("/content/CITRA-3D-Extracted-reports")

# Modo de execução: 'inspect' ou 'extract'
# Na primeira rodada sempre use 'inspect' para ver o que tem dentro.
MODE = "inspect"

# Se True, ignora ZIPs aninhados que contenham "_aug" no nome (augmentations
# = sintéticas). Isso economiza enormemente tempo e espaço.
SKIP_AUG_ZIPS = True

# Regex que identifica imagens REAIS pelo início do nome do arquivo.
# Ex: "22.06.2022-14-48-29.png" → match
# Ex: "NavyA140Atlantico_TerrainOFF_3501yd_74T.png" → no match
REAL_NAME_PATTERN = re.compile(r"^\d{2}\.\d{2}\.\d{4}")

# Extensões reconhecidas
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"

# Limite de exemplos mostrados no relatório (para não explodir)
EXAMPLES_PER_CATEGORY = 10


# ---------------------------------------------------------------------------
# Estruturas de relatório
# ---------------------------------------------------------------------------

@dataclass
class InnerZipReport:
    name: str                                           # nome do ZIP aninhado
    skipped_as_aug: bool = False                        # se foi ignorado
    total_entries: int = 0
    images_total: int = 0
    labels_total: int = 0
    images_real: int = 0                                # passam no regex de data
    images_synthetic: int = 0                           # não passam
    images_real_with_label: int = 0                     # reais com .txt pareado
    images_real_without_label: int = 0                  # reais sem .txt
    labels_orphan: int = 0                              # .txt sem imagem real
    unknown_extensions: Counter = field(default_factory=Counter)
    top_dirs: Counter = field(default_factory=Counter)  # primeiras pastas dentro
    real_examples: list = field(default_factory=list)
    synthetic_examples: list = field(default_factory=list)


@dataclass
class OuterReport:
    citra_zip_path: str
    citra_zip_size_bytes: int = 0
    inner_zips_found: int = 0
    inner_zips_inspected: int = 0
    inner_zips_skipped: int = 0
    total_images_real: int = 0
    total_images_synthetic: int = 0
    total_images_real_with_label: int = 0
    total_images_real_without_label: int = 0
    inner_reports: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def is_label(name: str) -> bool:
    return Path(name).suffix.lower() == LABEL_EXT


def is_real_image(name: str) -> bool:
    """True se o *basename* começa com padrão de data DD.MM.YYYY."""
    return bool(REAL_NAME_PATTERN.match(Path(name).name))


def should_skip_inner_zip(inner_name: str) -> bool:
    """True se o ZIP aninhado deve ser pulado (augmentation/sintético)."""
    if not SKIP_AUG_ZIPS:
        return False
    lower = inner_name.lower()
    return "_aug" in lower or "aug-" in lower or "aug_" in lower


def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# ---------------------------------------------------------------------------
# Fase 1 — Inspeção
# ---------------------------------------------------------------------------

def inspect_inner_zip(inner_name: str, inner_bytes: bytes) -> InnerZipReport:
    """
    Abre um ZIP aninhado a partir de bytes em memória e conta os arquivos
    por categoria. NÃO extrai nada.
    """
    rep = InnerZipReport(name=inner_name)

    try:
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zf:
            image_stems_real: dict[str, str] = {}   # stem → path completo
            image_stems_synth: set[str] = set()
            label_stems: dict[str, str] = {}

            for info in zf.infolist():
                if info.is_dir():
                    continue

                name = info.filename
                rep.total_entries += 1

                # Top-level dir (primeira pasta dentro do zip)
                parts = name.split("/")
                if len(parts) > 1:
                    rep.top_dirs[parts[0]] += 1

                stem = Path(name).stem
                ext = Path(name).suffix.lower()

                if is_image(name):
                    rep.images_total += 1
                    if is_real_image(name):
                        rep.images_real += 1
                        image_stems_real[stem] = name
                        if len(rep.real_examples) < EXAMPLES_PER_CATEGORY:
                            rep.real_examples.append(name)
                    else:
                        rep.images_synthetic += 1
                        image_stems_synth.add(stem)
                        if len(rep.synthetic_examples) < EXAMPLES_PER_CATEGORY:
                            rep.synthetic_examples.append(name)
                elif is_label(name):
                    rep.labels_total += 1
                    label_stems[stem] = name
                else:
                    rep.unknown_extensions[ext] += 1

            # Pareamento imagens reais ↔ labels
            for stem in image_stems_real:
                if stem in label_stems:
                    rep.images_real_with_label += 1
                else:
                    rep.images_real_without_label += 1

            # Labels órfãos (.txt sem imagem real correspondente;
            # labels de sintéticas também caem aqui, o que é esperado
            # e não é problema — a gente não vai extrair esses labels)
            for stem in label_stems:
                if stem not in image_stems_real and stem not in image_stems_synth:
                    rep.labels_orphan += 1

    except zipfile.BadZipFile as exc:
        print(f"  !! ERRO: {inner_name} não é um ZIP válido: {exc}")

    return rep


def run_inspect() -> OuterReport:
    print("=" * 78)
    print("FASE 1 — INSPEÇÃO DO CITRA-3D.zip")
    print("=" * 78)

    if not os.path.exists(CITRA_ZIP):
        print(f"ERRO: não encontrei {CITRA_ZIP}")
        sys.exit(1)

    outer = OuterReport(citra_zip_path=CITRA_ZIP)
    outer.citra_zip_size_bytes = os.path.getsize(CITRA_ZIP)
    print(f"\nArquivo: {CITRA_ZIP}")
    print(f"Tamanho: {fmt_bytes(outer.citra_zip_size_bytes)}\n")

    print("Abrindo ZIP principal (sem extrair)...")
    with zipfile.ZipFile(CITRA_ZIP, "r") as outer_zf:
        inner_zip_names = [
            info.filename for info in outer_zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".zip")
        ]
        outer.inner_zips_found = len(inner_zip_names)
        print(f"ZIPs aninhados encontrados: {len(inner_zip_names)}")
        for n in inner_zip_names:
            size = outer_zf.getinfo(n).file_size
            skip = should_skip_inner_zip(n)
            marker = "  SKIP" if skip else "      "
            print(f"  {marker}  {n}  ({fmt_bytes(size)})")

        print()
        for inner_name in inner_zip_names:
            if should_skip_inner_zip(inner_name):
                rep = InnerZipReport(name=inner_name, skipped_as_aug=True)
                outer.inner_zips_skipped += 1
                outer.inner_reports.append(rep)
                continue

            print(f">> Inspecionando {inner_name}...", flush=True)
            with outer_zf.open(inner_name) as handle:
                inner_bytes = handle.read()
            rep = inspect_inner_zip(inner_name, inner_bytes)
            outer.inner_reports.append(rep)
            outer.inner_zips_inspected += 1

            outer.total_images_real += rep.images_real
            outer.total_images_synthetic += rep.images_synthetic
            outer.total_images_real_with_label += rep.images_real_with_label
            outer.total_images_real_without_label += rep.images_real_without_label

            print(
                f"   total entries: {rep.total_entries} | "
                f"imgs: {rep.images_total} (real {rep.images_real}, "
                f"synth {rep.images_synthetic}) | "
                f"labels: {rep.labels_total} | "
                f"real pareados: {rep.images_real_with_label}"
            )

    return outer


def render_inspect_report(outer: OuterReport) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("RELATÓRIO DE INSPEÇÃO — CITRA-3D.zip")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Arquivo:              {outer.citra_zip_path}")
    lines.append(f"Tamanho:              {fmt_bytes(outer.citra_zip_size_bytes)}")
    lines.append(f"ZIPs aninhados:       {outer.inner_zips_found}")
    lines.append(f"  inspecionados:      {outer.inner_zips_inspected}")
    lines.append(f"  pulados (aug):      {outer.inner_zips_skipped}")
    lines.append("")
    lines.append("── TOTAIS (apenas ZIPs inspecionados) ──")
    lines.append(f"Imagens reais totais:             {outer.total_images_real:,}")
    lines.append(f"Imagens sintéticas totais:        {outer.total_images_synthetic:,}")
    lines.append(f"Imagens reais com label pareado:  {outer.total_images_real_with_label:,}")
    lines.append(f"Imagens reais SEM label pareado:  {outer.total_images_real_without_label:,}")
    lines.append("")

    for rep in outer.inner_reports:
        lines.append("─" * 78)
        lines.append(f"### {rep.name}")
        if rep.skipped_as_aug:
            lines.append("    STATUS: pulado (contém '_aug' → augmentation/sintético)")
            continue
        lines.append(f"    total entries:           {rep.total_entries}")
        lines.append(f"    imagens total:           {rep.images_total}")
        lines.append(f"      reais:                 {rep.images_real}")
        lines.append(f"      sintéticas:            {rep.images_synthetic}")
        lines.append(f"    labels total:            {rep.labels_total}")
        lines.append(f"    reais com label:         {rep.images_real_with_label}")
        lines.append(f"    reais sem label:         {rep.images_real_without_label}")
        lines.append(f"    labels órfãos:           {rep.labels_orphan}")
        if rep.unknown_extensions:
            lines.append(f"    extensões desconhecidas: {dict(rep.unknown_extensions)}")
        if rep.top_dirs:
            top_str = ", ".join(f"{d}={c}" for d, c in rep.top_dirs.most_common(5))
            lines.append(f"    top-level dirs:          {top_str}")
        if rep.real_examples:
            lines.append(f"    ex. reais:               {rep.real_examples[:3]}")
        if rep.synthetic_examples:
            lines.append(f"    ex. sintéticas:          {rep.synthetic_examples[:3]}")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fase 2 — Extração seletiva
# ---------------------------------------------------------------------------

def split_from_inner_name(inner_name: str) -> str:
    """
    Deriva o split (train/val/test) a partir do nome do ZIP aninhado.
    Ex: 'train-20260103T082001Z-1-001.zip' → 'train'
        'val-20260103T104420Z-1-001.zip'   → 'val'
        'test-20260103T110900Z-1-001.zip'  → 'test'
    """
    base = Path(inner_name).name.lower()
    if base.startswith("train"):
        return "train"
    if base.startswith("val"):
        return "val"
    if base.startswith("test"):
        return "test"
    return "unknown"


def run_extract() -> dict:
    print("=" * 78)
    print("FASE 2 — EXTRAÇÃO SELETIVA (apenas imagens reais + labels)")
    print("=" * 78)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Copia o data.yaml original para referência (não para uso direto —
    # vamos gerar um novo depois com val/test corrigidos)
    if os.path.exists(DATA_YAML):
        import shutil
        shutil.copy2(DATA_YAML, OUTPUT_ROOT / "data.yaml.original")
        print(f"  Copiei data.yaml original para {OUTPUT_ROOT}/data.yaml.original")

    stats = {
        "splits": {},
        "totals": {
            "images_extracted": 0,
            "labels_extracted": 0,
            "images_skipped_synthetic": 0,
            "images_skipped_no_label": 0,
        },
        "skipped_inner_zips": [],
    }

    with zipfile.ZipFile(CITRA_ZIP, "r") as outer_zf:
        inner_zip_names = [
            info.filename for info in outer_zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".zip")
        ]

        for inner_name in inner_zip_names:
            if should_skip_inner_zip(inner_name):
                stats["skipped_inner_zips"].append(inner_name)
                print(f">> SKIP {inner_name} (aug)")
                continue

            split = split_from_inner_name(inner_name)
            print(f"\n>> Processando {inner_name} → split '{split}'")

            split_stats = stats["splits"].setdefault(split, {
                "inner_zips": [],
                "images_extracted": 0,
                "labels_extracted": 0,
                "images_skipped_synthetic": 0,
                "images_skipped_no_label": 0,
            })
            split_stats["inner_zips"].append(inner_name)

            images_dir = OUTPUT_ROOT / split / "images"
            labels_dir = OUTPUT_ROOT / split / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            with outer_zf.open(inner_name) as handle:
                inner_bytes = handle.read()

            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                # Indexar: stems de labels e de imagens
                label_infos: dict[str, zipfile.ZipInfo] = {}
                real_image_infos: dict[str, zipfile.ZipInfo] = {}

                for info in inner_zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    stem = Path(name).stem
                    if is_label(name):
                        label_infos[stem] = info
                    elif is_image(name):
                        if is_real_image(name):
                            real_image_infos[stem] = info
                        else:
                            split_stats["images_skipped_synthetic"] += 1

                # Extrair só imagens reais que tenham label pareado
                for stem, img_info in real_image_infos.items():
                    if stem not in label_infos:
                        split_stats["images_skipped_no_label"] += 1
                        continue

                    # Extrai imagem (renomeia para só o basename, descartando
                    # a estrutura de pastas interna)
                    img_basename = Path(img_info.filename).name
                    lbl_basename = Path(label_infos[stem].filename).name

                    with inner_zf.open(img_info) as src, \
                         open(images_dir / img_basename, "wb") as dst:
                        dst.write(src.read())

                    with inner_zf.open(label_infos[stem]) as src, \
                         open(labels_dir / lbl_basename, "wb") as dst:
                        dst.write(src.read())

                    split_stats["images_extracted"] += 1
                    split_stats["labels_extracted"] += 1

            print(
                f"   extraídos: {split_stats['images_extracted']} imgs, "
                f"{split_stats['labels_extracted']} labels"
            )

        # Totais finais
        for split, s in stats["splits"].items():
            for k in ("images_extracted", "labels_extracted",
                      "images_skipped_synthetic", "images_skipped_no_label"):
                stats["totals"][k] += s[k]

    return stats


def render_extract_report(stats: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("RELATÓRIO DE EXTRAÇÃO — CITRA-3D (imagens reais)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Destino: {OUTPUT_ROOT}")
    lines.append("")
    lines.append("── TOTAIS ──")
    lines.append(f"Imagens extraídas:              {stats['totals']['images_extracted']:,}")
    lines.append(f"Labels extraídos:               {stats['totals']['labels_extracted']:,}")
    lines.append(f"Sintéticas puladas:             {stats['totals']['images_skipped_synthetic']:,}")
    lines.append(f"Reais sem label (puladas):      {stats['totals']['images_skipped_no_label']:,}")
    lines.append("")
    lines.append("── POR SPLIT ──")
    for split, s in stats["splits"].items():
        lines.append(f"  [{split}]")
        lines.append(f"    imgs extraídas:    {s['images_extracted']:,}")
        lines.append(f"    labels extraídos:  {s['labels_extracted']:,}")
        lines.append(f"    sintéticas:        {s['images_skipped_synthetic']:,}")
        lines.append(f"    reais s/ label:    {s['images_skipped_no_label']:,}")
        lines.append(f"    zips processados:  {s['inner_zips']}")
    lines.append("")
    lines.append("── ZIPs AUG PULADOS ──")
    for n in stats["skipped_inner_zips"]:
        lines.append(f"  {n}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_outer_report(outer: OuterReport) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Texto
    text = render_inspect_report(outer)
    (REPORT_DIR / "inspect_report.txt").write_text(text, encoding="utf-8")
    # JSON — construído manualmente para evitar problemas com Counter
    outer_dict = {
        "citra_zip_path": outer.citra_zip_path,
        "citra_zip_size_bytes": outer.citra_zip_size_bytes,
        "inner_zips_found": outer.inner_zips_found,
        "inner_zips_inspected": outer.inner_zips_inspected,
        "inner_zips_skipped": outer.inner_zips_skipped,
        "total_images_real": outer.total_images_real,
        "total_images_synthetic": outer.total_images_synthetic,
        "total_images_real_with_label": outer.total_images_real_with_label,
        "total_images_real_without_label": outer.total_images_real_without_label,
        "inner_reports": [],
    }
    for rep in outer.inner_reports:
        outer_dict["inner_reports"].append({
            "name": rep.name,
            "skipped_as_aug": rep.skipped_as_aug,
            "total_entries": rep.total_entries,
            "images_total": rep.images_total,
            "labels_total": rep.labels_total,
            "images_real": rep.images_real,
            "images_synthetic": rep.images_synthetic,
            "images_real_with_label": rep.images_real_with_label,
            "images_real_without_label": rep.images_real_without_label,
            "labels_orphan": rep.labels_orphan,
            "unknown_extensions": {str(k): int(v) for k, v in rep.unknown_extensions.items()},
            "top_dirs": {str(k): int(v) for k, v in rep.top_dirs.items()},
            "real_examples": list(rep.real_examples),
            "synthetic_examples": list(rep.synthetic_examples),
        })
    (REPORT_DIR / "inspect_report.json").write_text(
        json.dumps(outer_dict, indent=2, ensure_ascii=False)
    )
    print()
    print(text)
    print(f"\n>> Relatório salvo em {REPORT_DIR}/inspect_report.txt")
    print(f">> JSON salvo em      {REPORT_DIR}/inspect_report.json")


def save_extract_report(stats: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = render_extract_report(stats)
    (REPORT_DIR / "extract_report.txt").write_text(text, encoding="utf-8")
    (REPORT_DIR / "extract_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    print()
    print(text)
    print(f"\n>> Relatório salvo em {REPORT_DIR}/extract_report.txt")


def main() -> None:
    print(f"MODE = {MODE}")
    print(f"SKIP_AUG_ZIPS = {SKIP_AUG_ZIPS}")
    print()

    if MODE == "inspect":
        outer = run_inspect()
        save_outer_report(outer)
        print("\n=== Próximo passo ===")
        print("Se o relatório estiver OK, edite MODE='extract' e rode de novo.")
    elif MODE == "extract":
        stats = run_extract()
        save_extract_report(stats)
        print("\n=== Próximo passo ===")
        print(f"Verifique os arquivos em {OUTPUT_ROOT}")
        print("Depois, rode o analise_datasets_parallel.py nesse caminho")
        print("para auditoria completa (pareamento, classes, bboxes).")
    else:
        print(f"ERRO: MODE inválido: {MODE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
