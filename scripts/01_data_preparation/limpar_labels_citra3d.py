"""
limpar_labels_citra3d.py

Limpa labels do CITRA-3D-Real removendo linhas malformadas e bboxes
degeneradas, preservando as linhas válidas do mesmo arquivo.

Escreve os labels limpos em pastas paralelas `labels_cleaned/` dentro
de cada split, sem tocar nos labels originais.

Regras de limpeza aplicadas a cada linha de cada arquivo:

  1. Linha em branco → descartada silenciosamente (normal).
  2. Linha com número de campos ≠ 5 → descartada, registrada como "malformed".
  3. Linha com campo não-numérico → descartada, registrada como "non_numeric".
  4. Linha com x, y, w, h fora de [0, 1] → descartada, registrada como "out_of_range".
  5. Linha com w ≤ 0 ou h ≤ 0 → descartada, registrada como "degenerate".
  6. Linha válida → escrita no arquivo de saída, inalterada.

Observações importantes:

  - Classes NÃO são tocadas — esta limpeza preserva os índices originais
    (0-8) do CITRA-3D. A redução para classe única é outro passo.
  - Arquivos que ficam totalmente vazios após a limpeza são registrados
    em seção destacada do relatório, para inspeção manual.
  - Labels originais permanecem intactos em labels/ — nada é sobrescrito.

Uso:
  python limpar_labels_citra3d.py

Saída:
  - Pastas labels_cleaned/ em {train,val,test}
  - Relatório texto: limpeza_labels_report.txt
  - Relatório JSON:  limpeza_labels_report.json
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CITRA_ROOT = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real")
REPORT_DIR = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets")
REPORT_TXT = REPORT_DIR / "limpeza_labels_report.txt"
REPORT_JSON = REPORT_DIR / "limpeza_labels_report.json"

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------

@dataclass
class LineIssue:
    filename: str
    split: str
    lineno: int
    raw_line: str
    reason: str


@dataclass
class SplitCleanStats:
    files_total: int = 0
    files_cleaned_unchanged: int = 0     # arquivos que não tinham nenhum problema
    files_cleaned_modified: int = 0      # arquivos que perderam 1+ linha
    files_emptied: int = 0               # arquivos que ficaram vazios após limpeza
    lines_total: int = 0
    lines_valid: int = 0
    lines_removed: int = 0
    removals_by_reason: Counter = field(default_factory=Counter)
    issues: list = field(default_factory=list)
    emptied_files: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Limpeza de uma linha
# ---------------------------------------------------------------------------

def classify_line(raw: str):
    """
    Retorna ('ok', formatted_line) para linhas válidas,
    ou (reason, None) para linhas inválidas onde reason é uma string.
    """
    line = raw.strip()
    if not line:
        return ("blank", None)

    parts = line.split()
    if len(parts) != 5:
        return ("malformed_field_count", None)

    try:
        cls = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return ("non_numeric", None)

    if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
        return ("out_of_range", None)

    if w <= 0.0 or h <= 0.0:
        return ("degenerate_zero_dim", None)

    # Reformata para consistência (mesma precisão em todos os arquivos).
    formatted = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
    return ("ok", formatted)


# ---------------------------------------------------------------------------
# Limpeza de um arquivo
# ---------------------------------------------------------------------------

def clean_label_file(
    src_path: Path,
    dst_path: Path,
    split: str,
    stats: SplitCleanStats,
) -> None:
    stats.files_total += 1

    try:
        text = src_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        stats.issues.append(LineIssue(
            filename=src_path.name, split=split, lineno=0,
            raw_line="", reason=f"read_error: {exc}"
        ))
        return

    valid_lines = []
    file_had_removals = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        kind, formatted = classify_line(raw)
        stats.lines_total += 1

        if kind == "ok":
            valid_lines.append(formatted)
            stats.lines_valid += 1
        elif kind == "blank":
            # Não conta como linha do arquivo (whitespace é normal).
            stats.lines_total -= 1
            continue
        else:
            stats.lines_removed += 1
            stats.removals_by_reason[kind] += 1
            file_had_removals = True
            stats.issues.append(LineIssue(
                filename=src_path.name,
                split=split,
                lineno=lineno,
                raw_line=raw.rstrip(),
                reason=kind,
            ))

    # Escreve arquivo de saída
    if valid_lines:
        dst_path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    else:
        # Arquivo ficou totalmente vazio após limpeza.
        dst_path.write_text("", encoding="utf-8")
        stats.files_emptied += 1
        stats.emptied_files.append(src_path.name)

    if file_had_removals:
        stats.files_cleaned_modified += 1
    else:
        stats.files_cleaned_unchanged += 1


# ---------------------------------------------------------------------------
# Limpeza de um split
# ---------------------------------------------------------------------------

def clean_split(split: str) -> SplitCleanStats:
    stats = SplitCleanStats()

    labels_dir = CITRA_ROOT / split / "labels"
    out_dir = CITRA_ROOT / split / "labels_cleaned"

    if not labels_dir.exists():
        print(f"[{split}] labels/ não encontrado em {labels_dir}")
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"[{split}] processando {len(label_files)} arquivos...")

    for i, src in enumerate(label_files, start=1):
        dst = out_dir / src.name
        clean_label_file(src, dst, split, stats)
        if i % 500 == 0:
            print(f"  [{split}] progresso: {i}/{len(label_files)}")

    return stats


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def render_text_report(all_stats: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("RELATÓRIO DE LIMPEZA DE LABELS — CITRA-3D-Real")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"root: {CITRA_ROOT}")
    lines.append("")
    lines.append("Saída: labels_cleaned/ em cada split (originais preservados)")
    lines.append("")

    # Totais globais
    g_files = 0
    g_unchanged = 0
    g_modified = 0
    g_emptied = 0
    g_lines = 0
    g_valid = 0
    g_removed = 0
    g_reasons: Counter = Counter()

    for split, s in all_stats.items():
        g_files += s.files_total
        g_unchanged += s.files_cleaned_unchanged
        g_modified += s.files_cleaned_modified
        g_emptied += s.files_emptied
        g_lines += s.lines_total
        g_valid += s.lines_valid
        g_removed += s.lines_removed
        g_reasons.update(s.removals_by_reason)

    lines.append("── TOTAIS GLOBAIS ──")
    lines.append(f"  arquivos processados:      {g_files}")
    lines.append(f"    sem alterações:          {g_unchanged}")
    lines.append(f"    com linhas removidas:    {g_modified}")
    lines.append(f"    vazios após limpeza:     {g_emptied}")
    lines.append(f"  linhas não-vazias totais:  {g_lines}")
    lines.append(f"    válidas (preservadas):   {g_valid}")
    lines.append(f"    removidas:               {g_removed}")
    if g_reasons:
        lines.append(f"  motivos de remoção:")
        for reason, count in sorted(g_reasons.items()):
            lines.append(f"    {reason:<25} {count}")
    lines.append("")

    # Por split
    for split in SPLITS:
        if split not in all_stats:
            continue
        s = all_stats[split]
        lines.append(f"── [{split}] ──")
        lines.append(f"  arquivos processados:      {s.files_total}")
        lines.append(f"    sem alterações:          {s.files_cleaned_unchanged}")
        lines.append(f"    com linhas removidas:    {s.files_cleaned_modified}")
        lines.append(f"    vazios após limpeza:     {s.files_emptied}")
        lines.append(f"  linhas não-vazias:         {s.lines_total}")
        lines.append(f"    válidas:                 {s.lines_valid}")
        lines.append(f"    removidas:               {s.lines_removed}")
        if s.removals_by_reason:
            for reason, count in sorted(s.removals_by_reason.items()):
                lines.append(f"    {reason:<25} {count}")
        lines.append("")

    # Detalhes de arquivos emptied (crítico — precisam inspeção manual)
    any_emptied = any(all_stats[s].emptied_files for s in all_stats)
    if any_emptied:
        lines.append("── ARQUIVOS VAZIOS APÓS LIMPEZA (inspeção manual recomendada) ──")
        for split, s in all_stats.items():
            for f in s.emptied_files:
                lines.append(f"  [{split}] {f}")
        lines.append("")
    else:
        lines.append("── ARQUIVOS VAZIOS APÓS LIMPEZA ──")
        lines.append("  (nenhum — todos os arquivos ainda têm pelo menos uma bbox válida)")
        lines.append("")

    # Detalhe linha-a-linha das remoções
    lines.append("── DETALHAMENTO DAS LINHAS REMOVIDAS ──")
    for split in SPLITS:
        if split not in all_stats:
            continue
        s = all_stats[split]
        if not s.issues:
            lines.append(f"  [{split}] nenhuma remoção")
            continue
        lines.append(f"  [{split}]")
        for issue in s.issues:
            lines.append(
                f"    {issue.filename}  line {issue.lineno}  "
                f"[{issue.reason}]  raw: {issue.raw_line!r}"
            )
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def stats_to_dict(s: SplitCleanStats) -> dict:
    return {
        "files_total": s.files_total,
        "files_cleaned_unchanged": s.files_cleaned_unchanged,
        "files_cleaned_modified": s.files_cleaned_modified,
        "files_emptied": s.files_emptied,
        "lines_total": s.lines_total,
        "lines_valid": s.lines_valid,
        "lines_removed": s.lines_removed,
        "removals_by_reason": dict(s.removals_by_reason),
        "emptied_files": list(s.emptied_files),
        "issues": [
            {
                "filename": i.filename,
                "split": i.split,
                "lineno": i.lineno,
                "raw_line": i.raw_line,
                "reason": i.reason,
            }
            for i in s.issues
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f">> CITRA_ROOT = {CITRA_ROOT}")
    print()

    all_stats = {}
    for split in SPLITS:
        stats = clean_split(split)
        all_stats[split] = stats

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Texto
    text_report = render_text_report(all_stats)
    REPORT_TXT.write_text(text_report, encoding="utf-8")

    # JSON
    json_report = {
        "citra_root": str(CITRA_ROOT),
        "splits": {s: stats_to_dict(all_stats[s]) for s in all_stats},
    }
    REPORT_JSON.write_text(json.dumps(json_report, indent=2, ensure_ascii=False))

    print()
    print(text_report)
    print(f"\n>> Texto salvo em: {REPORT_TXT}")
    print(f">> JSON salvo em:  {REPORT_JSON}")


if __name__ == "__main__":
    main()
