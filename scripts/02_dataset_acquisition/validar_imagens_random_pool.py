"""
validar_imagens_random_pool.py

Valida TODAS as imagens do random_pool_v2, separando arquivos legítimos
de arquivos corrompidos (HTML de erro, arquivos truncados, etc.).

Origem do problema:
  O shipspotting CDN aparentemente retorna 200 OK com uma página HTML
  de erro em vez de 404 para IDs que não existem. O script de download
  checou apenas status_code == 200 + tamanho > 5000 bytes, o que não
  descarta HTMLs (páginas de erro podem facilmente passar de 5KB).
  Resultado: arquivos salvos como .jpg mas que são HTML ou truncados.

O que este script faz:
  1. Itera sobre todas as imagens de cada split.
  2. Para cada arquivo, tenta abrir com PIL.Image.verify().
  3. Se passar, também tenta abrir para pegar dimensões (segundo check).
  4. Se passar ambos, é válido → fica onde está.
  5. Se falhar qualquer um, é corrompido → move para _corrompidas/{split}/.
  6. Gera relatório JSON estruturado.

Velocidade: ~500-1000 imagens/segundo via PIL lazy-verify.
Tempo estimado para 49k imagens: ~1-2 minutos.

Uso:
  python validar_imagens_random_pool.py            # valida tudo
  python validar_imagens_random_pool.py --split train   # só um split
  python validar_imagens_random_pool.py --dry-run  # não move nada
  python validar_imagens_random_pool.py --workers 4  # paralelização
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_DIR = Path.home() / "PROJETO_MARINHA" / "random_pool_v2"
QUARANTINE_DIR = BASE_DIR / "_corrompidas"
REPORT_FILE = BASE_DIR / "validation_report.json"
SPLITS = ("train", "val", "test")


def validate_one(img_path: Path) -> tuple[Path, bool, str]:
    """
    Valida uma imagem. Retorna (path, is_valid, reason).
    """
    try:
        from PIL import Image
    except ImportError:
        return (img_path, False, "PIL não instalado")

    # Check 1: tamanho mínimo razoável (HTMLs de erro tipicamente < 10KB,
    # mas JPEGs reais do shipspotting tipicamente > 30KB)
    try:
        size = img_path.stat().st_size
    except OSError as exc:
        return (img_path, False, f"stat failed: {exc}")

    if size < 5000:
        return (img_path, False, f"muito pequeno ({size} bytes)")

    # Check 2: header de JPEG (primeiros 3 bytes devem ser FF D8 FF)
    try:
        with open(img_path, "rb") as f:
            header = f.read(3)
        if header[:2] != b"\xff\xd8":
            # Não é um JPEG. Provavelmente HTML ou outra coisa.
            # Tenta detectar HTML explicitamente para o log.
            try:
                with open(img_path, "rb") as f:
                    start = f.read(100).lower()
                if b"<html" in start or b"<!doctype" in start or b"<head" in start:
                    return (img_path, False, "HTML em vez de JPEG")
            except OSError:
                pass
            return (img_path, False, f"header não é JPEG: {header.hex()}")
    except OSError as exc:
        return (img_path, False, f"read header failed: {exc}")

    # Check 3: PIL consegue abrir e verificar
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as exc:
        return (img_path, False, f"PIL verify falhou: {type(exc).__name__}")

    # Check 4: PIL consegue abrir e ler tamanho (verify invalida o handle)
    try:
        img2 = Image.open(img_path)
        w, h = img2.size
        if w < 50 or h < 50:
            return (img_path, False, f"dimensões absurdas: {w}x{h}")
    except Exception as exc:
        return (img_path, False, f"PIL open/size falhou: {type(exc).__name__}")

    return (img_path, True, "ok")


def validate_split(
    split: str,
    workers: int,
    dry_run: bool,
) -> dict:
    """Valida todas as imagens de um split."""
    images_dir = BASE_DIR / split / "images"
    if not images_dir.exists():
        return {"split": split, "error": f"pasta não existe: {images_dir}"}

    all_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".jpg")
    n_total = len(all_files)

    print(f"\n>> Validando {split}: {n_total:,} arquivos (workers={workers})")

    if n_total == 0:
        return {"split": split, "total": 0, "valid": 0, "corrupt": 0, "details": []}

    quarantine_split = QUARANTINE_DIR / split
    if not dry_run:
        quarantine_split.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    valid_count = 0
    corrupt_details: list[dict] = []
    reasons_summary: dict[str, int] = {}

    # Itera em ordem com report periódico
    def _task(p):
        return validate_one(p)

    completed = 0

    if workers == 1:
        # Sequencial
        for p in all_files:
            _, is_valid, reason = _task(p)
            if is_valid:
                valid_count += 1
            else:
                corrupt_details.append({"file": p.name, "reason": reason})
                reasons_summary[reason.split(":")[0]] = reasons_summary.get(
                    reason.split(":")[0], 0
                ) + 1
                if not dry_run:
                    try:
                        shutil.move(str(p), str(quarantine_split / p.name))
                    except OSError as exc:
                        print(f"   AVISO: falhou mover {p.name}: {exc}")
            completed += 1
            if completed % 2000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (n_total - completed) / rate if rate > 0 else 0
                print(
                    f"   [{completed:>6,}/{n_total:,}] "
                    f"valid={valid_count:,} corrupt={len(corrupt_details):,} "
                    f"| {rate:.0f}/s ETA {eta:.0f}s"
                )
    else:
        # Paralelo
        results: list[tuple[Path, bool, str]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_task, p) for p in all_files]
            for fut in as_completed(futures):
                results.append(fut.result())
                completed += 1
                if completed % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (n_total - completed) / rate if rate > 0 else 0
                    n_valid_so_far = sum(1 for r in results if r[1])
                    n_corrupt_so_far = completed - n_valid_so_far
                    print(
                        f"   [{completed:>6,}/{n_total:,}] "
                        f"valid={n_valid_so_far:,} corrupt={n_corrupt_so_far:,} "
                        f"| {rate:.0f}/s ETA {eta:.0f}s"
                    )

        # Processa resultados
        for path, is_valid, reason in results:
            if is_valid:
                valid_count += 1
            else:
                corrupt_details.append({"file": path.name, "reason": reason})
                reasons_summary[reason.split(":")[0]] = reasons_summary.get(
                    reason.split(":")[0], 0
                ) + 1
                if not dry_run:
                    try:
                        shutil.move(str(path), str(quarantine_split / path.name))
                    except OSError as exc:
                        print(f"   AVISO: falhou mover {path.name}: {exc}")

    elapsed = time.time() - t0

    print(f"   ✓ {split}: {valid_count:,} válidas, {len(corrupt_details):,} corrompidas "
          f"({len(corrupt_details) * 100 / n_total:.1f}%) em {elapsed:.0f}s")
    if reasons_summary:
        print(f"   Motivos:")
        for reason, count in sorted(reasons_summary.items(), key=lambda x: -x[1]):
            print(f"     {reason}: {count}")

    return {
        "split": split,
        "total": n_total,
        "valid": valid_count,
        "corrupt": len(corrupt_details),
        "corrupt_percentage": len(corrupt_details) * 100 / n_total,
        "reasons_summary": reasons_summary,
        "corrupt_details": corrupt_details,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida imagens do random_pool_v2")
    parser.add_argument(
        "--split", choices=SPLITS + ("all",), default="all",
        help="Qual split validar (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Workers para validação paralela (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Só reporta, não move arquivos corrompidos",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Validação de imagens do random_pool_v2")
    print("=" * 70)
    print(f"  Base:        {BASE_DIR}")
    print(f"  Quarentena:  {QUARANTINE_DIR}")
    print(f"  Split(s):    {args.split}")
    print(f"  Workers:     {args.workers}")
    print(f"  Dry-run:     {args.dry_run}")
    print("=" * 70)

    if not BASE_DIR.exists():
        print(f"ERRO: {BASE_DIR} não existe.")
        sys.exit(1)

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("ERRO: PIL/Pillow não instalado. `pip install Pillow`")
        sys.exit(1)

    splits_to_run = SPLITS if args.split == "all" else (args.split,)

    t_start = time.time()
    results = {}
    for split in splits_to_run:
        results[split] = validate_split(split, args.workers, args.dry_run)

    # Relatório final
    total_total = sum(r.get("total", 0) for r in results.values())
    total_valid = sum(r.get("valid", 0) for r in results.values())
    total_corrupt = sum(r.get("corrupt", 0) for r in results.values())

    report = {
        "generated_at": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "elapsed_seconds": time.time() - t_start,
        "results": results,
        "totals": {
            "total": total_total,
            "valid": total_valid,
            "corrupt": total_corrupt,
            "corrupt_percentage": (total_corrupt * 100 / total_total) if total_total > 0 else 0,
        },
    }

    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print()
    print("=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)
    print(f"  {'split':<10}{'total':>10}{'válidas':>12}{'corrompidas':>14}{'%':>8}")
    print(f"  " + "-" * 54)
    for split in splits_to_run:
        r = results[split]
        if "error" in r:
            print(f"  {split:<10}ERRO: {r['error']}")
            continue
        pct = r["corrupt_percentage"]
        print(
            f"  {split:<10}{r['total']:>10,}{r['valid']:>12,}"
            f"{r['corrupt']:>14,}{pct:>7.1f}%"
        )
    print(f"  " + "-" * 54)
    total_pct = (total_corrupt * 100 / total_total) if total_total > 0 else 0
    print(
        f"  {'TOTAL':<10}{total_total:>10,}{total_valid:>12,}"
        f"{total_corrupt:>14,}{total_pct:>7.1f}%"
    )
    print()
    print(f"  Relatório: {REPORT_FILE}")
    if not args.dry_run:
        print(f"  Arquivos corrompidos movidos para: {QUARANTINE_DIR}")
    else:
        print(f"  (dry-run: nenhum arquivo foi movido)")
    print("=" * 70)


if __name__ == "__main__":
    main()
