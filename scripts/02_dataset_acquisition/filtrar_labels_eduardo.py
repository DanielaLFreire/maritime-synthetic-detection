"""
filtrar_labels_eduardo.py

Filtra os labels compartilhados pelo Eduardo Teixeira (8 pastas,
~2,8M labels) para extrair apenas os 39.628 IDs do random_pool_v2.

CONTEXTO

  Eduardo compartilhou labels via Google Drive em 8 pastas organizadas
  por faixa de ID (0-500k, 500k-1M, etc.). Todos já estão em classe 0
  (single-class). Alguns IDs podem não ter label (limiar do PointRend
  não detectou embarcação).

  Este script:
    1. Lê a lista de 39.628 IDs do random_pool_v2
    2. Busca os labels correspondentes nas 8 pastas do Eduardo
    3. Copia os encontrados para a estrutura do random_pool_v2
    4. Reporta quantos foram encontrados vs faltantes

PRÉ-REQUISITOS

  - Drive montado no Colab
  - Pasta compartilhada do Eduardo adicionada ao Drive (atalho ou cópia)
    Caminho esperado: /content/drive/MyDrive/labels_eduardo/ (ou similar)
  - ids_random_pool_v2_validos.txt no Drive
  - random_pool_v2/{train,val,test}/images/ com as imagens

USO

  python filtrar_labels_eduardo.py
  python filtrar_labels_eduardo.py --labels-src /caminho/para/pastas/eduardo
  python filtrar_labels_eduardo.py --dry-run    # só conta, não copia

SAÍDA

  Labels copiados para:
    /content/drive/MyDrive/InaTechShips/random_pool_v2/{train,val,test}/labels/
  
  Relatório:
    /content/drive/MyDrive/InaTechShips/random_pool_v2/filtrar_labels_report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

DRIVE_BASE = Path("/content/drive/MyDrive")

# Caminho da pasta compartilhada do Eduardo (ajustar após adicionar ao Drive)
EDUARDO_LABELS_DEFAULT = DRIVE_BASE / "labels_eduardo"

# Alternativas comuns
EDUARDO_LABELS_ALTERNATIVES = [
    DRIVE_BASE / "labels_eduardo",
    DRIVE_BASE / "InaTechShips" / "labels_eduardo",
    Path("/content/drive/Shareddrives"),  # se for Shared Drive
]

# IDs do random_pool_v2
IDS_FILE = DRIVE_BASE / "InaTechShips" / "random_pool_v2" / "ids_random_pool_v2_validos.txt"

# random_pool_v2
RANDOM_POOL_ROOT = DRIVE_BASE / "InaTechShips" / "random_pool_v2"

SPLITS = ("train", "val", "test")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def find_eduardo_labels(custom_path: Path | None = None) -> Path | None:
    """Procura a pasta de labels do Eduardo."""
    if custom_path and custom_path.exists():
        return custom_path
    for p in EDUARDO_LABELS_ALTERNATIVES:
        if p.exists():
            return p
    return None


def extract_rars(root: Path, extract_to: Path) -> Path:
    """
    Extrai todos os .rar da pasta do Eduardo para um diretório local.
    
    Retorna o diretório onde os labels foram extraídos.
    """
    rar_files = sorted(root.glob("*.rar"))
    if not rar_files:
        print(f"   Nenhum .rar encontrado em {root}")
        return root
    
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"   Encontrados {len(rar_files)} arquivos .rar")
    
    # Instala unrar se necessário
    try:
        subprocess.run(["unrar", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"   Instalando unrar...")
        subprocess.run(["pip", "install", "unrar-free", "-q"],
                       capture_output=True)
        # Tenta instalar via apt (Colab/Linux)
        subprocess.run(["apt-get", "install", "-y", "-q", "unrar"],
                       capture_output=True)
    
    for rar_file in rar_files:
        rar_name = rar_file.stem
        dest = extract_to / rar_name
        
        # Pula se já extraído
        if dest.exists() and any(dest.glob("*.txt")):
            n_existing = sum(1 for _ in dest.glob("*.txt"))
            print(f"   ✓ {rar_name}: já extraído ({n_existing:,} labels)")
            continue
        
        dest.mkdir(parents=True, exist_ok=True)
        print(f"   Extraindo {rar_file.name}...", end=" ", flush=True)
        
        try:
            result = subprocess.run(
                ["unrar", "e", "-o+", str(rar_file), str(dest) + "/"],
                capture_output=True, text=True, timeout=600,
            )
            n_extracted = sum(1 for _ in dest.glob("*.txt"))
            print(f"OK ({n_extracted:,} labels)")
        except FileNotFoundError:
            # Tenta com python-unrar como fallback
            try:
                import rarfile
                with rarfile.RarFile(str(rar_file)) as rf:
                    rf.extractall(str(dest))
                n_extracted = sum(1 for _ in dest.glob("*.txt"))
                print(f"OK ({n_extracted:,} labels)")
            except ImportError:
                print(f"FALHOU — instale unrar: apt-get install unrar")
                continue
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT")
            continue
    
    return extract_to


def scan_eduardo_folders(root: Path) -> dict[str, Path]:
    """
    Escaneia as subpastas do Eduardo e indexa todos os labels por ID.
    
    Retorna: {photo_id_str: Path_do_label}
    """
    index: dict[str, Path] = {}
    
    # Lista subpastas
    subfolders = sorted([d for d in root.iterdir() if d.is_dir()])
    
    if not subfolders:
        # Talvez os .txt estejam direto na raiz
        subfolders = [root]
    
    print(f"   Escaneando {len(subfolders)} pasta(s)...")
    
    for folder in subfolders:
        n_in_folder = 0
        for f in folder.iterdir():
            if f.suffix == ".txt":
                index[f.stem] = f
                n_in_folder += 1
        print(f"     {folder.name}: {n_in_folder:,} labels")
    
    # Também verifica .txt soltos na raiz (caso extração flatten)
    n_root = 0
    for f in root.iterdir():
        if f.suffix == ".txt" and f.stem not in index:
            index[f.stem] = f
            n_root += 1
    if n_root > 0:
        print(f"     (raiz): {n_root:,} labels")
    
    return index


def get_split_for_id(photo_id: str, pool_root: Path) -> str | None:
    """Determina em qual split (train/val/test) um ID está no random_pool_v2."""
    for split in SPLITS:
        img_path = pool_root / split / "images" / f"{photo_id}.jpg"
        if img_path.exists():
            return split
    # Tenta no _excedente
    for split in SPLITS:
        exc_path = pool_root / "_excedente" / split / f"{photo_id}.jpg"
        if exc_path.exists():
            return f"_excedente/{split}"
    return None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Filtrar labels do Eduardo para random_pool_v2")
    parser.add_argument("--labels-src", type=Path, default=None,
                        help="Pasta raiz dos labels do Eduardo")
    parser.add_argument("--ids-file", type=Path, default=IDS_FILE,
                        help="Arquivo com IDs do random_pool_v2")
    parser.add_argument("--pool-root", type=Path, default=RANDOM_POOL_ROOT,
                        help="Root do random_pool_v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Só conta, não copia")
    args = parser.parse_args()

    print("=" * 72)
    print("  Filtrar labels do Eduardo → random_pool_v2")
    print("=" * 72)
    print(f"  Dry-run: {args.dry_run}")
    print("=" * 72)

    # ── Encontra labels do Eduardo ──
    labels_root = find_eduardo_labels(args.labels_src)
    if labels_root is None:
        print(f"\n✗ Pasta de labels do Eduardo não encontrada.")
        print(f"  Opções:")
        print(f"    1. Adicione a pasta compartilhada ao seu Drive")
        print(f"    2. Use --labels-src /caminho/para/pasta")
        print(f"\n  Caminhos tentados:")
        for p in EDUARDO_LABELS_ALTERNATIVES:
            print(f"    {p}")
        sys.exit(1)

    print(f"\n>> Labels do Eduardo: {labels_root}")

    # ── Extrai RARs (se houver) ──
    rar_files = list(labels_root.glob("*.rar"))
    if rar_files:
        print(f"\n>> Detectados {len(rar_files)} arquivos .rar — extraindo...")
        extract_dir = labels_root.parent / "labels_eduardo_extracted"
        labels_scan_dir = extract_rars(labels_root, extract_dir)
    else:
        labels_scan_dir = labels_root

    # ── Lê IDs alvo ──
    if not args.ids_file.exists():
        print(f"\n✗ Arquivo de IDs não encontrado: {args.ids_file}")
        sys.exit(1)

    with open(args.ids_file) as f:
        target_ids = set(line.strip() for line in f if line.strip())
    print(f">> IDs alvo: {len(target_ids):,}")

    # ── Indexa labels do Eduardo ──
    print(f"\n>> Indexando labels...")
    t0 = time.time()
    label_index = scan_eduardo_folders(labels_scan_dir)
    print(f"   Total indexado: {len(label_index):,} labels em {time.time()-t0:.0f}s")

    # ── Cruza IDs ──
    print(f"\n>> Cruzando IDs...")
    found_ids = target_ids & set(label_index.keys())
    missing_ids = target_ids - set(label_index.keys())

    print(f"   Encontrados: {len(found_ids):,} ({len(found_ids)/len(target_ids)*100:.1f}%)")
    print(f"   Faltantes:   {len(missing_ids):,} ({len(missing_ids)/len(target_ids)*100:.1f}%)")

    if missing_ids:
        # Mostra alguns exemplos
        sample = sorted(missing_ids)[:10]
        print(f"   Exemplos de IDs faltantes: {', '.join(sample[:10])}")

    # ── Determina split de cada ID e copia ──
    print(f"\n>> Mapeando IDs para splits e copiando labels...")
    
    stats = defaultdict(int)
    copy_errors = []
    n_copied = 0
    print_every = max(500, len(found_ids) // 20)

    # Cria diretórios de labels
    if not args.dry_run:
        for split in SPLITS:
            labels_dir = args.pool_root / split / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

    for i, photo_id in enumerate(sorted(found_ids), 1):
        split = get_split_for_id(photo_id, args.pool_root)
        
        if split is None:
            stats["no_image"] += 1
            continue

        if split.startswith("_excedente"):
            stats["in_excedente"] += 1
            # Não copia label para excedentes (não serão usados no treino)
            continue

        src_label = label_index[photo_id]
        dst_label = args.pool_root / split / "labels" / f"{photo_id}.txt"

        if not args.dry_run:
            try:
                shutil.copy2(str(src_label), str(dst_label))
                n_copied += 1
                stats[split] += 1
            except Exception as exc:
                copy_errors.append({"photo_id": photo_id, "error": str(exc)})
                stats["errors"] += 1
        else:
            n_copied += 1
            stats[split] += 1

        if i % print_every == 0:
            print(f"   [{i:>6,}/{len(found_ids):,}] copiados: {n_copied:,}")

    # ── Verificação pós-cópia ──
    if not args.dry_run:
        print(f"\n>> Verificando labels copiados...")
        for split in SPLITS:
            labels_dir = args.pool_root / split / "labels"
            images_dir = args.pool_root / split / "images"
            n_labels = sum(1 for f in labels_dir.iterdir() if f.suffix == ".txt") if labels_dir.exists() else 0
            n_images = sum(1 for f in images_dir.iterdir() if f.suffix == ".jpg") if images_dir.exists() else 0
            coverage = n_labels / n_images * 100 if n_images > 0 else 0
            marker = "✓" if coverage > 90 else "⚠"
            print(f"   {split}: {n_labels:,} labels / {n_images:,} imagens "
                  f"({coverage:.1f}% cobertura) {marker}")

    # ── Relatório ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "labels_source": str(labels_root),
        "ids_file": str(args.ids_file),
        "pool_root": str(args.pool_root),
        "dry_run": args.dry_run,
        "n_target_ids": len(target_ids),
        "n_found_in_eduardo": len(found_ids),
        "n_missing": len(missing_ids),
        "missing_pct": round(len(missing_ids) / len(target_ids) * 100, 1),
        "n_copied": n_copied,
        "per_split": dict(stats),
        "n_copy_errors": len(copy_errors),
        "copy_errors_sample": copy_errors[:10],
        "missing_ids_sample": sorted(missing_ids)[:50],
    }

    report_path = args.pool_root / "filtrar_labels_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ── Sumário ──
    print(f"\n{'='*72}")
    if args.dry_run:
        print(f"  DRY-RUN — nada copiado")
    else:
        print(f"  FILTRAGEM CONCLUÍDA")
    print(f"{'='*72}")
    print(f"  IDs alvo:          {len(target_ids):,}")
    print(f"  Labels encontrados: {len(found_ids):,} ({len(found_ids)/len(target_ids)*100:.1f}%)")
    print(f"  Labels faltantes:   {len(missing_ids):,} ({len(missing_ids)/len(target_ids)*100:.1f}%)")
    print(f"  Copiados:          {n_copied:,}")
    print(f"  Por split:")
    for split in SPLITS:
        print(f"    {split}: {stats.get(split, 0):,}")
    print(f"    excedente: {stats.get('in_excedente', 0):,}")
    print(f"    sem imagem: {stats.get('no_image', 0):,}")
    print(f"    erros: {stats.get('errors', 0):,}")
    if not args.dry_run:
        print(f"  Relatório: {report_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
