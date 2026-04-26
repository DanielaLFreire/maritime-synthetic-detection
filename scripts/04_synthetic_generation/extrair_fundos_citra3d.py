"""
extrair_fundos_citra3d.py

Extrai fundos oceânicos do CITRA-3D-Real para uso no Scale-Aware
Copy-Paste.

CONTEXTO

  Pipeline Scale-Aware Copy-Paste — Passo 3 de 6.
  
  Passos anteriores:
    D1: análise de escala (concluído)
    D2: extração de crops com SAM (concluído — 23.828 crops usáveis)

  Para compor imagens sintéticas realistas, precisamos de fundos que
  representem o cenário operacional do CITRA-3D-Real: oceano aberto,
  condições climáticas variadas, linhas de horizonte reais.

O QUE ESTE SCRIPT FAZ

  Duas estratégias de extração de fundos:

  Estratégia A — Imagens com poucos/pequenos objetos (default):
    Para cada imagem do CITRA-3D-Real, calcula a fração de área coberta
    por bounding boxes. Imagens onde os objetos cobrem < 5% da imagem
    são usadas diretamente como fundos (com os objetos minúsculos — na
    composição, novos navios serão colados por cima, e os originais
    ficam tão pequenos que funcionam como ruído realista).

  Estratégia B — Inpainting simples:
    Para imagens com objetos maiores, preenche as regiões dos bboxes
    com o conteúdo ao redor (interpolação bilinear). Produz fundos
    mais limpos mas pode ter artefatos de inpainting.

  Na prática, a Estratégia A produz ~800-1200 fundos de alta qualidade
  (imagens com 1-2 navios pequenos). A Estratégia B adiciona mais ~800
  fundos processados. O total de ~1500-2000 fundos é suficiente para
  gerar ~27k imagens sintéticas com variação de fundo.

PRÉ-REQUISITOS

  - CITRA-3D-Real acessível (Drive ou disco local)
  - Labels single-class para calcular área dos bboxes
  - opencv-python (para inpainting, estratégia B)

USO

  python extrair_fundos_citra3d.py
  python extrair_fundos_citra3d.py --strategy both     # A + B (default)
  python extrair_fundos_citra3d.py --strategy clean     # só A (sem inpainting)
  python extrair_fundos_citra3d.py --strategy inpaint   # só B

SAÍDA

  /content/drive/MyDrive/InaTechShips/backgrounds_citra3d/
  ├── {image_id}.jpg        (fundos extraídos)
  ├── ...
  └── backgrounds_report.json  (metadados e estatísticas)

TEMPO ESTIMADO

  ~5-10 min (leitura de 2.081 imagens + inpainting simples)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

DRIVE_BASE = Path("/content/drive/MyDrive")

# Tenta vários caminhos do CITRA-3D-Real
CITRA3D_CANDIDATES = [
    DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "CITRA-3D-Real",
    Path("/content/data/CITRA-3D-Real"),
]

BACKGROUNDS_DIR = DRIVE_BASE / "InaTechShips" / "backgrounds_citra3d"

SPLITS = ("train", "val", "test")

# Limiar: imagens onde bboxes cobrem < MAX_BBOX_COVERAGE são usadas
# diretamente como fundo (estratégia A — "clean enough")
MAX_BBOX_COVERAGE = 0.05  # 5% da imagem


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def find_citra3d_root() -> Path | None:
    for p in CITRA3D_CANDIDATES:
        if p.exists() and (p / "train").exists():
            return p
    return None


def read_yolo_bboxes(label_path: Path) -> list[list[float]]:
    """Lê bboxes YOLO de um arquivo de label."""
    bboxes = []
    if not label_path.exists():
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    bboxes.append([float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4])])
                except ValueError:
                    continue
    return bboxes


def compute_bbox_coverage(bboxes: list[list[float]]) -> float:
    """Calcula fração da imagem coberta pelos bboxes (aproximado, sem tratar overlap)."""
    total_area = sum(w * h for _, _, w, h in bboxes)
    return min(total_area, 1.0)  # Clamp a 1.0


def yolo_to_pixel_rect(bbox: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Converte bbox YOLO para (x1, y1, x2, y2) em pixels."""
    x_c, y_c, w, h = bbox
    x1 = max(0, int((x_c - w / 2) * img_w))
    y1 = max(0, int((y_c - h / 2) * img_h))
    x2 = min(img_w, int((x_c + w / 2) * img_w))
    y2 = min(img_h, int((y_c + h / 2) * img_h))
    return x1, y1, x2, y2


def inpaint_bboxes(image: np.ndarray, bboxes: list[list[float]]) -> np.ndarray:
    """
    Remove objetos da imagem preenchendo as regiões dos bboxes.

    Usa cv2.inpaint (Telea algorithm) quando disponível, senão faz
    preenchimento simples com a média local ao redor do bbox.
    """
    img_h, img_w = image.shape[:2]
    result = image.copy()

    try:
        import cv2
        # Cria máscara binária com as regiões dos bboxes
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for bbox in bboxes:
            x1, y1, x2, y2 = yolo_to_pixel_rect(bbox, img_w, img_h)
            # Expande um pouco o bbox para cobrir bordas
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_w, x2 + pad)
            y2 = min(img_h, y2 + pad)
            mask[y1:y2, x1:x2] = 255

        # Inpainting Telea
        result = cv2.inpaint(result, mask, inpaintRadius=10,
                             flags=cv2.INPAINT_TELEA)
    except ImportError:
        # Fallback: preenche com média local
        for bbox in bboxes:
            x1, y1, x2, y2 = yolo_to_pixel_rect(bbox, img_w, img_h)
            # Amostra pixels ao redor do bbox
            margin = 20
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(img_w, x2 + margin)
            y2m = min(img_h, y2 + margin)

            # Máscara dos pixels fora do bbox (região de contexto)
            region = result[y1m:y2m, x1m:x2m]
            local_mask = np.ones(region.shape[:2], dtype=bool)
            # Marca pixels dentro do bbox como False
            inner_y1 = y1 - y1m
            inner_x1 = x1 - x1m
            inner_y2 = y2 - y1m
            inner_x2 = x2 - x1m
            local_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

            if local_mask.any():
                mean_color = region[local_mask].mean(axis=0).astype(np.uint8)
            else:
                mean_color = np.array([128, 128, 128], dtype=np.uint8)

            result[y1:y2, x1:x2] = mean_color

    return result


def collect_images_with_labels(root: Path) -> list[dict]:
    """Coleta todos os pares imagem/label do CITRA-3D-Real (todos os splits)."""
    pairs = []
    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels_single_class"
        if not labels_dir.exists():
            labels_dir = root / split / "labels"

        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            pairs.append({
                "image_path": img_path,
                "label_path": label_path,
                "image_id": img_path.stem,
                "split": split,
            })

    return pairs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Extrair fundos oceânicos do CITRA-3D-Real")
    parser.add_argument("--root", type=Path, default=None,
                        help="Root do CITRA-3D-Real")
    parser.add_argument("--dst", type=Path, default=BACKGROUNDS_DIR,
                        help="Diretório de saída")
    parser.add_argument("--strategy", choices=["clean", "inpaint", "both"],
                        default="both",
                        help="Estratégia de extração (default: both)")
    parser.add_argument("--max-coverage", type=float, default=MAX_BBOX_COVERAGE,
                        help=f"Limiar de cobertura para estratégia clean (default: {MAX_BBOX_COVERAGE})")
    args = parser.parse_args()

    # ── Encontra CITRA-3D-Real ──
    root = args.root or find_citra3d_root()
    if root is None:
        print("✗ CITRA-3D-Real não encontrado. Use --root.")
        sys.exit(1)

    print("=" * 72)
    print("  Extração de fundos oceânicos — Scale-Aware Copy-Paste (Passo 3)")
    print("=" * 72)
    print(f"  Source:      {root}")
    print(f"  Destino:     {args.dst}")
    print(f"  Estratégia:  {args.strategy}")
    print(f"  Max coverage (clean): {args.max_coverage:.1%}")
    print("=" * 72)

    args.dst.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    # ── Coleta imagens ──
    print(f"\n>> Coletando imagens do CITRA-3D-Real...")
    pairs = collect_images_with_labels(root)
    print(f"   Total: {len(pairs):,} imagens")

    # ── Análise de cobertura ──
    print(f"\n>> Analisando cobertura dos bboxes...")
    coverage_data = []
    for pair in pairs:
        bboxes = read_yolo_bboxes(pair["label_path"])
        coverage = compute_bbox_coverage(bboxes) if bboxes else 0.0
        pair["bboxes"] = bboxes
        pair["coverage"] = coverage
        coverage_data.append(coverage)

    # Distribuição
    n_very_clean = sum(1 for c in coverage_data if c < 0.01)
    n_clean = sum(1 for c in coverage_data if c < args.max_coverage)
    n_moderate = sum(1 for c in coverage_data if args.max_coverage <= c < 0.15)
    n_heavy = sum(1 for c in coverage_data if c >= 0.15)

    print(f"   < 1% cobertura (quase vazio):     {n_very_clean:>5,}")
    print(f"   < {args.max_coverage:.0%} cobertura (clean enough):  {n_clean:>5,}")
    print(f"   {args.max_coverage:.0%}-15% cobertura (moderado):    {n_moderate:>5,}")
    print(f"   > 15% cobertura (denso):           {n_heavy:>5,}")

    # ── Extração ──
    print(f"\n>> Extraindo fundos...")
    t0 = time.time()

    backgrounds_clean = []
    backgrounds_inpaint = []
    n_saved = 0

    for i, pair in enumerate(pairs, 1):
        img_id = pair["image_id"]
        coverage = pair["coverage"]
        bboxes = pair["bboxes"]

        try:
            img = np.array(Image.open(pair["image_path"]).convert("RGB"))
        except Exception:
            continue

        saved_as = None

        # Estratégia A: imagens com pouca cobertura → usa direto
        if args.strategy in ("clean", "both") and coverage < args.max_coverage:
            out_path = args.dst / f"{img_id}.jpg"
            Image.fromarray(img).save(out_path, "JPEG", quality=90)
            backgrounds_clean.append({
                "image_id": img_id,
                "strategy": "clean",
                "coverage": round(coverage, 4),
                "size": [img.shape[1], img.shape[0]],
                "split": pair["split"],
            })
            saved_as = "clean"
            n_saved += 1

        # Estratégia B: inpainting para remover objetos
        elif args.strategy in ("inpaint", "both") and bboxes and coverage >= args.max_coverage:
            inpainted = inpaint_bboxes(img, bboxes)
            out_path = args.dst / f"{img_id}.jpg"
            Image.fromarray(inpainted).save(out_path, "JPEG", quality=90)
            backgrounds_inpaint.append({
                "image_id": img_id,
                "strategy": "inpaint",
                "coverage": round(coverage, 4),
                "n_bboxes_removed": len(bboxes),
                "size": [img.shape[1], img.shape[0]],
                "split": pair["split"],
            })
            saved_as = "inpaint"
            n_saved += 1

        if i % 200 == 0:
            print(f"   [{i:>5,}/{len(pairs):,}] salvos: {n_saved:,}")

    elapsed = time.time() - t0
    all_backgrounds = backgrounds_clean + backgrounds_inpaint

    # ── Relatório ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "source": str(root),
        "destination": str(args.dst),
        "strategy": args.strategy,
        "max_coverage_threshold": args.max_coverage,
        "n_images_source": len(pairs),
        "n_backgrounds_total": len(all_backgrounds),
        "n_clean": len(backgrounds_clean),
        "n_inpaint": len(backgrounds_inpaint),
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{elapsed/60:.1f} min",
        "coverage_distribution": {
            "very_clean_lt1pct": n_very_clean,
            "clean_lt5pct": n_clean,
            "moderate_5_15pct": n_moderate,
            "heavy_gt15pct": n_heavy,
        },
        "backgrounds": all_backgrounds,
    }

    report_path = args.dst / "backgrounds_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*72}")
    print(f"  EXTRAÇÃO DE FUNDOS CONCLUÍDA")
    print(f"{'='*72}")
    print(f"  Fundos clean:    {len(backgrounds_clean):,}")
    print(f"  Fundos inpaint:  {len(backgrounds_inpaint):,}")
    print(f"  Total:           {len(all_backgrounds):,}")
    print(f"  Tempo:           {elapsed/60:.1f} min")
    print(f"  Destino:         {args.dst}")
    print(f"  Relatório:       {report_path}")
    print(f"\n  Próximo passo: gerar_dataset_copypaste.py")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
