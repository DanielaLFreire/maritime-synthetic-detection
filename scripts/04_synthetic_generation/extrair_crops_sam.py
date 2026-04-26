"""
extrair_crops_sam.py

Extrai recortes (crops) de embarcações do dataset_25k_v2 usando o
Segment Anything Model (SAM) para gerar máscaras de segmentação limpas.

CONTEXTO

  Pipeline Scale-Aware Copy-Paste — Passo 2 de 6.
  
  O passo 1 (analisar_escala_citra3d.py) determinou que as embarcações
  no CITRA-3D-Real ocupam ~1-10% da imagem (mediana ~3%). Para compor
  imagens sintéticas realistas, precisamos de recortes limpos dos navios
  do InaTechShips separados do fundo original (porto, cais, etc.).

  SAM é usado para gerar uma máscara precisa do navio, guiado pelo
  bounding box do label YOLO como prompt. O resultado é um crop RGBA
  (com canal alpha para transparência) que pode ser colado em qualquer
  fundo sem artefatos de borda visíveis.

O QUE ESTE SCRIPT FAZ

  Para cada imagem do dataset_25k_v2 (ou subset configurável):
    1. Lê o bounding box do label YOLO.
    2. Usa SAM com o bbox como prompt para gerar máscara de segmentação.
    3. Aplica a máscara à imagem original para criar crop RGBA.
    4. Salva o crop como PNG com canal alpha (fundo transparente).
    5. Registra metadados (tamanho original, bbox, área da máscara).

  Modos de operação:
    --mode sam     (default): SAM ViT-B para segmentação precisa (~0.3s/img)
    --mode bbox    (rápido):  crop retangular com feathering nas bordas (~0.01s/img)

  O modo bbox é um fallback rápido que produz resultados aceitáveis
  quando os crops serão redimensionados para ~20-60px (a imprecisão
  da máscara fica invisível nessa escala).

PRÉ-REQUISITOS

  - Google Colab com GPU (SAM requer CUDA)
  - dataset_25k_v2 em /content/drive/MyDrive/InaTechShips/dataset_25k_v2/
  - pip install segment-anything (para modo SAM)
  - Pesos do SAM: sam_vit_b_01ec64.pth (375MB, baixados automaticamente)

USO

  python extrair_crops_sam.py                        # SAM, todos os splits
  python extrair_crops_sam.py --mode bbox            # modo rápido sem SAM
  python extrair_crops_sam.py --split train           # só split train
  python extrair_crops_sam.py --max-images 1000       # subset para teste
  python extrair_crops_sam.py --resume                # retoma de onde parou

SAÍDA

  /content/drive/MyDrive/InaTechShips/crops_sam/
  ├── {photo_id}.png        (crop RGBA com transparência)
  ├── ...
  ├── crops_metadata.json   (metadados de todos os crops)
  └── crops_progress.json   (progresso para retomada)

TEMPO ESTIMADO

  Modo SAM (ViT-B): ~1.5-2h para ~27k imagens no A100
  Modo bbox:         ~5 min para ~27k imagens (CPU)
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
DATASET_25K_V2 = DRIVE_BASE / "InaTechShips" / "dataset_25k_v2"
CROPS_DIR = DRIVE_BASE / "InaTechShips" / "crops_sam"

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

SPLITS = ("train", "val", "test")
FEATHER_RADIUS = 5  # pixels de blending para modo bbox


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def read_yolo_bbox(label_path: Path) -> list[list[float]]:
    """Lê bboxes YOLO de um arquivo de label. Retorna lista de [x_c, y_c, w, h] normalizados."""
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


def yolo_to_pixel_bbox(yolo_bbox: list[float], img_w: int, img_h: int) -> list[int]:
    """Converte bbox YOLO normalizado [x_c, y_c, w, h] para [x1, y1, x2, y2] em pixels."""
    x_c, y_c, w, h = yolo_bbox
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    # Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return [x1, y1, x2, y2]


def create_feathered_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Cria máscara com feathering (gradiente suave) nas bordas."""
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(radius):
        alpha = (i + 1) / (radius + 1)
        # Bordas
        mask[i, :] = np.minimum(mask[i, :], alpha)
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)
    return mask


def crop_with_bbox(image: np.ndarray, bbox_pixel: list[int],
                   feather_radius: int = 5) -> np.ndarray | None:
    """
    Modo bbox: recorta a região do bbox e aplica feathering nas bordas.
    Retorna imagem RGBA (H, W, 4) ou None se bbox inválido.
    """
    x1, y1, x2, y2 = bbox_pixel
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]

    if h < 4 or w < 4:
        return None

    # Cria canal alpha com feathering
    mask = create_feathered_mask(w, h, min(feather_radius, min(w, h) // 2))
    alpha = (mask * 255).astype(np.uint8)

    # RGBA
    if crop.shape[2] == 3:
        rgba = np.dstack([crop, alpha])
    else:
        rgba = crop.copy()
        rgba[:, :, 3] = alpha

    return rgba


def crop_with_sam(image: np.ndarray, bbox_pixel: list[int],
                  predictor) -> np.ndarray | None:
    """
    Modo SAM: usa o predictor SAM com bbox como prompt para obter máscara.
    Retorna imagem RGBA (H, W, 4) ou None se falhar.
    """
    x1, y1, x2, y2 = bbox_pixel
    if x2 <= x1 or y2 <= y1:
        return None

    # SAM espera bbox como np.array([x1, y1, x2, y2])
    input_box = np.array([x1, y1, x2, y2])

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=True,
    )

    # Pega a máscara com maior score
    best_idx = np.argmax(scores)
    mask = masks[best_idx]  # (H, W) bool

    # Crop da região do bbox
    crop_rgb = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        return None

    # Aplica Gaussian blur suave na máscara para bordas mais naturais
    try:
        import cv2
        crop_mask_float = crop_mask.astype(np.float32)
        kernel_size = max(3, min(h, w) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        crop_mask_float = cv2.GaussianBlur(crop_mask_float, (kernel_size, kernel_size), 0)
        alpha = (crop_mask_float * 255).astype(np.uint8)
    except ImportError:
        alpha = (crop_mask.astype(np.uint8) * 255)

    # RGBA
    rgba = np.dstack([crop_rgb, alpha])
    return rgba


def setup_sam(checkpoint_dir: Path):
    """Baixa pesos do SAM (se necessário) e retorna o predictor."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("   Instalando segment-anything...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "segment-anything", "--break-system-packages", "-q"],
                       check=True)
        from segment_anything import sam_model_registry, SamPredictor

    checkpoint_path = checkpoint_dir / SAM_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        print(f"   Baixando pesos SAM ({SAM_MODEL_TYPE})...")
        import urllib.request
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, str(checkpoint_path))
        print(f"   ✓ {checkpoint_path}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Carregando SAM {SAM_MODEL_TYPE} em {device}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"   ✓ SAM pronto")
    return predictor


def collect_image_label_pairs(root: Path, splits: tuple[str, ...],
                              max_images: int | None = None) -> list[dict]:
    """Coleta pares (imagem, label) do dataset_25k_v2."""
    pairs = []
    for split in splits:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"  # 10-class, mas só queremos o bbox
        # Fallback para labels_single_class
        if not labels_dir.exists():
            labels_dir = root / split / "labels_single_class"

        if not images_dir.exists():
            print(f"   ⚠ {images_dir} não existe, pulando")
            continue

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            pairs.append({
                "image_path": img_path,
                "label_path": label_path,
                "photo_id": img_path.stem,
                "split": split,
            })

    if max_images is not None and len(pairs) > max_images:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_images)

    return pairs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Extrair crops de navios com SAM")
    parser.add_argument("--src", type=Path, default=DATASET_25K_V2,
                        help="Root do dataset_25k_v2")
    parser.add_argument("--dst", type=Path, default=CROPS_DIR,
                        help="Diretório de saída dos crops")
    parser.add_argument("--mode", choices=["sam", "bbox"], default="sam",
                        help="Modo de extração (sam=SAM segmentation, bbox=crop simples)")
    parser.add_argument("--split", nargs="+", default=list(SPLITS),
                        help="Splits a processar (default: train val test)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limita o número de imagens (para teste)")
    parser.add_argument("--resume", action="store_true",
                        help="Retoma de onde parou (pula crops já existentes)")
    parser.add_argument("--sam-dir", type=Path,
                        default=Path("/content/sam_weights"),
                        help="Diretório para pesos do SAM")
    args = parser.parse_args()

    print("=" * 72)
    print("  Extração de crops de embarcações — Scale-Aware Copy-Paste (Passo 2)")
    print("=" * 72)
    print(f"  Source:      {args.src}")
    print(f"  Destino:     {args.dst}")
    print(f"  Modo:        {args.mode}")
    print(f"  Splits:      {args.split}")
    print(f"  Max images:  {args.max_images or 'todos'}")
    print(f"  Resume:      {args.resume}")
    print("=" * 72)

    if not args.src.exists():
        print(f"\n✗ Source não encontrado: {args.src}")
        sys.exit(1)

    # ── Setup ──
    args.dst.mkdir(parents=True, exist_ok=True)
    progress_file = args.dst / "crops_progress.json"
    metadata_file = args.dst / "crops_metadata.json"

    # Carrega progresso anterior (se resume)
    processed_ids = set()
    if args.resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        processed_ids = set(progress.get("processed_ids", []))
        print(f"\n>> Retomada: {len(processed_ids):,} crops já processados")

    # Setup SAM (se modo sam)
    predictor = None
    if args.mode == "sam":
        print(f"\n>> Configurando SAM...")
        predictor = setup_sam(args.sam_dir)

    # ── Coleta pares imagem/label ──
    print(f"\n>> Coletando imagens do dataset_25k_v2...")
    pairs = collect_image_label_pairs(args.src, tuple(args.split), args.max_images)
    print(f"   Total: {len(pairs):,} pares imagem/label")

    # Filtra já processados
    if args.resume:
        pairs = [p for p in pairs if p["photo_id"] not in processed_ids]
        print(f"   A processar: {len(pairs):,} (após filtrar já feitos)")

    if not pairs:
        print("\n   Nada a processar. Todos os crops já existem.")
        return

    # ── Processamento ──
    print(f"\n>> Processando crops ({args.mode})...")

    from PIL import Image
    import io

    all_metadata = []
    n_ok = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    save_every = 500  # salva progresso a cada N imagens
    print_every = max(100, len(pairs) // 20)

    for i, pair in enumerate(pairs, 1):
        photo_id = pair["photo_id"]
        img_path = pair["image_path"]
        label_path = pair["label_path"]

        try:
            # Lê imagem
            img = np.array(Image.open(img_path).convert("RGB"))
            img_h, img_w = img.shape[:2]

            # Lê bboxes
            bboxes = read_yolo_bbox(label_path)
            if not bboxes:
                n_skip += 1
                continue

            # Processa o primeiro bbox (InaTechShips tem 1 bbox/imagem)
            bbox_yolo = bboxes[0]
            bbox_pixel = yolo_to_pixel_bbox(bbox_yolo, img_w, img_h)

            # Extrai crop
            if args.mode == "sam" and predictor is not None:
                crop_rgba = crop_with_sam(img, bbox_pixel, predictor)
            else:
                crop_rgba = crop_with_bbox(img, bbox_pixel, FEATHER_RADIUS)

            if crop_rgba is None:
                n_skip += 1
                continue

            # Salva como PNG (com canal alpha)
            crop_img = Image.fromarray(crop_rgba, "RGBA")
            out_path = args.dst / f"{photo_id}.png"
            crop_img.save(out_path, "PNG", optimize=True)

            # Metadados
            crop_h, crop_w = crop_rgba.shape[:2]
            alpha_area = int(np.sum(crop_rgba[:, :, 3] > 128))
            total_area = crop_h * crop_w

            meta = {
                "photo_id": photo_id,
                "split": pair["split"],
                "original_size": [img_w, img_h],
                "bbox_yolo": bbox_yolo,
                "bbox_pixel": bbox_pixel,
                "crop_size": [crop_w, crop_h],
                "mask_area_px": alpha_area,
                "mask_coverage": round(alpha_area / total_area, 4) if total_area > 0 else 0,
                "aspect_ratio": round(crop_w / crop_h, 4) if crop_h > 0 else 0,
            }
            all_metadata.append(meta)
            processed_ids.add(photo_id)
            n_ok += 1

        except Exception as exc:
            n_fail += 1
            if n_fail <= 5:
                print(f"   ⚠ Erro em {photo_id}: {exc}")

        # Progresso
        if i % print_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pairs) - i) / rate if rate > 0 else 0
            print(f"   [{i:>6,}/{len(pairs):,}] ok={n_ok:,} skip={n_skip:,} "
                  f"fail={n_fail:,} | {rate:.1f}/s ETA {eta/60:.0f}min")

        # Salva progresso periodicamente
        if i % save_every == 0:
            progress_data = {
                "processed_ids": list(processed_ids),
                "n_ok": n_ok,
                "n_skip": n_skip,
                "n_fail": n_fail,
                "last_saved_at": datetime.now().isoformat(),
            }
            progress_file.write_text(json.dumps(progress_data))

    elapsed_total = time.time() - t0

    # ── Salva metadados finais ──
    print(f"\n>> Salvando metadados...")

    # Carrega metadados anteriores se resume
    existing_metadata = []
    if args.resume and metadata_file.exists():
        with open(metadata_file) as f:
            existing_metadata = json.load(f).get("crops", [])
        existing_ids = {m["photo_id"] for m in existing_metadata}
        # Merge: mantém existentes + adiciona novos
        for m in all_metadata:
            if m["photo_id"] not in existing_ids:
                existing_metadata.append(m)
        all_metadata = existing_metadata

    metadata_report = {
        "generated_at": datetime.now().isoformat(),
        "mode": args.mode,
        "source": str(args.src),
        "destination": str(args.dst),
        "n_crops_total": len(all_metadata),
        "n_processed_this_run": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "elapsed_seconds": elapsed_total,
        "elapsed_human": f"{elapsed_total/60:.1f} min",
        "scale_stats": {},
        "crops": all_metadata,
    }

    # Estatísticas de escala dos crops
    if all_metadata:
        widths = [m["crop_size"][0] for m in all_metadata]
        heights = [m["crop_size"][1] for m in all_metadata]
        ars = [m["aspect_ratio"] for m in all_metadata]
        coverages = [m["mask_coverage"] for m in all_metadata]

        import statistics as st
        metadata_report["scale_stats"] = {
            "crop_width_px": {
                "mean": round(st.mean(widths), 1),
                "median": round(st.median(widths), 1),
                "min": min(widths),
                "max": max(widths),
            },
            "crop_height_px": {
                "mean": round(st.mean(heights), 1),
                "median": round(st.median(heights), 1),
                "min": min(heights),
                "max": max(heights),
            },
            "aspect_ratio": {
                "mean": round(st.mean(ars), 4),
                "median": round(st.median(ars), 4),
            },
            "mask_coverage": {
                "mean": round(st.mean(coverages), 4),
                "median": round(st.median(coverages), 4),
                "note": "Fração da área do crop coberta pela máscara (1.0 = bbox cheio)",
            },
        }

    metadata_file.write_text(json.dumps(metadata_report, indent=2, ensure_ascii=False))

    # Salva progresso final
    progress_data = {
        "processed_ids": list(processed_ids),
        "n_ok": len([m for m in all_metadata]),
        "n_skip": n_skip,
        "n_fail": n_fail,
        "last_saved_at": datetime.now().isoformat(),
        "completed": True,
    }
    progress_file.write_text(json.dumps(progress_data))

    # ── Relatório final ──
    print(f"\n{'='*72}")
    print(f"  EXTRAÇÃO DE CROPS CONCLUÍDA")
    print(f"{'='*72}")
    print(f"  Modo:           {args.mode}")
    print(f"  Crops gerados:  {n_ok:,}")
    print(f"  Pulados:        {n_skip:,}")
    print(f"  Falhas:         {n_fail:,}")
    print(f"  Tempo:          {elapsed_total/60:.1f} min")
    if n_ok > 0:
        print(f"  Taxa:           {n_ok/elapsed_total:.1f} crops/s")
    print(f"  Destino:        {args.dst}")
    print(f"  Metadados:      {metadata_file}")
    if all_metadata and "scale_stats" in metadata_report:
        ss = metadata_report["scale_stats"]
        print(f"\n  Estatísticas dos crops:")
        print(f"    Largura:    {ss['crop_width_px']['median']:.0f} px (mediana)")
        print(f"    Altura:     {ss['crop_height_px']['median']:.0f} px (mediana)")
        print(f"    AR:         {ss['aspect_ratio']['median']:.2f} (mediana)")
        print(f"    Cobertura:  {ss['mask_coverage']['median']:.1%} (mediana)")
    print(f"\n  Próximo passo: extrair_fundos_citra3d.py")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
