"""
gerar_dataset_copypaste.py (v3 — substituição in-place)

Gera dataset sintético substituindo as embarcações reais do CITRA-3D-Real
por crops do InaTechShips, na MESMA posição e MESMA dimensão dos bboxes
originais.

ABORDAGEM

  Para cada imagem do CITRA-3D-Real:
    1. Lê os N bounding boxes originais (posição e tamanho reais)
    2. Para cada bbox, sorteia um crop aleatório do pool InaTechShips
    3. Redimensiona o crop para caber exatamente no bbox
    4. Cola o crop com alpha blending na posição do bbox
    5. O label YOLO é IDÊNTICO ao original (mesmo x,y,w,h)

  Para gerar ~27k imagens a partir de 2.081 fundos:
    - Gera ~13 variações por imagem (cada uma com crops diferentes)
    - Cada variação tem os mesmos slots mas navios diferentes

VANTAGENS

  - Posição garantida na água (navios reais estavam lá)
  - Escala garantida correta (mesmo tamanho do bbox real)
  - Densidade garantida correta (mesmo número de objetos)
  - Zero heurística de cor, zero detecção de água
  - Labels 100% precisos (reutiliza os originais)
  - Cientificamente defensável: zero decisões arbitrárias de posicionamento

USO

  python gerar_dataset_copypaste.py --n-images 100 --preview   # teste
  python gerar_dataset_copypaste.py                             # batch (~27k)

SAÍDA

  /content/drive/MyDrive/InaTechShips/dataset_sintetico/
  ├── train/{images,labels}/
  ├── val/{images,labels}/
  ├── test/{images,labels}/
  ├── data_single_class.yaml
  └── composicao_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════

DRIVE_BASE = Path("/content/drive/MyDrive")

CROPS_DIR = DRIVE_BASE / "InaTechShips" / "crops_sam"
CROPS_META = CROPS_DIR / "crops_metadata_full.json"
CITRA3D_ROOT = DRIVE_BASE / "PROJETO_MARINHA" / "Datasets" / "CITRA-3D-Real"
OUTPUT_DIR = DRIVE_BASE / "InaTechShips" / "dataset_sintetico"

SEED = 42

# Filtro de qualidade dos crops
MIN_COVERAGE = 0.25
MAX_COVERAGE = 0.95
MIN_SIZE_PX = 50
MIN_AR = 0.2
MAX_AR = 8.0

SPLIT_PROPORTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_filtered_crops(meta_path: Path) -> list[dict]:
    """Carrega crops que passam no filtro de qualidade."""
    with open(meta_path) as f:
        crops = json.load(f).get("crops", [])
    filtered = [c for c in crops
                if MIN_COVERAGE <= c.get("mask_coverage", 0) <= MAX_COVERAGE
                and c.get("crop_size", [0, 0])[0] >= MIN_SIZE_PX
                and c.get("crop_size", [0, 0])[1] >= MIN_SIZE_PX
                and MIN_AR <= c.get("aspect_ratio", 0) <= MAX_AR]
    print(f"   Crops: {len(crops):,} total → {len(filtered):,} após filtro")
    return filtered


def read_yolo_labels(label_path: Path) -> list[list[float]]:
    """Lê labels YOLO. Retorna lista de [class_id, xc, yc, w, h]."""
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    labels.append([float(p) for p in parts[:5]])
                except ValueError:
                    continue
    return labels


def collect_citra3d_images(root: Path) -> list[dict]:
    """Coleta todas as imagens do CITRA-3D-Real com seus labels."""
    pairs = []
    for split in ("train", "val", "test"):
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
            labels = read_yolo_labels(label_path)
            if labels:  # só usa imagens que têm pelo menos 1 bbox
                pairs.append({
                    "image_path": img_path,
                    "label_path": label_path,
                    "image_id": img_path.stem,
                    "split": split,
                    "labels": labels,
                    "n_objects": len(labels),
                })

    return pairs


def compose_one_variation(
    bg_image: np.ndarray,
    labels: list[list[float]],
    crop_pool: list[dict],
    crops_dir: Path,
    rng: random.Random,
    img_size: int = 640,
) -> np.ndarray | None:
    """
    Gera UMA variação de uma imagem do CITRA-3D substituindo cada navio
    por um crop aleatório do InaTechShips, na mesma posição e dimensão.

    Retorna a imagem composta (RGB numpy array) ou None se falhar.
    Os labels são idênticos aos originais (não precisam ser recalculados).
    """
    # Redimensiona fundo para img_size × img_size
    try:
        bg_pil = Image.fromarray(bg_image).resize((img_size, img_size), Image.LANCZOS)
        canvas = np.array(bg_pil)
    except Exception:
        return None

    n_placed = 0

    for label in labels:
        # Label YOLO: [class_id, x_center, y_center, width, height] normalizado
        _, xc, yc, w, h = label

        # Converte para pixels
        bbox_w = max(4, int(w * img_size))
        bbox_h = max(4, int(h * img_size))
        x1 = max(0, int((xc - w / 2) * img_size))
        y1 = max(0, int((yc - h / 2) * img_size))
        x2 = min(img_size, x1 + bbox_w)
        y2 = min(img_size, y1 + bbox_h)

        actual_w = x2 - x1
        actual_h = y2 - y1
        if actual_w < 4 or actual_h < 4:
            continue

        # Sorteia um crop
        crop_meta = rng.choice(crop_pool)
        crop_path = crops_dir / f"{crop_meta['photo_id']}.png"

        try:
            crop_img = Image.open(crop_path).convert("RGBA")
        except Exception:
            continue

        # Redimensiona o crop para caber exatamente no bbox
        crop_resized = crop_img.resize((actual_w, actual_h), Image.LANCZOS)
        crop_arr = np.array(crop_resized)

        # Alpha blending
        alpha = crop_arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = crop_arr[:, :, :3].astype(np.float32)
        region = canvas[y1:y2, x1:x2].astype(np.float32)

        blended = rgb * alpha + region * (1 - alpha)
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        n_placed += 1

    return canvas if n_placed > 0 else None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gerar dataset sintético v3 (substituição in-place)")
    parser.add_argument("--crops-dir", type=Path, default=CROPS_DIR)
    parser.add_argument("--crops-meta", type=Path, default=CROPS_META)
    parser.add_argument("--citra3d-root", type=Path, default=CITRA3D_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-images", type=int, default=None,
                        help="Número total de imagens (default: 27796)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preview", action="store_true",
                        help="Salva 10 previews com bboxes desenhados")
    args = parser.parse_args()

    n_target = args.n_images or 27796

    print("=" * 72)
    print("  Dataset sintético v3 — Substituição In-Place (Passo 4)")
    print("=" * 72)
    print(f"  Crops:     {args.crops_dir}")
    print(f"  CITRA-3D:  {args.citra3d_root}")
    print(f"  Saída:     {args.output}")
    print(f"  N imagens: {n_target:,}")
    print(f"  Img size:  {args.img_size}×{args.img_size}")
    print(f"  Seed:      {args.seed}")
    print("=" * 72)

    # ── Carrega recursos ──
    print(f"\n>> Carregando recursos...")
    crop_pool = load_filtered_crops(args.crops_meta)
    if not crop_pool:
        print("✗ Nenhum crop usável"); sys.exit(1)

    print(f"\n>> Coletando imagens do CITRA-3D-Real...")
    citra_images = collect_citra3d_images(args.citra3d_root)
    n_citra = len(citra_images)
    total_bboxes = sum(img["n_objects"] for img in citra_images)
    print(f"   {n_citra:,} imagens com {total_bboxes:,} bboxes")

    if n_citra == 0:
        print("✗ Nenhuma imagem CITRA-3D encontrada"); sys.exit(1)

    # ── Calcula variações por imagem ──
    n_variations = max(1, n_target // n_citra)
    n_remainder = n_target - (n_variations * n_citra)
    print(f"\n>> Plano: {n_variations} variações × {n_citra:,} imagens "
          f"= {n_variations * n_citra:,}")
    if n_remainder > 0:
        print(f"   + {n_remainder:,} variações extras (primeiras imagens)")

    # ── Calcula split ──
    n_train = int(round(SPLIT_PROPORTIONS["train"] * n_target))
    n_val = int(round(SPLIT_PROPORTIONS["val"] * n_target))
    n_test = n_target - n_train - n_val
    split_plan = {"train": n_train, "val": n_val, "test": n_test}
    print(f"   Split: {' | '.join(f'{s}={n:,}' for s, n in split_plan.items())}")

    for s in split_plan:
        (args.output / s / "images").mkdir(parents=True, exist_ok=True)
        (args.output / s / "labels").mkdir(parents=True, exist_ok=True)

    # ── Gera lista de trabalho (imagem × variação) ──
    rng = random.Random(args.seed)
    work_list = []
    for var_idx in range(n_variations):
        for citra_img in citra_images:
            work_list.append((citra_img, var_idx))

    # Adiciona variações extras para atingir n_target
    if n_remainder > 0:
        extra_images = rng.sample(citra_images, min(n_remainder, n_citra))
        for citra_img in extra_images:
            work_list.append((citra_img, n_variations))

    # Shuffle para misturar variações (evita que todas as var_0 fiquem no train)
    rng.shuffle(work_list)

    # Trunca ao alvo
    work_list = work_list[:n_target]

    # Atribui splits
    split_assignments = []
    idx = 0
    for split, n_split in split_plan.items():
        for _ in range(n_split):
            if idx < len(work_list):
                split_assignments.append(split)
                idx += 1

    # ── Geração ──
    print(f"\n>> Gerando {len(work_list):,} imagens sintéticas...")
    t0 = time.time()
    stats = {s: {"generated": 0, "failed": 0, "objects": 0} for s in split_plan}
    total_gen = 0
    print_every = max(100, len(work_list) // 20)

    # Cache de imagens CITRA-3D (evita reler do Drive repetidamente)
    bg_cache: dict[str, np.ndarray] = {}
    CACHE_MAX = 200  # mantém até 200 imagens em memória

    for i, (citra_img, var_idx) in enumerate(work_list):
        split = split_assignments[i] if i < len(split_assignments) else "train"
        image_id = citra_img["image_id"]

        # Carrega imagem de fundo (com cache)
        if image_id not in bg_cache:
            try:
                bg = np.array(Image.open(citra_img["image_path"]).convert("RGB"))
                if len(bg_cache) < CACHE_MAX:
                    bg_cache[image_id] = bg
            except Exception:
                stats[split]["failed"] += 1
                continue
        else:
            bg = bg_cache[image_id]

        # Compõe variação
        canvas = compose_one_variation(
            bg, citra_img["labels"], crop_pool, args.crops_dir, rng, args.img_size
        )

        if canvas is None:
            stats[split]["failed"] += 1
            continue

        # Salva imagem
        img_name = f"synth_{image_id}_v{var_idx:02d}"
        img_path = args.output / split / "images" / f"{img_name}.jpg"
        Image.fromarray(canvas).save(img_path, "JPEG", quality=90)

        # Salva label (idêntico ao original — mesmas posições)
        label_path = args.output / split / "labels" / f"{img_name}.txt"
        with open(label_path, "w") as f:
            for lbl in citra_img["labels"]:
                # Reescreve como classe 0 (single-class)
                f.write(f"0 {lbl[1]:.6f} {lbl[2]:.6f} "
                        f"{lbl[3]:.6f} {lbl[4]:.6f}\n")

        stats[split]["generated"] += 1
        stats[split]["objects"] += citra_img["n_objects"]
        total_gen += 1

        if (i + 1) % print_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"   [{i+1:>6,}/{len(work_list):,}] "
                  f"gen={total_gen:,} | {rate:.1f} img/s")

    elapsed_total = time.time() - t0

    # ── Preview ──
    if args.preview:
        try:
            import cv2
            preview_dir = args.output / "preview"
            preview_dir.mkdir(exist_ok=True)
            train_imgs = sorted((args.output / "train" / "images").glob("*.jpg"))[:10]
            for img_path in train_imgs:
                img = cv2.imread(str(img_path))
                lbl_path = args.output / "train" / "labels" / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    for line in open(lbl_path):
                        p = line.strip().split()
                        if len(p) >= 5:
                            xc = float(p[1]) * args.img_size
                            yc = float(p[2]) * args.img_size
                            w = float(p[3]) * args.img_size
                            h = float(p[4]) * args.img_size
                            cv2.rectangle(img,
                                          (int(xc - w / 2), int(yc - h / 2)),
                                          (int(xc + w / 2), int(yc + h / 2)),
                                          (0, 255, 0), 2)
                cv2.imwrite(str(preview_dir / img_path.name), img)
            print(f"\n   ✓ {len(train_imgs)} previews em {preview_dir}")
        except ImportError:
            print(f"\n   ⚠ cv2 indisponível para preview")

    # ── data.yaml ──
    yaml_content = f"""# Dataset sintético v3 — Substituição In-Place
# Fundos: CITRA-3D-Real (posições e escalas reais preservadas)
# Navios: crops InaTechShips (SAM segmentation)
# Cada imagem é uma variação de um fundo CITRA-3D com navios diferentes
path: {args.output}
train: train/images
val: val/images
test: test/images
nc: 1
names:
  - embarcacao
"""
    (args.output / "data_single_class.yaml").write_text(yaml_content)

    # ── Relatório ──
    total_objects = sum(s["objects"] for s in stats.values())
    report = {
        "generated_at": datetime.now().isoformat(),
        "version": "v3_in_place_substitution",
        "seed": args.seed,
        "img_size": args.img_size,
        "n_citra_images": n_citra,
        "n_citra_bboxes": total_bboxes,
        "n_variations_per_image": n_variations,
        "n_images_total": total_gen,
        "n_objects_total": total_objects,
        "avg_objects_per_image": round(total_objects / total_gen, 2) if total_gen else 0,
        "splits": stats,
        "crops_pool_size": len(crop_pool),
        "elapsed_seconds": elapsed_total,
        "elapsed_human": f"{elapsed_total / 60:.1f} min",
    }

    report_path = args.output / "composicao_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ── Sumário ──
    print(f"\n{'=' * 72}")
    print(f"  CONCLUÍDO (v3 — substituição in-place)")
    print(f"{'=' * 72}")
    print(f"  Fundos CITRA-3D:    {n_citra:,} imagens × {n_variations} variações")
    print(f"  Imagens geradas:    {total_gen:,}")
    print(f"  Objetos totais:     {total_objects:,}")
    if total_gen > 0:
        print(f"  Obj/imagem médio:   {total_objects / total_gen:.1f}")
    print(f"  Tempo:              {elapsed_total / 60:.1f} min")
    print(f"  Por split:")
    for s, st in stats.items():
        print(f"    {s}: {st['generated']:,} imgs, {st['objects']:,} objs")
    print(f"  Saída:              {args.output}")
    print(f"  Relatório:          {report_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
