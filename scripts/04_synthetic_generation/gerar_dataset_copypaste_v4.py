"""
gerar_dataset_copypaste_v4.py

Versão corrigida que respeita os splits originais do CITRA-3D-Real.
Cada split sintético é gerado APENAS a partir do split correspondente:
  - synthetic/train ← fundos de CITRA-3D/train
  - synthetic/val   ← fundos de CITRA-3D/val
  - synthetic/test  ← fundos de CITRA-3D/test

Isso elimina o data leakage: o modelo nunca vê fundos do test set
durante o pré-treino.

USO:
  python gerar_dataset_copypaste_v4.py

CONFIGURAÇÃO (edite os caminhos abaixo):
"""

from pathlib import Path
from PIL import Image
import numpy as np
import json
import time
import random
import shutil

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════

CITRA_ROOT = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real")
CROPS_DIR = Path("/content/drive/MyDrive/InaTechShips/crops_sam")
CROPS_META = CROPS_DIR / "crops_metadata_full.json"
OUTPUT_DIR = Path("/content/drive/MyDrive/InaTechShips/dataset_sintetico_v4")
N_VARIATIONS = 13
SEED = 42
IMG_SIZE = 640

# ═══════════════════════════════════════════════════════════════════
# FUNÇÕES
# ═══════════════════════════════════════════════════════════════════

def load_crops_pool(crops_dir, metadata_path):
    """Carrega pool de crops filtrados."""
    with open(metadata_path) as f:
        meta = json.load(f)
    
    pool = []
    for crop_id, info in meta.items():
        crop_path = crops_dir / info.get("filename", f"{crop_id}.png")
        if crop_path.exists() and info.get("passed_filter", True):
            pool.append(crop_path)
    
    print(f"  Pool de crops: {len(pool):,}")
    return pool


def read_labels(label_path):
    """Lê labels YOLO (classe x_center y_center width height)."""
    bboxes = []
    if label_path.exists():
        for line in open(label_path):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append((cls, xc, yc, w, h))
    return bboxes


def compose_image(bg_path, bboxes, crops_pool, rng, img_size=640):
    """
    Compõe uma imagem sintética por substituição in-place.
    Para cada bbox, seleciona um crop aleatório, redimensiona para
    o tamanho do bbox, e cola via alpha blending.
    """
    bg = Image.open(bg_path).convert("RGB").resize((img_size, img_size))
    bg_arr = np.array(bg, dtype=np.float32)
    
    n_placed = 0
    for cls, xc, yc, w, h in bboxes:
        # Converte normalized → pixels
        px_w = max(4, int(w * img_size))
        px_h = max(4, int(h * img_size))
        px_x = int((xc - w/2) * img_size)
        px_y = int((yc - h/2) * img_size)
        
        # Clipa aos limites da imagem
        px_x = max(0, min(px_x, img_size - px_w))
        px_y = max(0, min(px_y, img_size - px_h))
        
        # Seleciona crop aleatório
        crop_path = rng.choice(crops_pool)
        try:
            crop = Image.open(crop_path).convert("RGBA").resize((px_w, px_h))
        except Exception:
            continue
        
        crop_arr = np.array(crop, dtype=np.float32)
        alpha = crop_arr[:, :, 3:4] / 255.0
        rgb = crop_arr[:, :, :3]
        
        # Ajusta tamanho se necessário
        actual_h = min(px_h, img_size - px_y)
        actual_w = min(px_w, img_size - px_x)
        if actual_h < px_h or actual_w < px_w:
            alpha = alpha[:actual_h, :actual_w]
            rgb = rgb[:actual_h, :actual_w]
        
        # Alpha blending
        region = bg_arr[px_y:px_y+actual_h, px_x:px_x+actual_w]
        bg_arr[px_y:px_y+actual_h, px_x:px_x+actual_w] = (
            rgb[:actual_h, :actual_w] * alpha[:actual_h, :actual_w] +
            region * (1 - alpha[:actual_h, :actual_w])
        )
        n_placed += 1
    
    result = Image.fromarray(bg_arr.astype(np.uint8))
    return result, n_placed


def process_split(split_name, citra_root, crops_pool, output_dir, 
                  n_variations, rng, img_size=640):
    """
    Gera variações sintéticas para UM split, usando apenas
    imagens desse split como fundo.
    """
    img_dir = citra_root / split_name / "images"
    # Tenta labels_single_class primeiro, depois labels
    lbl_dir = citra_root / split_name / "labels_single_class"
    if not lbl_dir.exists():
        lbl_dir = citra_root / split_name / "labels"
    
    out_img = output_dir / split_name / "images"
    out_lbl = output_dir / split_name / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"\n  Split '{split_name}': {len(images)} imagens fonte")
    
    n_generated = 0
    n_objects = 0
    n_failed = 0
    
    for img_path in images:
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.txt"
        bboxes = read_labels(label_path)
        
        if not bboxes:
            continue
        
        for var_idx in range(n_variations):
            out_name = f"synth_{stem}_v{var_idx:02d}"
            out_img_path = out_img / f"{out_name}.jpg"
            out_lbl_path = out_lbl / f"{out_name}.txt"
            
            # Skip se já existe (retomada)
            if out_img_path.exists() and out_lbl_path.exists():
                n_generated += 1
                n_objects += len(bboxes)
                continue
            
            try:
                result, placed = compose_image(
                    img_path, bboxes, crops_pool, rng, img_size
                )
                result.save(out_img_path, quality=95)
                
                # Labels idênticos aos originais
                with open(out_lbl_path, "w") as f:
                    for cls, xc, yc, w, h in bboxes:
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
                n_generated += 1
                n_objects += placed
            except Exception as e:
                n_failed += 1
                print(f"    ERRO: {out_name}: {e}")
    
    print(f"    Geradas: {n_generated:,} | Objetos: {n_objects:,} | Falhas: {n_failed}")
    return {"generated": n_generated, "objects": n_objects, "failed": n_failed}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  GERAÇÃO DE DATASET SINTÉTICO v4 (splits isolados)")
    print("=" * 70)
    
    # Limpa saída anterior se existir
    if OUTPUT_DIR.exists():
        print(f"  Removendo {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    rng = random.Random(SEED)
    np.random.seed(SEED)
    
    # Carrega crops
    print("\n  Carregando pool de crops...")
    crops_pool = load_crops_pool(CROPS_DIR, CROPS_META)
    
    t0 = time.time()
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "v4_split_isolated",
        "seed": SEED,
        "img_size": IMG_SIZE,
        "n_variations_per_image": N_VARIATIONS,
        "crops_pool_size": len(crops_pool),
        "data_leakage_fix": "Each synthetic split generated ONLY from "
                            "the corresponding CITRA-3D-Real split. "
                            "No test backgrounds in synthetic train.",
        "splits": {}
    }
    
    total_imgs = 0
    total_objs = 0
    
    for split in ("train", "val", "test"):
        result = process_split(
            split, CITRA_ROOT, crops_pool, OUTPUT_DIR,
            N_VARIATIONS, rng, IMG_SIZE
        )
        report["splits"][split] = result
        total_imgs += result["generated"]
        total_objs += result["objects"]
    
    elapsed = time.time() - t0
    report["n_images_total"] = total_imgs
    report["n_objects_total"] = total_objs
    report["avg_objects_per_image"] = round(total_objs / max(total_imgs, 1), 2)
    report["elapsed_seconds"] = elapsed
    report["elapsed_human"] = f"{elapsed/60:.1f} min"
    
    # Verifica que splits estão corretos
    print(f"\n{'=' * 70}")
    print(f"  CONCLUÍDO (v4 — splits isolados)")
    print(f"{'=' * 70}")
    print(f"  Imagens geradas:    {total_imgs:,}")
    print(f"  Objetos totais:     {total_objs:,}")
    print(f"  Obj/imagem médio:   {report['avg_objects_per_image']}")
    print(f"  Tempo:              {report['elapsed_human']}")
    print(f"\n  Por split (cada um gerado do split CORRESPONDENTE):")
    for split, r in report["splits"].items():
        print(f"    {split}: {r['generated']:,} imgs, {r['objects']:,} objs")
    
    # Verifica proporções
    if total_imgs > 0:
        for split in ("train", "val", "test"):
            pct = report["splits"][split]["generated"] / total_imgs * 100
            print(f"    {split}: {pct:.1f}%")
    
    # Salva relatório
    report_path = OUTPUT_DIR / "composicao_report_v4.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Relatório: {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
