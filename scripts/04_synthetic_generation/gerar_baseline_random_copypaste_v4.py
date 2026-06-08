"""
gerar_baseline_random_copypaste_v4.py

Gera o baseline "context-aware random copy-paste" para comparação com in-place.

═══════════════════════════════════════════════════════════════════
NOVIDADE v4:
  Floor mínimo do horizonte (HORIZON_MIN_PCT). Em v3 o horizonte
  era puramente adaptativo (topo da bbox real mais alta), mas
  algumas imagens do CITRA têm bboxes labeled muito altas na
  imagem (ships cargo distantes na linha de nuvens), o que
  liberava placement em quase todo o topo da imagem.

  v4 introduz um piso: o sea_mask nunca começa acima de
  HORIZON_MIN_PCT da altura (default 35%). Assim, mesmo se há
  uma label real muito alta na imagem, os crops sintéticos
  ficam restritos à porção inferior da imagem.

  Também adiciona um SEGUNDO grid de debug
  (smoke_test_seamask.jpg) mostrando a região permitida em
  vermelho transparente, para validação visual direta do que
  o algoritmo está liberando.

NOVIDADE v3:
  Horizonte adaptativo via topo do bbox real mais alto.

NOVIDADE v2:
  Inpainting TELEA dos objetos reais antes do compositing.
═══════════════════════════════════════════════════════════════════

DIFERENÇA vs IN-PLACE:
  - In-place: crop é colado NA POSIÇÃO e NO TAMANHO do bbox real anotado
  - Random:   crop é colado em POSIÇÃO ALEATÓRIA no mar (abaixo do
              horizonte adaptativo), com TAMANHO AMOSTRADO da distribuição
              real + objetos reais removidos via inpainting TELEA

TUDO o mais é idêntico:
  - Mesmos crops SAM
  - Mesmos fundos (imagens CITRA-3D-Real, agora SEM objetos reais)
  - Mesma densidade (mesmo nº de objetos por imagem)
  - Mesmo nº de variações (13)
  - Mesma separação de splits (train gera train, val gera val)

Uso:
    python gerar_baseline_random_copypaste_v3.py

Para validação visual antes da geração completa, defina
SMOKE_TEST = True abaixo. Gera apenas 3 imagens × 1 variation e
salva um grid de inspeção em OUTPUT_DIR/smoke_test_inpaint.jpg
(colunas: original | inpainted | random copy-paste).
"""

from pathlib import Path
import numpy as np
import cv2
import json
import random
import time
from datetime import datetime

# ═══ CONFIGURAÇÃO ═══
CITRA_ROOT = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real")
CROPS_DIR  = Path("/content/drive/MyDrive/InaTechShips/crops_sam")
OUTPUT_DIR = Path("/content/drive/MyDrive/InaTechShips/dataset_random_copypaste_v4")

N_VARIATIONS     = 13
SEED             = 42
IOU_THRESHOLD    = 0.2
SEA_COVERAGE_MIN = 0.75
MAX_ATTEMPTS     = 100

# ── Inpainting dos objetos reais (novo em v2) ──────────────────────
INPAINT_RADIUS   = 3      # cv2.inpaint TELEA — raio em pixels
INPAINT_PADDING  = 2      # padding extra ao redor da bbox na máscara

# ── Horizonte adaptativo (novo em v3, refinado em v4) ──────────────
# Margem em pixels acima do topo do bbox real mais alto.
HORIZON_MARGIN   = 0
# Fallback: se imagem não tem bboxes reais, usa este percentual da
# altura como divisor céu/mar.
HORIZON_FALLBACK_PCT = 0.5
# Floor: o sea_mask NUNCA começa acima de HORIZON_MIN_PCT da altura,
# mesmo se houver uma bbox real labeled muito alta na imagem
# (proteção contra outliers nas labels — câmeras costeiras altas
# tipicamente têm céu ocupando ≥35% da altura da imagem).
HORIZON_MIN_PCT      = 0.35

# ── Smoke test (validação visual antes da geração completa) ────────
SMOKE_TEST       = False  # True = roda apenas 3 imagens + grid de inspeção

random.seed(SEED)
np.random.seed(SEED)


# ═══ FUNÇÕES ═══

def compute_sea_mask(img_np, bboxes_px=None,
                     horizon_margin=HORIZON_MARGIN,
                     fallback_pct=HORIZON_FALLBACK_PCT,
                     min_pct=HORIZON_MIN_PCT):
    """Detecta região de mar usando horizonte adaptativo + floor mínimo.

    1. Se bboxes_px fornecido: topo do navio mais alto define horizonte
       candidato. Se bboxes vazias: usa fallback_pct.
    2. Floor: o horizonte é forçado a ser pelo menos min_pct da altura
       (proteção contra outliers nas labels).
    3. Refinamento HSV abaixo do horizonte (exclui obstáculos verticais).
    """
    h, w = img_np.shape[:2]

    # ── Limiar vertical (horizonte adaptativo + floor) ──
    if bboxes_px:
        top_y = min(y1 for (x1, y1, x2, y2) in bboxes_px)
        top_y = max(0, top_y - horizon_margin)
    else:
        top_y = int(h * fallback_pct)

    # Floor: jamais permite horizonte acima de min_pct da altura
    floor_y = int(h * min_pct)
    top_y = max(top_y, floor_y)

    pos_mask = np.zeros((h, w), dtype=np.uint8)
    pos_mask[top_y:, :] = 255

    # ── Refinamento HSV (exclui obstáculos verticais abaixo do horizonte) ──
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 20, 30), (140, 255, 255))
    gray = cv2.inRange(hsv, (0, 0, 60), (180, 50, 200))
    color_mask = cv2.bitwise_or(blue, gray)
    combined = cv2.bitwise_and(color_mask, pos_mask)

    # Fallback: se cor detectou pouco, usa só posição
    pos_area = pos_mask.sum() / 255
    if pos_area > 0 and combined.sum() / 255 < pos_area * 0.2:
        combined = pos_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)


def iou(a, b):
    """IoU entre (x1,y1,x2,y2)."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def load_real_bboxes_px(lbl_path, img_h, img_w):
    """Lê labels YOLO normalizadas e devolve lista de bboxes em pixel
    (x1, y1, x2, y2). Usado para construir a máscara de inpainting
    e para contar densidade."""
    bboxes = []
    if not lbl_path.exists():
        return bboxes
    for line in open(lbl_path):
        p = line.strip().split()
        if len(p) < 5:
            continue
        cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        bw = w * img_w
        bh = h * img_h
        x1 = int(round(cx * img_w - bw / 2))
        y1 = int(round(cy * img_h - bh / 2))
        x2 = int(round(x1 + bw))
        y2 = int(round(y1 + bh))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 > x1 and y2 > y1:
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def inpaint_real_objects(img, bboxes_px,
                         radius=INPAINT_RADIUS, padding=INPAINT_PADDING):
    """Remove os objetos reais via inpainting TELEA. Necessário no random
    copy-paste para que o navio real não fique visível como falso negativo
    quando o crop sintético é colado em outra posição.

    Returns: imagem (BGR) com objetos reais removidos."""
    if not bboxes_px:
        return img
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes_px:
        xx1 = max(0, x1 - padding)
        yy1 = max(0, y1 - padding)
        xx2 = min(w, x2 + padding)
        yy2 = min(h, y2 + padding)
        mask[yy1:yy2, xx1:xx2] = 255
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def load_wh_distribution(labels_dir):
    """Carrega (w_norm, h_norm) de todos os bboxes reais."""
    wh = []
    for lbl in labels_dir.glob("*.txt"):
        for line in open(lbl):
            p = line.strip().split()
            if len(p) >= 5:
                wh.append((float(p[3]), float(p[4])))
    return wh


def generate_variations(img_path, lbl_path, crops, wh_dist):
    """Gera N_VARIATIONS imagens com crops em posições aleatórias no mar.
    Aplica inpainting TELEA nos objetos reais ANTES do compositing para
    evitar que o navio real fique visível sem anotação.
    Usa horizonte adaptativo (topo do bbox real mais alto) para
    restringir placement ao espaço abaixo do horizonte."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    h, w = img.shape[:2]

    # Lê bboxes reais ANTES de calcular sea_mask (v3: usado para horizonte)
    real_bboxes_px = load_real_bboxes_px(lbl_path, h, w)
    n_objects = len(real_bboxes_px)
    if n_objects == 0:
        return []

    # Sea mask agora usa as bboxes para definir horizonte adaptativo
    sea_mask = compute_sea_mask(img, bboxes_px=real_bboxes_px)

    # Inpaint UMA VEZ por imagem (determinístico, reusa em todas as variations)
    img_clean = inpaint_real_objects(img, real_bboxes_px)

    results = []
    for var in range(N_VARIATIONS):
        synth = img_clean.copy()   # base é a imagem SEM os objetos reais
        labels = []
        boxes = []

        for _ in range(n_objects):
            # Tamanho: amostrado da distribuição real (NÃO do bbox específico)
            bw_n, bh_n = random.choice(wh_dist)
            bw = max(4, int(bw_n * w))
            bh = max(4, int(bh_n * h))

            # Posição: aleatória na região de mar
            placed = False
            for _ in range(MAX_ATTEMPTS):
                cx = random.randint(bw//2, w - bw//2 - 1)
                cy = random.randint(bh//2, h - bh//2 - 1)
                x1, y1 = cx - bw//2, cy - bh//2
                x2, y2 = x1 + bw, y1 + bh

                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                # ≥75% da caixa no mar
                roi = sea_mask[y1:y2, x1:x2]
                if roi.mean() / 255.0 < SEA_COVERAGE_MIN:
                    continue

                # IoU < 0.2 com caixas já colocadas
                if any(iou((x1,y1,x2,y2), b) > IOU_THRESHOLD for b in boxes):
                    continue

                placed = True
                break

            if not placed:
                continue

            # Cola crop
            crop = cv2.imread(str(random.choice(crops)), cv2.IMREAD_UNCHANGED)
            if crop is None:
                continue

            crop_r = cv2.resize(crop, (bw, bh))
            if crop_r.shape[2] == 4:
                a = crop_r[:,:,3:4].astype(np.float32) / 255.0
                blended = (crop_r[:,:,:3].astype(np.float32) * a +
                          synth[y1:y2, x1:x2].astype(np.float32) * (1-a))
                synth[y1:y2, x1:x2] = blended.astype(np.uint8)
            else:
                synth[y1:y2, x1:x2] = crop_r[:,:,:3]

            boxes.append((x1, y1, x2, y2))
            labels.append(f"0 {(x1+bw/2)/w:.6f} {(y1+bh/2)/h:.6f} {bw_n:.6f} {bh_n:.6f}")

        if labels:
            results.append((synth, labels, var))

    return results


def overlay_sea_mask(img, sea_mask, alpha=0.4):
    """Sobrepõe o sea_mask em vermelho transparente sobre a imagem.
    Útil para validação visual: regiões vermelhas = onde o algoritmo
    permite placement; regiões sem tinta = forçadas como céu/terra."""
    overlay = img.copy().astype(np.float32)
    red_tint = np.array([0, 0, 200], dtype=np.float32)   # BGR vermelho
    mask3 = (sea_mask > 0)[:, :, None].astype(np.float32)
    overlay = overlay * (1 - alpha * mask3) + red_tint * (alpha * mask3)
    return overlay.clip(0, 255).astype(np.uint8)


# ═══ SMOKE TEST (validação visual) ═══

def run_smoke_test(crops, n_samples=3):
    """Gera 1 variation de n_samples imagens train e salva DOIS grids:

    1. smoke_test_inpaint.jpg — 3 colunas: original | inpainted | random
    2. smoke_test_seamask.jpg — 3 colunas: original | sea_mask overlay |
       random com sea_mask overlay (mostra o que o algoritmo permite)

    Imprime também top_y do horizonte calculado para cada imagem
    (debug numérico)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs_train = sorted((CITRA_ROOT / "train" / "images").glob("*"))[:n_samples]
    lbls_train = CITRA_ROOT / "train" / "labels_single_class"
    wh_dist    = load_wh_distribution(lbls_train)

    if not imgs_train:
        print("⚠️  Nenhuma imagem encontrada em CITRA-3D-Real/train/images")
        return

    thumbs_main = []
    thumbs_dbg  = []
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│ Debug numérico do horizonte                              │")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│ HORIZON_MIN_PCT (floor):     {HORIZON_MIN_PCT:.2f}                       │")
    print(f"│ HORIZON_MARGIN (px):         {HORIZON_MARGIN}                          │")
    print("└──────────────────────────────────────────────────────────┘")

    for img_path in imgs_train:
        lbl_path = lbls_train / f"{img_path.stem}.txt"

        img_orig = cv2.imread(str(img_path))
        h, w = img_orig.shape[:2]
        bboxes_px = load_real_bboxes_px(lbl_path, h, w)
        img_inpainted = inpaint_real_objects(img_orig, bboxes_px)
        sea_mask = compute_sea_mask(img_orig, bboxes_px=bboxes_px)

        # Debug numérico
        if bboxes_px:
            top_bbox = min(y1 for (_, y1, _, _) in bboxes_px)
        else:
            top_bbox = None
        floor_y = int(h * HORIZON_MIN_PCT)
        # Detecta onde sea_mask realmente começa
        mask_top = next((y for y in range(h) if sea_mask[y].any()), h)
        print(f"  {img_path.name}: h={h}, "
              f"topo bbox real={top_bbox}, "
              f"floor={floor_y}, "
              f"sea_mask começa em y={mask_top} ({mask_top/h:.0%})")

        # Generate variation
        results = generate_variations(img_path, lbl_path, crops, wh_dist)
        img_with_crops = results[0][0] if results else img_inpainted

        # Grid principal
        thumb_main = np.hstack([
            cv2.resize(img_orig,       (480, 270)),
            cv2.resize(img_inpainted,  (480, 270)),
            cv2.resize(img_with_crops, (480, 270)),
        ])
        thumbs_main.append(thumb_main)

        # Grid de debug com sea_mask overlay
        thumb_dbg = np.hstack([
            cv2.resize(img_orig,                                   (480, 270)),
            cv2.resize(overlay_sea_mask(img_orig, sea_mask),       (480, 270)),
            cv2.resize(overlay_sea_mask(img_with_crops, sea_mask), (480, 270)),
        ])
        thumbs_dbg.append(thumb_dbg)

    out_main = OUTPUT_DIR / "smoke_test_inpaint.jpg"
    out_dbg  = OUTPUT_DIR / "smoke_test_seamask.jpg"
    cv2.imwrite(str(out_main), np.vstack(thumbs_main), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out_dbg),  np.vstack(thumbs_dbg),  [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n✓ Grids salvos:")
    print(f"   {out_main}")
    print(f"     Colunas: original | inpainted | random copy-paste")
    print(f"   {out_dbg}")
    print(f"     Colunas: original | sea_mask em vermelho | random + sea_mask")
    print(f"     (regiões em vermelho = permitidas para placement)")


# ═══ MAIN ═══

def main():
    t0 = time.time()
    crops = sorted(CROPS_DIR.glob("*.png"))
    if not crops:
        crops = sorted(CROPS_DIR.glob("*.jpg"))
    print(f"Crops SAM: {len(crops)}")

    if not crops:
        raise FileNotFoundError(
            f"Nenhum crop encontrado em {CROPS_DIR}. "
            f"Verifique se o pool SAM foi gerado.")

    # ── Smoke test ───────────────────────────────────────────────
    if SMOKE_TEST:
        print("\n🔥 Modo SMOKE_TEST ativado — gerando 3 imagens de inspeção.\n")
        run_smoke_test(crops, n_samples=3)
        print("\n→ Defina SMOKE_TEST = False e rode novamente para gerar tudo.")
        return

    # ── Geração completa ────────────────────────────────────────
    stats = {"seed": SEED, "n_variations": N_VARIATIONS, "n_crops": len(crops),
             "iou_threshold": IOU_THRESHOLD, "sea_coverage_min": SEA_COVERAGE_MIN,
             "inpaint_radius": INPAINT_RADIUS,
             "inpaint_padding": INPAINT_PADDING,
             "horizon_margin": HORIZON_MARGIN,
             "horizon_fallback_pct": HORIZON_FALLBACK_PCT,
             "horizon_min_pct": HORIZON_MIN_PCT,
             "generated_at": datetime.now().isoformat(), "splits": {}}

    for split in ["train", "val"]:
        imgs = CITRA_ROOT / split / "images"
        lbls = CITRA_ROOT / split / "labels_single_class"
        out_i = OUTPUT_DIR / split / "images"
        out_l = OUTPUT_DIR / split / "labels"
        out_i.mkdir(parents=True, exist_ok=True)
        out_l.mkdir(parents=True, exist_ok=True)

        wh_dist = load_wh_distribution(lbls)
        img_list = sorted(imgs.glob("*"))
        print(f"\n{split}: {len(img_list)} imgs, {len(wh_dist)} bboxes na distribuição")

        n_imgs, n_objs = 0, 0
        for i, img_path in enumerate(img_list):
            lbl_path = lbls / f"{img_path.stem}.txt"
            for synth, labels, var in generate_variations(img_path, lbl_path, crops, wh_dist):
                name = f"rcp_{img_path.stem}_v{var:02d}"
                cv2.imwrite(str(out_i / f"{name}.jpg"), synth, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out_l / f"{name}.txt").write_text("\n".join(labels) + "\n")
                n_imgs += 1
                n_objs += len(labels)
            if (i+1) % 200 == 0:
                print(f"  {i+1}/{len(img_list)}...")

        print(f"  ✓ {n_imgs} imagens, {n_objs} objetos ({n_objs/max(n_imgs,1):.2f} obj/img)")
        stats["splits"][split] = {"images": n_imgs, "objects": n_objs}

    # ── YAML do dataset ────────────────────────────────────────
    (OUTPUT_DIR / "data.yaml").write_text(
        f"path: {OUTPUT_DIR}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names:\n"
        f"  - embarcacao\n"
    )

    # ── Metadata ───────────────────────────────────────────────
    stats["total_time_s"] = round(time.time() - t0, 1)
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Salvo em {OUTPUT_DIR} ({stats['total_time_s']}s)")


if __name__ == "__main__":
    main()
