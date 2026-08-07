"""
analisar_prior_posicional.py

Diagnóstico de PRIOR POSICIONAL (viés de banda vertical) nos braços do
experimento CITRA-3D-Real.

═══════════════════════════════════════════════════════════════════════════
PERGUNTA QUE ESTE SCRIPT RESPONDE
═══════════════════════════════════════════════════════════════════════════

  A composição sintética in-place ancora os crops nas MESMAS posições das
  embarcações reais anotadas. Isso pode induzir o detector a internalizar um
  prior sobre ONDE embarcações aparecem (banda em torno do horizonte), em vez
  de aprender apenas COMO elas se parecem.

  A ablação A'joint-rand NÃO testa isso: o placement "aleatório" é restrito ao
  sea_mask (horizonte adaptativo com piso em 35% da altura), portanto amostra
  aproximadamente a MESMA banda. Além disso, o ramo real oversampleado ×13
  carrega as coordenadas verdadeiras em AMBOS os braços.

  Este script estratifica AP e AR por DECIL (ou quantil) de y_center do GT no
  test set do CITRA-3D-Real e compara os braços contra uma referência (B2).

  LEITURA DO RESULTADO:
    - Ganho do braço sintético UNIFORME entre bins  -> hipótese do prior CAI.
    - Ganho concentrado nos bins centrais, nulo ou negativo nas caudas
      (y baixo = horizonte distante, y alto = primeiro plano)
                                                   -> prior posicional CONFIRMADO.

═══════════════════════════════════════════════════════════════════════════
COMO A ESTRATIFICAÇÃO É FEITA (nota metodológica importante)
═══════════════════════════════════════════════════════════════════════════

  O pycocotools estratifica por tamanho usando `params.areaRng`, e o faz com a
  semântica correta em dois lados:
    (i)  GT fora da faixa é marcado como `_ignore` -> sai do denominador do recall;
    (ii) detecções NÃO pareadas fora da faixa também são ignoradas -> não contam
         como falso positivo daquela faixa.

  Precisamos exatamente dessa semântica, mas no eixo Y. A solução aqui é
  substituir o campo `area` por um PROXY = y_center_normalizado * 1e6, tanto no
  GT quanto nas detecções, e definir `areaRng` como a faixa do bin. Isso é
  matematicamente idêntico à estratificação por tamanho do COCO, apenas com
  outra variável de estratificação.

  Consequência: `E.summarize()` NÃO pode ser usado (assume as 4 faixas padrão).
  AP/AR são lidos diretamente de `E.eval['precision']` e `E.eval['recall']`.

  IMPORTANTE: como o proxy ocupa o campo `area`, este script NÃO produz
  AP_small/medium/large. Para isso continue usando o pipeline de AP por tamanho.

═══════════════════════════════════════════════════════════════════════════
USO
═══════════════════════════════════════════════════════════════════════════

    # padrão: 5 bins por quantil, braços B2 / A'joint / A'joint-rand
    python analisar_prior_posicional.py

    # decis, e reaproveitando predições já cacheadas
    python analisar_prior_posicional.py --bins 10

    # bordas fixas em vez de quantis (útil para a narrativa "banda")
    python analisar_prior_posicional.py --edges 0,0.35,0.45,0.55,0.65,1.0

    # forçar nova inferência (ignora cache)
    python analisar_prior_posicional.py --no-cache

SAÍDAS (em --out, default: results/prior_posicional/)
    prior_posicional_per_seed.json   detalhado, por braço × seed × bin
    prior_posicional_flat.csv        formato plano p/ o Streamlit
    prior_posicional_agg.csv         agregado mean±std + Δ vs ref + p-valor
    fig_prior_posicional.pdf/.png    Δ AP50 vs referência por bin
    preds_cache/<braço>_<seed>.json  detecções cacheadas (COCO results)

DEPENDÊNCIAS
    ultralytics, pycocotools, numpy, pandas, matplotlib, (scipy opcional)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — ajuste os caminhos ao layout da sua sessão
# ═══════════════════════════════════════════════════════════════════════════

# Raiz do CITRA-3D-Real (espera <root>/test/images e <root>/test/<LABEL_SUBDIR>)
CITRA_ROOT = Path("/content/data/CITRA-3D-Real")

LABEL_SUBDIR = "labels_single_class"
SPLIT = "test"

# Raiz dos runs. Os caminhos dos braços abaixo são relativos a ela.
RUNS_ROOT = Path(
    "/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs"
)

# Braço -> subdiretório do run. O peso esperado é:
#   RUNS_ROOT / <subdir> / seed_<seed:04d> / weights / best.pt
# Se algum braço usar outro padrão, edite RESOLVE_WEIGHTS abaixo.
ARMS = {
    "B2 (COCO)": "baselines/B2_coco",
    "A' joint": "braco_balanced",
    "A' joint-rand": "braco_random_copypaste_v4",
}

# Braço usado como referência nas comparações (Δ e teste t pareado).
REF_ARM = "B2 (COCO)"

SEEDS = [42, 123, 2024]

# Inferência (idêntica ao protocolo in-domain do artigo)
IMGSZ = 640
CONF = 0.001
IOU_NMS = 0.7
MAX_DET = 300
DEVICE = 0

OUT_DIR = Path("results/prior_posicional")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Classes de tamanho COCO, em pixels² de área da bbox.
SIZE_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}

# Faixas usadas na estatística de assimetria centro–cauda.
# Com 5 bins: centrais = {1,2,3}, caudas = {0,4}.
def central_tail_bins(n_bins: int) -> tuple[list[int], list[int]]:
    k = n_bins // 3
    tails = list(range(k)) + list(range(n_bins - k, n_bins))
    central = [b for b in range(n_bins) if b not in tails]
    return central, tails

# Escala do proxy de estratificação (ver nota metodológica no cabeçalho)
PROXY_SCALE = 1e6


def resolve_weights(arm_subdir: str, seed: int) -> Path:
    """
    Localiza o best.pt de um braço/seed tolerando as três convenções presentes
    no Drive deste projeto:

        <arm>/seed_0042/weights/best.pt            (braco_balanced, *_v4)
        <arm>/seed_42/train/weights/best.pt        (baselines/B2_coco, B1_random)
        <arm>/seed_0042_finetune/weights/best.pt   (braços sequenciais)

    Checkpoints de pré-treino são descartados; o de fine-tune tem precedência
    quando ambos existem. Também tolera o padding irregular de
    `seed_02024_finetune` em braco_a_sintetico.
    """
    base = RUNS_ROOT / arm_subdir
    fallback = base / f"seed_{seed:04d}" / "weights" / "best.pt"
    if not base.exists():
        return fallback

    tokens = {f"seed_{seed:04d}", f"seed_{seed}", f"seed_{seed:05d}"}
    cands = []
    for p in base.rglob("weights/best.pt"):
        if "pretrain" in str(p):
            continue
        if any(part in tokens or any(part.startswith(t + "_") for t in tokens)
               for part in p.parts):
            cands.append(p)

    if not cands:
        return fallback
    if len(cands) > 1:
        ft = [c for c in cands if "finetune" in str(c)]
        if ft:
            cands = ft
    if len(cands) > 1:
        print(f"    ⚠ múltiplos best.pt para seed {seed} em {arm_subdir}: "
              f"usando {cands[0].parent.parent.name}")
    return sorted(cands)[0]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Ground truth
# ═══════════════════════════════════════════════════════════════════════════

def list_test_images(root: Path, split: str) -> list[Path]:
    img_dir = root / split / "images"
    if not img_dir.exists():
        img_dir = root / split  # layout alternativo: imagens soltas no split
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {img_dir}")
    return imgs


def image_size(path: Path) -> tuple[int, int]:
    """(W, H) sem carregar a imagem inteira quando possível."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2

        arr = cv2.imread(str(path))
        if arr is None:
            raise FileNotFoundError(f"Não consegui ler {path}")
        return arr.shape[1], arr.shape[0]


def load_ground_truth(root: Path, split: str, label_subdir: str) -> dict:
    """
    Lê labels YOLO e devolve estrutura COCO-like, já com o proxy de y_center
    no campo `area`.

    Retorna dict com: images, annotations, categories, e y_centers (np.array).
    """
    imgs = list_test_images(root, split)
    label_dir = root / split / label_subdir
    if not label_dir.exists():
        raise FileNotFoundError(f"Diretório de labels não encontrado: {label_dir}")

    images, annotations, y_centers = [], [], []
    ann_id = 1
    sem_label = 0

    for img_id, img_path in enumerate(imgs, start=1):
        W, H = image_size(img_path)
        images.append(
            {"id": img_id, "file_name": img_path.name, "width": W, "height": H}
        )

        lbl = label_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            sem_label += 1
            continue

        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = (float(v) for v in parts[:5])
            bw, bh = w * W, h * H
            x0, y0 = (xc * W) - bw / 2.0, (yc * H) - bh / 2.0
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x0, y0, bw, bh],
                    # PROXY: y_center normalizado -> campo `area`
                    "area": float(yc) * PROXY_SCALE,
                    "bbox_area": bw * bh,
                    "y_center": float(yc),
                    "iscrowd": 0,
                    "ignore": 0,
                }
            )
            y_centers.append(float(yc))
            ann_id += 1

    if sem_label:
        print(f"  ⚠ {sem_label} imagens sem arquivo de label (tratadas como vazias)")

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "vessel"}],
        "y_centers": np.asarray(y_centers, dtype=float),
        "paths": imgs,
        "name_to_id": {im["file_name"]: im["id"] for im in images},
    }


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float | None]:
    """Spearman ρ (e p-valor se scipy disponível). Fallback: Pearson dos postos."""
    try:
        from scipy import stats as sps

        r = sps.spearmanr(a, b)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1]), None


def size_class_of(area: float) -> str:
    for name in ("small", "medium", "large"):
        lo, hi = SIZE_RANGES[name]
        if lo <= area < hi:
            return name
    return "large"


def filter_by_size(gt: dict, size_class: str) -> dict:
    """
    Restringe o GT a uma classe de tamanho COCO.

    O campo `ignore` do COCO NÃO é confiável: `COCOeval._prepare` o sobrescreve
    com `iscrowd`. Por isso a filtragem REMOVE as anotações fora da classe em
    vez de marcá-las. As detecções fora da classe são descartadas em
    `filter_dets_by_size` — sem isso, um FP de qualquer tamanho contaria contra
    todas as classes.

    Desvio conhecido em relação ao COCO: uma detecção fora da classe que
    pareasse um GT dentro da classe seria contada como TP pelo COCO e aqui vira
    FN. Com IoU ≥ 0,5 nas fronteiras 32²/96², o efeito é de segunda ordem.
    """
    if size_class == "all":
        return gt

    lo, hi = SIZE_RANGES[size_class]
    anns = [a for a in gt["annotations"] if lo <= a["bbox_area"] < hi]
    out = dict(gt)
    out["annotations"] = anns
    out["y_centers"] = np.asarray([a["y_center"] for a in anns], dtype=float)
    return out


def filter_dets_by_size(dets: list[dict], size_class: str) -> list[dict]:
    if size_class == "all":
        return dets
    lo, hi = SIZE_RANGES[size_class]
    return [d for d in dets if lo <= (d["bbox"][2] * d["bbox"][3]) < hi]


def report_size_confound(gt_full: dict) -> dict:
    """
    Quantifica o confundidor y_center × tamanho aparente.

    Em câmera marítima fixa, embarcação distante fica alta na imagem (y baixo) e
    pequena; embarcação próxima fica embaixo (y alto) e grande. Sem correlação,
    os eixos são independentes e a estratificação vertical é limpa.
    """
    y = np.asarray([a["y_center"] for a in gt_full["annotations"]], dtype=float)
    area = np.asarray([a["bbox_area"] for a in gt_full["annotations"]], dtype=float)
    rho, p = spearman(y, area)

    counts: dict[str, int] = {"small": 0, "medium": 0, "large": 0}
    for a in gt_full["annotations"]:
        counts[size_class_of(a["bbox_area"])] += 1

    ptxt = "" if p is None else f" (p={p:.2e})"
    print(f"  Spearman(y_center, área da bbox): ρ = {rho:+.3f}{ptxt}")
    if abs(rho) < 0.2:
        print("    → correlação fraca: eixo vertical e tamanho são "
              "aproximadamente independentes.")
    elif abs(rho) < 0.5:
        print("    → correlação moderada: o controle por classe de tamanho "
              "(--size-class) é necessário.")
    else:
        print("    → correlação forte: AP por faixa vertical é, em boa medida, "
              "AP por tamanho. Interprete apenas os resultados controlados.")

    n = sum(counts.values())
    print("  Classes de tamanho COCO: " + " · ".join(
        f"{k} {v} ({100 * v / n:.1f}%)" for k, v in counts.items()))

    return {"spearman_rho": rho, "spearman_p": p, "size_counts": counts}


def report_area_by_bin(gt: dict, edges: np.ndarray) -> None:
    """Área mediana da bbox por faixa vertical — magnitude do confundidor."""
    y = np.asarray([a["y_center"] for a in gt["annotations"]], dtype=float)
    area = np.asarray([a["bbox_area"] for a in gt["annotations"]], dtype=float)
    print("  Área mediana da bbox por faixa (px²):")
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (y >= lo) & (y < hi) if b < len(edges) - 2 else (y >= lo) & (y <= hi)
        if not m.any():
            continue
        med = float(np.median(area[m]))
        print(f"    [{lo:.2f}–{hi:.2f}]  {med:>9.0f}  "
              f"({size_class_of(med)}, n={int(m.sum())})")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Predições
# ═══════════════════════════════════════════════════════════════════════════

def predict_arm(weights: Path, gt: dict, cache_file: Path, use_cache: bool) -> list[dict]:
    """Roda inferência (ou carrega cache) e devolve detecções COCO-results."""
    if use_cache and cache_file.exists():
        dets = json.loads(cache_file.read_text())
        print(f"    cache: {len(dets):,} detecções ({cache_file.name})")
        return dets

    from ultralytics import YOLO

    model = YOLO(str(weights))
    dets: list[dict] = []
    paths = gt["paths"]
    name_to_id = gt["name_to_id"]
    hw_by_id = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}

    CHUNK = 64
    for i in range(0, len(paths), CHUNK):
        batch = [str(p) for p in paths[i : i + CHUNK]]
        results = model.predict(
            source=batch,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU_NMS,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False,
            stream=False,
        )
        for path_str, r in zip(batch, results):
            img_id = name_to_id[Path(path_str).name]
            _, H = hw_by_id[img_id]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, scores):
                bw, bh = float(x2 - x1), float(y2 - y1)
                yc_norm = ((float(y1) + bh / 2.0) / H) if H else 0.0
                dets.append(
                    {
                        "image_id": int(img_id),
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), bw, bh],
                        "score": float(s),
                        # PROXY do lado das detecções (mesma semântica do GT)
                        "area": yc_norm * PROXY_SCALE,
                    }
                )
        print(f"    inferência {min(i + CHUNK, len(paths))}/{len(paths)}", end="\r")

    print(f"    inferência concluída: {len(dets):,} detecções            ")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(dets))
    return dets


# ═══════════════════════════════════════════════════════════════════════════
# 3. Avaliação estratificada por bin de y_center
# ═══════════════════════════════════════════════════════════════════════════

def build_coco(dataset: dict):
    from pycocotools.coco import COCO

    c = COCO()
    c.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def evaluate_bins(gt: dict, dets: list[dict], edges: np.ndarray) -> list[dict]:
    """
    Avalia AP/AR restritos a cada bin de y_center, com a semântica de `ignore`
    do COCO aplicada tanto ao GT quanto às detecções não pareadas.
    """
    from pycocotools.cocoeval import COCOeval

    gt_ds = {
        "images": gt["images"],
        "annotations": gt["annotations"],
        "categories": gt["categories"],
    }
    coco_gt = build_coco(gt_ds)

    dt_anns = []
    for i, d in enumerate(dets, start=1):
        dt_anns.append({**d, "id": i, "iscrowd": 0})
    coco_dt = build_coco(
        {
            "images": gt["images"],
            "annotations": dt_anns,
            "categories": gt["categories"],
        }
    )

    img_ids = [im["id"] for im in gt["images"]]
    y_all = gt["y_centers"]
    rows = []

    for b in range(len(edges) - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        # bin fechado à direita no último para não perder y_center == hi
        n_gt = int(
            np.sum((y_all >= lo) & (y_all < hi))
            if b < len(edges) - 2
            else np.sum((y_all >= lo) & (y_all <= hi))
        )

        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.params.imgIds = img_ids
        E.params.catIds = [1]
        E.params.maxDets = [MAX_DET]
        E.params.areaRng = [[lo * PROXY_SCALE, hi * PROXY_SCALE]]
        E.params.areaRngLbl = ["ybin"]
        with contextlib.redirect_stdout(io.StringIO()):  # silencia o pycocotools
            E.evaluate()
            E.accumulate()

        prec = E.eval["precision"]  # [T, R, K, A, M]
        rec = E.eval["recall"]  # [T, K, A, M]

        def _mean(x):
            x = x[x > -1]
            return float(np.mean(x)) if x.size else float("nan")

        rows.append(
            {
                "bin": b,
                "y_lo": round(lo, 4),
                "y_hi": round(hi, 4),
                "n_gt": n_gt,
                "AP": _mean(prec[:, :, 0, 0, 0]),        # AP@[.50:.95]
                "AP50": _mean(prec[0, :, 0, 0, 0]),      # AP@.50
                "AP75": _mean(prec[5, :, 0, 0, 0]),      # AP@.75
                "AR": _mean(rec[:, 0, 0, 0]),            # AR@[.50:.95]
                "AR50": _mean(rec[0:1, 0, 0, 0]),        # AR@.50
            }
        )

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 4. Bordas dos bins
# ═══════════════════════════════════════════════════════════════════════════

def compute_edges(y: np.ndarray, n_bins: int, custom: str | None) -> np.ndarray:
    if custom:
        e = np.array([float(v) for v in custom.split(",")], dtype=float)
        if e.size < 2 or not np.all(np.diff(e) > 0):
            sys.exit("--edges precisa de valores crescentes, ex: 0,0.35,0.55,1.0")
        return e
    qs = np.linspace(0, 1, n_bins + 1)
    e = np.quantile(y, qs)
    e[0], e[-1] = 0.0, 1.0
    # colapsa bordas duplicadas (caso patológico de distribuição concentrada)
    e = np.unique(np.round(e, 6))
    return e


# ═══════════════════════════════════════════════════════════════════════════
# 5. Agregação, teste pareado e saídas
# ═══════════════════════════════════════════════════════════════════════════

def paired_t(a: list[float], b: list[float]) -> float | None:
    """Teste t pareado por seed (a vs b). None se scipy ausente ou n<2."""
    if len(a) != len(b) or len(a) < 2:
        return None
    try:
        from scipy import stats as sps

        return float(sps.ttest_rel(a, b).pvalue)
    except Exception:
        return None


def aggregate(per_seed: dict, edges: np.ndarray, ref_arm: str):
    import pandas as pd

    flat = []
    for arm, seeds in per_seed.items():
        for seed, rows in seeds.items():
            for r in rows:
                flat.append({"arm": arm, "seed": int(seed), **r})
    df_flat = pd.DataFrame(flat)

    metrics = ["AP", "AP50", "AP75", "AR", "AR50"]
    agg_rows = []
    for (arm, b), g in df_flat.groupby(["arm", "bin"]):
        row = {
            "arm": arm,
            "bin": int(b),
            "y_lo": float(g["y_lo"].iloc[0]),
            "y_hi": float(g["y_hi"].iloc[0]),
            "n_gt": int(g["n_gt"].iloc[0]),
            "n_seeds": len(g),
        }
        for m in metrics:
            vals = g.sort_values("seed")[m].tolist()
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))  # ddof=0, convenção do projeto
        agg_rows.append(row)
    df_agg = pd.DataFrame(agg_rows)

    # Δ vs referência + teste t pareado por seed
    if ref_arm in df_flat["arm"].unique():
        for m in metrics:
            deltas, pvals = [], []
            for _, row in df_agg.iterrows():
                arm, b = row["arm"], row["bin"]
                ref = df_flat[(df_flat.arm == ref_arm) & (df_flat.bin == b)].sort_values("seed")
                cur = df_flat[(df_flat.arm == arm) & (df_flat.bin == b)].sort_values("seed")
                common = sorted(set(ref.seed) & set(cur.seed))
                rv = ref[ref.seed.isin(common)][m].tolist()
                cv = cur[cur.seed.isin(common)][m].tolist()
                deltas.append(100.0 * (np.mean(cv) - np.mean(rv)) if rv else float("nan"))
                pvals.append(paired_t(cv, rv) if arm != ref_arm else None)
            df_agg[f"d{m}_pp"] = deltas
            df_agg[f"d{m}_p"] = pvals

    return df_flat, df_agg.sort_values(["arm", "bin"]).reset_index(drop=True)


def asymmetry_stats(per_seed: dict, n_bins: int, ref_arm: str, metric: str = "AP50"):
    """
    Estatística centro–cauda: UM escalar por braço/seed, em vez de um teste por
    bin.

    Com n=3 seeds e 5 bins, testar bin a bin gera 10 comparações e p-valores não
    interpretáveis sem correção. A assimetria (AP médio nas faixas centrais menos
    AP médio nas caudas) é o contraste pré-especificado que a hipótese do prior
    posicional prevê: positiva e maior nos braços sintéticos.

    Atenção: com n=3, t crítico bilateral = 4,30. Só um efeito muito grande
    atinge p < 0,05 — a informação está na CONSISTÊNCIA DE SINAL entre seeds.
    """
    import pandas as pd

    central, tails = central_tail_bins(n_bins)

    per_arm: dict[str, dict[str, float]] = {}
    for arm, seeds in per_seed.items():
        per_arm[arm] = {}
        for seed, rows in seeds.items():
            v = {r["bin"]: r[metric] for r in rows}
            if not all(b in v for b in central + tails):
                continue
            per_arm[arm][seed] = (
                float(np.mean([v[b] for b in central]))
                - float(np.mean([v[b] for b in tails]))
            )

    ref = per_arm.get(ref_arm, {})
    rows = []
    for arm, vals in per_arm.items():
        seeds = sorted(vals, key=int)
        a = [vals[s] for s in seeds]
        row = {
            "arm": arm,
            "n_seeds": len(a),
            "central_bins": str(central),
            "tail_bins": str(tails),
            "asymmetry_mean_pp": 100.0 * float(np.mean(a)),
            "asymmetry_std_pp": 100.0 * float(np.std(a)),
            "per_seed_pp": {s: 100.0 * vals[s] for s in seeds},
        }
        common = [s for s in seeds if s in ref]
        if arm != ref_arm and len(common) >= 2:
            cur = [vals[s] for s in common]
            base = [ref[s] for s in common]
            d = 100.0 * (np.array(cur) - np.array(base))
            row["delta_vs_ref_pp"] = float(d.mean())
            row["delta_std_pp"] = float(d.std(ddof=1))
            row["all_seeds_same_sign"] = bool(np.all(d > 0) or np.all(d < 0))
            row["p_paired"] = paired_t(cur, base)
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"  ASSIMETRIA CENTRO–CAUDA ({metric}) — faixas centrais {central} "
          f"vs caudas {tails}")
    print("=" * 78)
    for r in rows:
        ps = "  ".join(f"{v:+7.2f}" for v in r["per_seed_pp"].values())
        print(f"  {r['arm']:16} {ps}   média {r['asymmetry_mean_pp']:+6.2f} pp")
    print()
    for r in rows:
        if "delta_vs_ref_pp" not in r:
            continue
        p = r.get("p_paired")
        ptxt = "n/d" if p is None else f"{p:.4f}"
        sign = "todas iguais" if r["all_seeds_same_sign"] else "MISTOS"
        print(f"  {r['arm']:16} Δ vs {ref_arm}: {r['delta_vs_ref_pp']:+6.2f} pp  "
              f"dp {r['delta_std_pp']:5.2f}  p={ptxt}  sinais: {sign}")

    return df


def make_figure(df_agg, ref_arm: str, out_dir: Path, metric: str = "AP50",
                suffix: str = ""):
    if ref_arm not in set(df_agg["arm"]) or f"d{metric}_pp" not in df_agg.columns:
        print(f"  ⚠ figura pulada: braço de referência '{ref_arm}' sem resultados.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.6,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    arms = [a for a in df_agg["arm"].unique() if a != ref_arm]
    if not arms:
        return
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#D4537E", "#378ADD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ref = df_agg[df_agg.arm == ref_arm].sort_values("bin")
    labels = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(ref["y_lo"], ref["y_hi"])]
    x = np.arange(len(ref))

    # (a) valor absoluto por bin
    ax1.errorbar(
        x, ref[f"{metric}_mean"], yerr=ref[f"{metric}_std"],
        marker="o", ms=3.5, lw=1.0, capsize=2, color="#888780", label=ref_arm,
    )
    for i, arm in enumerate(arms):
        d = df_agg[df_agg.arm == arm].sort_values("bin")
        ax1.errorbar(
            x, d[f"{metric}_mean"], yerr=d[f"{metric}_std"],
            marker="s", ms=3.5, lw=1.0, capsize=2,
            color=colors[i % len(colors)], label=arm,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.set_xlabel("Faixa de $y_{center}$ do GT (normalizado)")
    ax1.set_ylabel(metric)
    cls = suffix.lstrip("_") or "todos os tamanhos"
    ax1.set_title(f"(a) {metric} por faixa vertical ({cls})")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(axis="y", lw=0.3, alpha=0.4)

    # (b) delta vs referência
    width = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        d = df_agg[df_agg.arm == arm].sort_values("bin")
        ax2.bar(
            x + (i - (len(arms) - 1) / 2) * width,
            d[f"d{metric}_pp"], width=width,
            color=colors[i % len(colors)], alpha=0.85, label=arm,
        )
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.set_xlabel("Faixa de $y_{center}$ do GT (normalizado)")
    ax2.set_ylabel(f"Δ {metric} vs {ref_arm} (pp)")
    ax2.set_title("(b) Ganho estratificado")
    ax2.legend(fontsize=7, frameon=False)
    ax2.grid(axis="y", lw=0.3, alpha=0.4)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig_prior_posicional{suffix}.{ext}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    global RUNS_ROOT

    ap = argparse.ArgumentParser(
        description="AP/AR estratificados por faixa vertical (prior posicional)."
    )
    ap.add_argument("--citra-root", type=Path, default=CITRA_ROOT)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--bins", type=int, default=5, help="nº de bins por quantil")
    ap.add_argument("--edges", type=str, default=None,
                    help="bordas fixas, ex: 0,0.35,0.45,0.55,0.65,1.0")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--ref-arm", type=str, default=REF_ARM)
    ap.add_argument("--size-class", type=str, default="all",
                    choices=list(SIZE_RANGES),
                    help="restringe a análise a uma classe de tamanho COCO, "
                         "controlando o confundidor y_center × tamanho")
    ap.add_argument("--metric", type=str, default="AP50",
                    choices=["AP", "AP50", "AP75", "AR", "AR50"],
                    help="métrica da figura e da estatística de assimetria")
    ap.add_argument("--no-cache", action="store_true",
                    help="força nova inferência mesmo com cache presente")
    args = ap.parse_args()

    RUNS_ROOT = args.runs_root

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "preds_cache"

    print("=" * 78)
    print("  DIAGNÓSTICO DE PRIOR POSICIONAL — AP/AR por faixa de y_center")
    print("=" * 78)

    print("\n>> Ground truth")
    gt_full = load_ground_truth(args.citra_root, SPLIT, LABEL_SUBDIR)
    y_full = gt_full["y_centers"]
    print(f"  {len(gt_full['images']):,} imagens · "
          f"{len(gt_full['annotations']):,} objetos")
    print(f"  y_center: média {y_full.mean():.3f} · dp {y_full.std():.3f} · "
          f"p10 {np.quantile(y_full, .10):.3f} · "
          f"p50 {np.quantile(y_full, .50):.3f} · "
          f"p90 {np.quantile(y_full, .90):.3f}")

    print("\n>> Confundidor y_center × tamanho")
    confound = report_size_confound(gt_full)

    gt = filter_by_size(gt_full, args.size_class)
    y = gt["y_centers"]
    if args.size_class != "all":
        print(f"\n>> Restrição a objetos '{args.size_class}': "
              f"{len(gt['annotations']):,} de {len(gt_full['annotations']):,} "
              f"objetos ({100 * len(gt['annotations']) / len(gt_full['annotations']):.1f}%)")
        if len(gt["annotations"]) < 200:
            print("  ⚠ amostra pequena — considere --bins 3.")

    edges = compute_edges(y, args.bins, args.edges)
    n_bins = len(edges) - 1
    print(f"\n>> Bordas dos bins ({n_bins} faixas)")
    print("  " + "  ".join(f"{e:.4f}" for e in edges))
    report_area_by_bin(gt, edges)

    per_bin = len(gt["annotations"]) / max(n_bins, 1)
    if per_bin < 100:
        print(f"  ⚠ ~{per_bin:.0f} objetos por faixa: AP fica ruidoso. "
              f"Reduza --bins.")

    per_seed: dict[str, dict[str, list]] = defaultdict(dict)

    for arm, subdir in ARMS.items():
        print(f"\n>> {arm}")
        for seed in args.seeds:
            w = resolve_weights(subdir, seed)
            if not w.exists():
                print(f"  seed {seed}: ✗ peso não encontrado ({w})")
                continue
            print(f"  seed {seed}:")
            # O cache é independente da classe de tamanho: as predições são
            # sempre completas e o filtro é aplicado depois. Trocar
            # --size-class não custa nova inferência.
            cache_file = cache_dir / f"{subdir.replace('/', '__')}_seed{seed}.json"
            dets = predict_arm(w, gt_full, cache_file, use_cache=not args.no_cache)
            dets = filter_dets_by_size(dets, args.size_class)
            rows = evaluate_bins(gt, dets, edges)
            per_seed[arm][str(seed)] = rows
            resumo = "  ".join(f"[{r['y_lo']:.2f}–{r['y_hi']:.2f}] {r['AP50']:.3f}"
                               for r in rows)
            print(f"    AP50 por bin: {resumo}")

    if not per_seed:
        sys.exit("\n✗ Nenhum braço avaliado. Verifique RUNS_ROOT / ARMS.")

    print("\n>> Agregação")
    df_flat, df_agg = aggregate(per_seed, edges, args.ref_arm)
    df_asym = asymmetry_stats(per_seed, n_bins, args.ref_arm, metric=args.metric)

    # Sufixo evita que rodadas por classe de tamanho se sobrescrevam.
    sfx = "" if args.size_class == "all" else f"_{args.size_class}"

    df_flat.to_csv(out_dir / f"prior_posicional_flat{sfx}.csv", index=False)
    df_agg.to_csv(out_dir / f"prior_posicional_agg{sfx}.csv", index=False)
    df_asym.to_csv(out_dir / f"prior_posicional_asymmetry{sfx}.csv", index=False)

    central, tails = central_tail_bins(n_bins)
    (out_dir / f"prior_posicional_per_seed{sfx}.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "citra_root": str(args.citra_root),
                "runs_root": str(args.runs_root),
                "split": SPLIT,
                "eval": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU_NMS,
                         "max_det": MAX_DET},
                "size_class": args.size_class,
                "size_range_px2": [SIZE_RANGES[args.size_class][0],
                                   None if np.isinf(SIZE_RANGES[args.size_class][1])
                                   else SIZE_RANGES[args.size_class][1]],
                "n_objects_evaluated": len(gt["annotations"]),
                "n_objects_total": len(gt_full["annotations"]),
                "confound": confound,
                "edges": [float(e) for e in edges],
                "central_bins": central,
                "tail_bins": tails,
                "metric": args.metric,
                "ref_arm": args.ref_arm,
                "note": ("Estratificação por y_center via proxy no campo `area` "
                         "do COCO; AP_small/medium/large NÃO são válidos aqui. "
                         "O filtro por classe de tamanho REMOVE anotações e "
                         "detecções fora da faixa, porque COCOeval._prepare "
                         "sobrescreve o campo `ignore` com `iscrowd`."),
                "per_seed": per_seed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    make_figure(df_agg, args.ref_arm, out_dir, metric=args.metric, suffix=sfx)

    # ── Tabela no console ──
    print("\n" + "=" * 78)
    print(f"  Δ {args.metric} (pp) vs {args.ref_arm}, por faixa de y_center "
          f"[{args.size_class}]")
    print("=" * 78)
    if args.ref_arm not in set(df_agg["arm"]):
        print(f"\n  ⚠ Braço de referência '{args.ref_arm}' sem resultados — "
              f"Δ e teste t não calculados.")
        print(f"    Braços com dados: {sorted(set(df_agg['arm']))}")
    else:
        ref_bins = df_agg[df_agg.arm == args.ref_arm].sort_values("bin")
        header = "  faixa            n_GT  " + "  ".join(
            f"{a:>18}" for a in df_agg.arm.unique() if a != args.ref_arm
        )
        print(header)
        for _, rb in ref_bins.iterrows():
            line = f"  {rb['y_lo']:.2f}–{rb['y_hi']:.2f}  {int(rb['n_gt']):>8}  "
            cells = []
            for arm in df_agg.arm.unique():
                if arm == args.ref_arm:
                    continue
                r = df_agg[(df_agg.arm == arm) & (df_agg.bin == rb["bin"])]
                if r.empty:
                    cells.append(f"{'—':>18}")
                    continue
                d = r[f"d{args.metric}_pp"].iloc[0]
                p = r[f"d{args.metric}_p"].iloc[0]
                ptxt = "" if p is None or (isinstance(p, float) and np.isnan(p)) else f" p={p:.3f}"
                cells.append(f"{d:>+9.2f}{ptxt:>9}")
            print(line + "  ".join(cells))

    print("\n  Leitura: ganho UNIFORME entre faixas -> hipótese do prior posicional cai.")
    print("           Ganho concentrado nas faixas centrais e nulo/negativo nas")
    print("           caudas -> prior posicional confirmado.")
    print(f"\n✓ Saídas em {out_dir.resolve()}")


if __name__ == "__main__":
    main()
