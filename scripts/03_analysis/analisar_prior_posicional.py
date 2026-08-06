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
    "B2 (COCO)": "b2_coco",
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

# Escala do proxy de estratificação (ver nota metodológica no cabeçalho)
PROXY_SCALE = 1e6


def resolve_weights(arm_subdir: str, seed: int) -> Path:
    """Caminho do best.pt de um braço/seed. Edite aqui se o layout diferir."""
    return RUNS_ROOT / arm_subdir / f"seed_{seed:04d}" / "weights" / "best.pt"


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


def make_figure(df_agg, ref_arm: str, out_dir: Path, metric: str = "AP50"):
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
    ax1.set_title(f"(a) {metric} por faixa vertical")
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
        plt.savefig(out_dir / f"fig_prior_posicional.{ext}")
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
    gt = load_ground_truth(args.citra_root, SPLIT, LABEL_SUBDIR)
    y = gt["y_centers"]
    print(f"  {len(gt['images']):,} imagens · {len(gt['annotations']):,} objetos")
    print(f"  y_center: média {y.mean():.3f} · dp {y.std():.3f} · "
          f"p10 {np.quantile(y, .10):.3f} · p50 {np.quantile(y, .50):.3f} · "
          f"p90 {np.quantile(y, .90):.3f}")

    edges = compute_edges(y, args.bins, args.edges)
    print(f"\n>> Bordas dos bins ({len(edges) - 1} faixas)")
    print("  " + "  ".join(f"{e:.4f}" for e in edges))

    per_seed: dict[str, dict[str, list]] = defaultdict(dict)

    for arm, subdir in ARMS.items():
        print(f"\n>> {arm}")
        for seed in args.seeds:
            w = resolve_weights(subdir, seed)
            if not w.exists():
                print(f"  seed {seed}: ✗ peso não encontrado ({w})")
                continue
            print(f"  seed {seed}:")
            cache_file = cache_dir / f"{subdir}_seed{seed}.json"
            dets = predict_arm(w, gt, cache_file, use_cache=not args.no_cache)
            rows = evaluate_bins(gt, dets, edges)
            per_seed[arm][str(seed)] = rows
            resumo = "  ".join(f"[{r['y_lo']:.2f}–{r['y_hi']:.2f}] {r['AP50']:.3f}"
                               for r in rows)
            print(f"    AP50 por bin: {resumo}")

    if not per_seed:
        sys.exit("\n✗ Nenhum braço avaliado. Verifique RUNS_ROOT / ARMS.")

    print("\n>> Agregação")
    df_flat, df_agg = aggregate(per_seed, edges, args.ref_arm)

    df_flat.to_csv(out_dir / "prior_posicional_flat.csv", index=False)
    df_agg.to_csv(out_dir / "prior_posicional_agg.csv", index=False)
    (out_dir / "prior_posicional_per_seed.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "citra_root": str(args.citra_root),
                "runs_root": str(args.runs_root),
                "split": SPLIT,
                "eval": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU_NMS,
                         "max_det": MAX_DET},
                "edges": [float(e) for e in edges],
                "ref_arm": args.ref_arm,
                "note": ("Estratificação por y_center via proxy no campo `area` "
                         "do COCO; AP_small/medium/large NÃO são válidos aqui."),
                "per_seed": per_seed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    make_figure(df_agg, args.ref_arm, out_dir, metric="AP50")

    # ── Tabela no console ──
    print("\n" + "=" * 78)
    print(f"  Δ AP50 (pp) vs {args.ref_arm}, por faixa de y_center")
    print("=" * 78)
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
            d = r["dAP50_pp"].iloc[0]
            p = r["dAP50_p"].iloc[0]
            ptxt = "" if p is None or (isinstance(p, float) and np.isnan(p)) else f" p={p:.3f}"
            cells.append(f"{d:>+9.2f}{ptxt:>9}")
        print(line + "  ".join(cells))

    print("\n  Leitura: ganho UNIFORME entre faixas -> hipótese do prior posicional cai.")
    print("           Ganho concentrado nas faixas centrais e nulo/negativo nas")
    print("           caudas -> prior posicional confirmado.")
    print(f"\n✓ Saídas em {out_dir.resolve()}")


if __name__ == "__main__":
    main()
