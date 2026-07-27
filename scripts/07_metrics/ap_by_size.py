#!/usr/bin/env python3
"""
ap_by_size.py — AP/AR estratificados por tamanho (convenção COCO) com
pycocotools, POR SEMENTE, no test set do CITRA-3D-Real.

Reproduz a Tabela III do artigo e responde ao revisor R1.5, que perguntou em
que resolução a estratificação foi calculada. Como isso não foi registrado, o
script roda as DUAS convenções possíveis no mesmo passe:

  native : caixas em pixels da imagem original (1920x1080). Convenção COCO
           literal — small < 32^2 px, medium < 96^2 px, large >= 96^2 px.
  c640   : caixas reescaladas para uma tela 640x640, que é a convenção usada
           no resto do projeto (analisar_escala_citra3d.py usa
           small_threshold = (32/640)**2 sobre a área normalizada, e é dela
           que saem os 71,6% de objetos "small" da Tabela I).

A convenção que reproduzir os valores publicados é a resposta ao R1.5.

VERIFICAÇÃO EMBUTIDA: o mapeamento entre as duas convenções é um escalonamento
linear por eixo, e IoU é invariante a isso. Logo AP50 e AP50-95 GLOBAIS têm de
sair idênticos nas duas convenções — só o corte small/medium/large muda. Se
divergirem, há bug. O script checa isso e avisa.

Saídas (em RESULTS_DIR):
  ap_by_size_per_seed.json  — tudo, por braço/semente/convenção
  ap_by_size.json           — substitui o artefato hoje commitado ZERADO
  ap_by_size.csv            — mean ± std por braço (substitui o que só tinha médias)

Uso no Colab:
  !pip install pycocotools -q
  !python scripts/07_metrics/ap_by_size.py
  !python scripts/07_metrics/ap_by_size.py --arms "B2 (COCO)" "A' joint balanced"
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuração — mesmos caminhos de extrair_metricas_detalhadas.py
# ─────────────────────────────────────────────────────────────────────────────
CITRA_YAML = "/content/data/CITRA-3D-Real/data_single_class.yaml"
TEST_IMAGES = Path("/content/data/CITRA-3D-Real/test/images")
TEST_LABELS = Path("/content/data/CITRA-3D-Real/test/labels")
R = Path("/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/runs")
RESULTS_DIR = Path("results")

MAIN_ARMS = {
    "B2 (COCO)": {
        42:   R / "baselines/B2_coco/seed_42/train/weights/best.pt",
        123:  R / "baselines/B2_coco/seed_123/train/weights/best.pt",
        2024: R / "baselines/B2_coco/seed_2024/train/weights/best.pt",
    },
    "A' joint balanced": {
        42:   R / "braco_balanced/seed_0042/weights/best.pt",
        123:  R / "braco_balanced/seed_0123/weights/best.pt",
        2024: R / "braco_balanced/seed_2024/weights/best.pt",
    },
    "A' sequential 100ep": {
        42:   R / "braco_a_sintetico_v4/seed_0042_finetune/weights/best.pt",
        123:  R / "braco_a_sintetico_v4/seed_0123_finetune/weights/best.pt",
        2024: R / "braco_a_sintetico_v4/seed_2024_finetune/weights/best.pt",
    },
}

# Valores publicados na Tabela III (para decidir qual convenção foi usada)
PUBLISHED = {
    "B2 (COCO)": dict(AP_small=0.255, AP_medium=0.571, AP_large=0.679,
                      AR_small=0.385, AR_medium=0.647, AR_large=0.751),
    "A' joint balanced": dict(AP_small=0.262, AP_medium=0.576, AP_large=0.691,
                              AR_small=0.400, AR_medium=0.655, AR_large=0.759),
    "A' sequential 100ep": dict(AP_small=0.255, AP_medium=0.565, AP_large=0.672,
                                AR_small=0.395, AR_medium=0.645, AR_large=0.745),
}

CONVENTIONS = ("native", "c640")
METRICS = ("AP", "AP50", "AP_small", "AP_medium", "AP_large",
           "AR_small", "AR_medium", "AR_large")


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth e predições em coordenadas NORMALIZADAS (0-1).
# Materializamos em pixels só na hora de montar cada convenção.
# ─────────────────────────────────────────────────────────────────────────────
def load_gt_normalized():
    """Lê labels YOLO do test set. Retorna (imgs, anns_norm).

    imgs:      [{id, file, W, H}]
    anns_norm: [{image_id, x, y, w, h}] com x,y = canto superior esquerdo (0-1)
    """
    from PIL import Image

    files = sorted([p for p in TEST_IMAGES.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        raise SystemExit(f"[erro] nenhuma imagem em {TEST_IMAGES}")

    imgs, anns = [], []
    stems = set()
    for i, f in enumerate(files):
        stems.add(f.stem)
        with Image.open(f) as im:
            W, H = im.size
        imgs.append(dict(id=i, file=str(f), W=W, H=H))

        lbl = TEST_LABELS / (f.stem + ".txt")
        if not lbl.exists():
            continue
        for line in open(lbl):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            anns.append(dict(image_id=i, x=cx - w / 2, y=cy - h / 2, w=w, h=h))

    # sanidade: labels sem imagem correspondente indicam descompasso de nomes
    orphans = [p.name for p in TEST_LABELS.glob("*.txt") if p.stem not in stems]
    if orphans:
        print(f"[aviso] {len(orphans)} label(s) sem imagem correspondente "
              f"(ex.: {orphans[:3]}) — objetos desses arquivos NÃO contam")
    n_missing = sum(1 for f in files if not (TEST_LABELS / (f.stem + '.txt')).exists())
    if n_missing:
        print(f"[aviso] {n_missing} imagem(ns) sem arquivo de label")
    return imgs, anns


def predict_normalized(weights, imgs, imgsz, conf, iou, max_det):
    """Roda o detector e devolve predições em coordenadas normalizadas.

    Uma chamada por imagem, indexando pelo caminho QUE NÓS passamos —
    r.path da Ultralytics não é confiável (renomeia para image0.jpg etc.).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    dets = []
    for im in imgs:
        W, H = im["W"], im["H"]
        r = model.predict(im["file"], imgsz=imgsz, conf=conf, iou=iou,
                          max_det=max_det, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()      # px na imagem ORIGINAL
        scores = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), s in zip(xyxy, scores):
            dets.append(dict(image_id=im["id"], score=float(s),
                             x=x1 / W, y=y1 / H,
                             w=(x2 - x1) / W, h=(y2 - y1) / H))
    return dets


def materialize(imgs, anns, dets, convention):
    """Converte normalizado -> pixels na convenção pedida."""
    if convention == "native":
        scale = {im["id"]: (im["W"], im["H"]) for im in imgs}
        dims = {im["id"]: (im["W"], im["H"]) for im in imgs}
    elif convention == "c640":
        scale = {im["id"]: (640.0, 640.0) for im in imgs}
        dims = {im["id"]: (640, 640) for im in imgs}
    else:
        raise ValueError(convention)

    gt = {
        "images": [dict(id=im["id"], file_name=os.path.basename(im["file"]),
                        width=dims[im["id"]][0], height=dims[im["id"]][1])
                   for im in imgs],
        "categories": [dict(id=1, name="vessel")],
        "annotations": [],
    }
    for k, a in enumerate(anns):
        sx, sy = scale[a["image_id"]]
        w, h = a["w"] * sx, a["h"] * sy
        gt["annotations"].append(dict(
            id=k + 1, image_id=a["image_id"], category_id=1, iscrowd=0,
            bbox=[a["x"] * sx, a["y"] * sy, w, h], area=w * h))

    dt = []
    for d in dets:
        sx, sy = scale[d["image_id"]]
        w, h = d["w"] * sx, d["h"] * sy
        dt.append(dict(image_id=d["image_id"], category_id=1, score=d["score"],
                       bbox=[d["x"] * sx, d["y"] * sy, w, h], area=w * h))
    return gt, dt


def cocoeval(gt, dt):
    """Roda COCOeval em silêncio e devolve as métricas nomeadas."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        if not dt:
            return {m: 0.0 for m in METRICS}
        coco_dt = coco_gt.loadRes(dt)
        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    # COCOeval devolve -1 (não 0) quando um bucket de tamanho não tem GT.
    # Virar NaN evita contaminar as médias.
    def v(x):
        x = float(x)
        return float("nan") if x < 0 else x

    s = E.stats
    return dict(AP=v(s[0]), AP50=v(s[1]),
                AP_small=v(s[3]), AP_medium=v(s[4]), AP_large=v(s[5]),
                AR_small=v(s[9]), AR_medium=v(s[10]), AR_large=v(s[11]))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(MAIN_ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7, help="IoU do NMS")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando ground truth…")
    imgs, anns = load_gt_normalized()
    print(f"  {len(imgs)} imagens, {len(anns)} objetos\n")

    results = {}     # results[arm][seed][convention] = {metric: value}
    for arm in args.arms:
        if arm not in MAIN_ARMS:
            print(f"[skip] braço desconhecido: {arm}")
            continue
        results[arm] = {}
        for seed in args.seeds:
            w = MAIN_ARMS[arm].get(seed)
            if w is None or not Path(w).exists():
                print(f"[skip] {arm} seed {seed}: pesos ausentes ({w})")
                continue
            dets = predict_normalized(w, imgs, args.imgsz, args.conf,
                                      args.iou, args.max_det)
            per_conv = {}
            for conv in CONVENTIONS:
                gt, dt = materialize(imgs, anns, dets, conv)
                per_conv[conv] = cocoeval(gt, dt)

            # sanidade: AP global tem de ser igual nas duas convenções
            d = abs(per_conv["native"]["AP"] - per_conv["c640"]["AP"])
            flag = "" if d < 1e-4 else f"  <-- ATENÇÃO: AP global difere em {d:.5f}"
            results[arm][seed] = per_conv
            print(f"{arm:22} seed {seed:>4} | "
                  f"native AP_s={per_conv['native']['AP_small']:.3f} "
                  f"AR_s={per_conv['native']['AR_small']:.3f} | "
                  f"c640 AP_s={per_conv['c640']['AP_small']:.3f} "
                  f"AR_s={per_conv['c640']['AR_small']:.3f}{flag}")

    if not results:
        raise SystemExit("Nenhum braço avaliado — confira os caminhos dos pesos.")

    # ── agregação (std com ddof=0, convenção do artigo) ──────────────────────
    agg = {}
    for arm, seeds in results.items():
        agg[arm] = {}
        for conv in CONVENTIONS:
            agg[arm][conv] = {}
            for m in METRICS:
                v = np.array([seeds[s][conv][m] for s in sorted(seeds)])
                if np.isnan(v).any():
                    print(f"  [aviso] {arm}/{conv}/{m}: bucket sem GT em "
                          f"{int(np.isnan(v).sum())} semente(s) — ignoradas na média")
                agg[arm][conv][m] = dict(mean=float(np.nanmean(v)),
                                         std=float(np.nanstd(v)),
                                         per_seed={str(s): seeds[s][conv][m]
                                                   for s in sorted(seeds)})

    # ── qual convenção reproduz a Tabela III publicada? ──────────────────────
    print("\n" + "=" * 78)
    print("  QUAL CONVENÇÃO REPRODUZ A TABELA III PUBLICADA?  (R1.5)")
    print("=" * 78)
    keys = ("AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large")
    totals = {c: 0.0 for c in CONVENTIONS}
    for arm in results:
        if arm not in PUBLISHED:
            continue
        print(f"\n{arm}")
        print(f"  {'métrica':11} {'publicado':>10} " +
              " ".join(f"{c:>10}" for c in CONVENTIONS))
        for m in keys:
            pub = PUBLISHED[arm][m]
            row = f"  {m:11} {pub:>10.3f} "
            for c in CONVENTIONS:
                got = agg[arm][c][m]["mean"]
                if not np.isnan(got):
                    totals[c] += abs(got - pub)
                row += f"{got:>10.3f} "
            print(row)
    print("\n  Erro absoluto acumulado vs. publicado:")
    for c in CONVENTIONS:
        print(f"    {c:8} {totals[c]:.4f}")
    winner = min(totals, key=totals.get)
    print(f"\n  => convenção compatível: {winner}")
    if totals[winner] > 0.05:
        print("     (divergência alta: os números publicados podem ter vindo de "
              "outro protocolo — conferir maxDets, conf ou o split usado)")

    # ── teste pareado AP_small / AR_small, alimenta o item 2.4 ───────────────
    if "B2 (COCO)" in agg and "A' joint balanced" in agg:
        from scipy import stats as st
        print("\n" + "=" * 78)
        print("  TESTE PAREADO — A' joint vs B2, por tamanho  (R2.W2)")
        print("=" * 78)
        for conv in CONVENTIONS:
            print(f"\n  convenção {conv}")
            for m in keys:
                a = np.array([agg["A' joint balanced"][conv][m]["per_seed"][s]
                              for s in sorted(agg["B2 (COCO)"][conv][m]["per_seed"])])
                b = np.array([agg["B2 (COCO)"][conv][m]["per_seed"][s]
                              for s in sorted(agg["B2 (COCO)"][conv][m]["per_seed"])])
                if np.isnan(a).any() or np.isnan(b).any():
                    print(f"    {m:11} (bucket sem GT — sem teste)")
                    continue
                diff = a - b
                t, p = st.ttest_rel(a, b)
                print(f"    {m:11} Δ={diff.mean()*100:+6.2f} pp  t={t:7.2f}  "
                      f"p={p:.4f}  {(diff>0).sum()}/{len(diff)} sementes")

    # ── gravação ─────────────────────────────────────────────────────────────
    json.dump(results, open(out_dir / "ap_by_size_per_seed.json", "w"), indent=2)

    # substitui o ap_by_size.json hoje commitado zerado (usa a convenção vencedora)
    flat = {}
    for arm, seeds in results.items():
        for seed, per_conv in seeds.items():
            flat[f"{arm} seed{seed}"] = {
                "convention": winner,
                **{k: round(v, 4) for k, v in per_conv[winner].items()}}
    json.dump(flat, open(out_dir / "ap_by_size.json", "w"), indent=2)

    with open(out_dir / "ap_by_size.csv", "w") as f:
        f.write("arm,convention,seeds," +
                ",".join(f"{m},{m}_std" for m in keys) + "\n")
        for arm in agg:
            for conv in CONVENTIONS:
                n = len(results[arm])
                vals = ",".join(f"{agg[arm][conv][m]['mean']:.4f},"
                                f"{agg[arm][conv][m]['std']:.4f}" for m in keys)
                f.write(f"\"{arm}\",{conv},{n},{vals}\n")

    print(f"\nGravado em {out_dir}/: ap_by_size_per_seed.json, "
          f"ap_by_size.json, ap_by_size.csv")


if __name__ == "__main__":
    main()
