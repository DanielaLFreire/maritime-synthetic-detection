#!/usr/bin/env python3
"""
clip_decile_profile.py — diagnóstico do R2.W4: a similaridade CLIP prediz
estrutura? Recalcula o score de similaridade de cada imagem do dataset_25k
ao CITRA-3D (protocolo idêntico ao da curadoria original: ViT-B-32/openai,
máximo sobre os embeddings de referência) e cruza com as variáveis
ESTRUTURAIS que causam o negative transfer (fração de área do navio,
densidade de objetos).

Se o rank de similaridade for ortogonal à estrutura (Spearman ~0, decis
planos), então NENHUM limiar CLIP — 0,60 ou top-k estrito — pode resolver o
gap: um corte mais estrito apenas sobe no mesmo ranking, selecionando
imagens com o mesmo perfil estrutural. Isso responde ao W4 por dominância,
sem treinar nada.

Entradas (Drive):
  citra3d_embeddings.npz  — embeddings de referência da curadoria original
  dataset_25k/            — imagens + labels YOLO (train/val/test)

Saídas (results/):
  clip_structural_profile.csv      — por imagem: id, split, max_sim,
                                     area_frac, max_area, density
  clip_decile_analysis.json        — decis, correlações, perfis top-k

Uso no Colab (copie o dataset para SSD local antes — FUSE é lento p/ 25k
arquivos):
  !mkdir -p /content/data
  !cp -r "/content/drive/MyDrive/PROJETO_MARINHA/Datasets/InaTechShips/dataset_25k" /content/data/
  !pip install open-clip-torch -q
  !python scripts/07_metrics/clip_decile_profile.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# ── protocolo da curadoria original (download_direto.py) ─────────────────────
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
ORIGINAL_THRESHOLD = 0.60

REF_NPZ = "/content/drive/MyDrive/InaTechShips_similar/citra3d_embeddings.npz"
DATASET = "/content/data/dataset_25k"
OUT_DIR = "results"
SPLITS = ("train", "val", "test")
BATCH = 256


def load_ref_embeddings(path):
    z = np.load(path)
    key = z.files[0] if len(z.files) == 1 else (
        "embeddings" if "embeddings" in z.files else z.files[0])
    ref = np.asarray(z[key], dtype=np.float32)
    if ref.ndim != 2:
        raise SystemExit(f"[erro] {path}: esperado 2D, veio {ref.shape} "
                         f"(chaves: {z.files})")
    # normaliza defensivamente: com refs e queries L2-normalizados, o produto
    # interno do protocolo original é a similaridade de cosseno
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
    print(f"referências: {ref.shape[0]} embeddings de dim {ref.shape[1]} "
          f"(chave '{key}')")
    return ref


def find_images(root):
    root = Path(root)
    out = []
    for split in SPLITS:
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                out.append((split, p))
    if not out:
        # fallback: imagens soltas na raiz
        for p in sorted(root.rglob("*.jpg")):
            out.append(("all", p))
    return out


LABEL_DIR_CANDIDATES = ("labels", "labels_single_class", "labels_cleaned")


def read_label_stats(img_path):
    """area_frac = soma de w*h normalizados; max_area; density = nº de boxes.

    A pasta de labels pode se chamar labels/, labels_single_class/ ou
    labels_cleaned/ conforme o preparo; as coordenadas são idênticas entre
    as variantes (só a coluna de classe difere, e ela não é usada aqui).
    """
    lbl = None
    for cand in LABEL_DIR_CANDIDATES:
        p = img_path.parent.parent / cand / (img_path.stem + ".txt")
        if p.exists():
            lbl = p
            break
    if lbl is None:
        return dict(area_frac=np.nan, max_area=np.nan, density=0)
    areas = []
    for line in open(lbl):
        p = line.split()
        if len(p) >= 5:
            areas.append(float(p[3]) * float(p[4]))
    if not areas:
        return dict(area_frac=np.nan, max_area=np.nan, density=0)
    return dict(area_frac=float(np.sum(areas)),
                max_area=float(np.max(areas)),
                density=len(areas))


def compute_similarities(items, ref, batch=BATCH):
    import torch
    import open_clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model = model.to(device).eval()
    ref_t = torch.from_numpy(ref).to(device)

    sims = np.zeros(len(items), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, len(items), batch):
            chunk = items[i0:i0 + batch]
            imgs = []
            for _, p in chunk:
                try:
                    imgs.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            t = torch.stack(imgs).to(device)
            e = model.encode_image(t).float()
            e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
            s = (e @ ref_t.T).max(dim=1).values          # protocolo original
            sims[i0:i0 + len(chunk)] = s.cpu().numpy()
            if (i0 // batch) % 10 == 0:
                print(f"  {i0 + len(chunk):>6}/{len(items)}", flush=True)
    return sims


def decile_analysis(df):
    from scipy import stats as st

    d = df.dropna(subset=["area_frac"]).copy()
    d["decile"] = np.clip(
        (d["max_sim"].rank(pct=True, method="first") * 10).astype(int) + 1,
        1, 10)  # 1 = menos similar, 10 = mais similar

    table = []
    for dec in range(1, 11):
        sub = d[d.decile == dec]
        table.append(dict(
            decile=dec, n=len(sub),
            sim_range=[round(float(sub.max_sim.min()), 4),
                       round(float(sub.max_sim.max()), 4)],
            median_area_frac=round(float(sub.area_frac.median()), 4),
            median_max_area=round(float(sub.max_area.median()), 4),
            median_density=float(sub.density.median()),
            mean_density=round(float(sub.density.mean()), 3)))

    corr = {}
    for var in ("area_frac", "max_area", "density"):
        rho, p = st.spearmanr(d["max_sim"], d[var])
        corr[var] = dict(spearman_rho=round(float(rho), 4),
                         p=float(p), n=int(len(d)))

    profiles = {}
    for name, frac in (("all", 1.0), ("top_10pct", 0.10), ("top_5pct", 0.05)):
        k = max(1, int(len(d) * frac))
        sub = d.nlargest(k, "max_sim")
        profiles[name] = dict(
            n=k,
            sim_cutoff=round(float(sub.max_sim.min()), 4),
            median_area_frac=round(float(sub.area_frac.median()), 4),
            median_density=float(sub.density.median()))

    score_dist = dict(
        p5=round(float(d.max_sim.quantile(0.05)), 4),
        median=round(float(d.max_sim.median()), 4),
        p95=round(float(d.max_sim.quantile(0.95)), 4),
        frac_above_original_threshold=round(
            float((d.max_sim >= ORIGINAL_THRESHOLD).mean()), 4))

    return dict(deciles=table, spearman=corr, topk_profiles=profiles,
                score_distribution=score_dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_NPZ)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    import pandas as pd

    ref = load_ref_embeddings(args.ref)
    items = find_images(args.dataset)
    if not items:
        raise SystemExit(f"[erro] nenhuma imagem em {args.dataset}")
    print(f"imagens: {len(items)}")

    print("labels…")
    stats = [read_label_stats(p) for _, p in items]
    n_ok = sum(1 for s_ in stats if s_["density"] > 0)
    print(f"  labels encontrados para {n_ok}/{len(items)} imagens")
    if n_ok < 0.5 * len(items):
        raise SystemExit(
            "[erro] menos da metade das imagens tem label — a pasta de labels "
            f"não foi encontrada (candidatas: {LABEL_DIR_CANDIDATES}). "
            "Confira a árvore do dataset com: ls <dataset>/train/")

    print("similaridades CLIP…")
    sims = compute_similarities(items, ref, args.batch)

    df = pd.DataFrame([
        dict(id=p.stem, split=split, max_sim=float(s), **st_)
        for (split, p), s, st_ in zip(items, sims, stats)])

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "clip_structural_profile.csv")
    df.to_csv(csv_path, index=False)

    analysis = decile_analysis(df)
    json_path = os.path.join(args.out_dir, "clip_decile_analysis.json")
    json.dump(analysis, open(json_path, "w"), indent=2)

    # ── relatório ────────────────────────────────────────────────────────────
    print("\n=== DISTRIBUIÇÃO DOS SCORES (protocolo original, cos-sim máx.) ===")
    sd = analysis["score_distribution"]
    print(f"  p5={sd['p5']}  mediana={sd['median']}  p95={sd['p95']}")
    print(f"  fração ≥ {ORIGINAL_THRESHOLD} (limiar original): "
          f"{sd['frac_above_original_threshold']*100:.1f}%")

    print("\n=== SIMILARIDADE vs ESTRUTURA (Spearman) ===")
    for var, c in analysis["spearman"].items():
        print(f"  max_sim × {var:10}  ρ={c['spearman_rho']:+.3f}  "
              f"p={c['p']:.2e}")

    print("\n=== DECIS DE SIMILARIDADE (1=menos, 10=mais similar) ===")
    print(f"  {'dec':>3} {'n':>6} {'sim range':>18} {'med area%':>10} "
          f"{'med dens':>9}")
    for t in analysis["deciles"]:
        print(f"  {t['decile']:>3} {t['n']:>6} "
              f"{t['sim_range'][0]:.3f}–{t['sim_range'][1]:.3f}    "
              f"{t['median_area_frac']*100:>9.1f} {t['median_density']:>9.1f}")

    print("\n=== PERFIL ESTRUTURAL: conjunto todo vs top-k estrito ===")
    for name, pr in analysis["topk_profiles"].items():
        print(f"  {name:10} n={pr['n']:>6}  corte sim ≥ {pr['sim_cutoff']}  "
              f"área mediana={pr['median_area_frac']*100:.1f}%  "
              f"densidade={pr['median_density']}")

    print(f"\ngravado: {csv_path}\n         {json_path}")
    print("\nLeitura: se ρ≈0 e os decis/top-k tiverem o MESMO perfil "
          "estrutural do conjunto todo, nenhum limiar CLIP resolveria o gap.")


if __name__ == "__main__":
    main()
