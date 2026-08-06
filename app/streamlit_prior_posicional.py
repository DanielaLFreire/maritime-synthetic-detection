"""
streamlit_prior_posicional.py

Painel para explorar o diagnóstico de prior posicional produzido por
`analisar_prior_posicional.py`.

Uso:
    streamlit run app/streamlit_prior_posicional.py
    streamlit run app/streamlit_prior_posicional.py -- --dir results/prior_posicional

Espera encontrar no diretório de resultados:
    prior_posicional_flat.csv   (arm, seed, bin, y_lo, y_hi, n_gt, AP, AP50, ...)
    prior_posicional_agg.csv    (agregado com *_mean, *_std, d<MÉTRICA>_pp, _p)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_DIR = Path("results/prior_posicional")
METRICS = ["AP50", "AP", "AP75", "AR", "AR50"]

st.set_page_config(page_title="Prior posicional — CITRA-3D", layout="wide")


def default_dir() -> Path:
    if "--dir" in sys.argv:
        return Path(sys.argv[sys.argv.index("--dir") + 1])
    return DEFAULT_DIR


@st.cache_data(show_spinner=False)
def load(dir_path: str):
    d = Path(dir_path)
    flat_p, agg_p = d / "prior_posicional_flat.csv", d / "prior_posicional_agg.csv"
    if not flat_p.exists() or not agg_p.exists():
        return None, None
    return pd.read_csv(flat_p), pd.read_csv(agg_p)


st.title("Prior posicional — AP/AR por faixa vertical")
st.caption(
    "Diagnóstico da hipótese de que a composição sintética *in-place* induz o "
    "detector a internalizar um prior sobre **onde** embarcações aparecem, e não "
    "apenas sobre **como** elas se parecem."
)

with st.sidebar:
    st.header("Dados")
    dir_path = st.text_input("Diretório de resultados", value=str(default_dir()))
    flat, agg = load(dir_path)
    if flat is None:
        st.error("CSVs não encontrados neste diretório.")
        st.stop()

    arms = sorted(agg["arm"].unique().tolist())
    ref_arm = st.selectbox(
        "Braço de referência",
        arms,
        index=next((i for i, a in enumerate(arms) if a.startswith("B2")), 0),
    )
    metric = st.selectbox("Métrica", METRICS, index=0)
    show_arms = st.multiselect(
        "Braços exibidos", arms, default=arms
    )

if not show_arms:
    st.warning("Selecione ao menos um braço.")
    st.stop()

agg_v = agg[agg["arm"].isin(show_arms)].copy()
ref = agg_v[agg_v.arm == ref_arm].sort_values("bin")
if ref.empty:
    st.warning("O braço de referência não está entre os exibidos.")
    st.stop()

labels = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(ref["y_lo"], ref["y_hi"])]
n_bins = len(ref)

# Recalcula Δ na métrica escolhida (o CSV traz todas, mas assim o painel fica
# consistente caso o usuário troque a referência aqui).
ref_map = dict(zip(ref["bin"], ref[f"{metric}_mean"]))
agg_v["delta_pp"] = agg_v.apply(
    lambda r: 100.0 * (r[f"{metric}_mean"] - ref_map.get(r["bin"], np.nan)), axis=1
)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{metric} por faixa de $y_{{center}}$")
    wide = (
        agg_v.pivot(index="bin", columns="arm", values=f"{metric}_mean")
        .rename(index=dict(enumerate(labels)))
    )
    st.line_chart(wide, height=320)

with col2:
    st.subheader(f"Δ {metric} vs {ref_arm} (pp)")
    wide_d = (
        agg_v[agg_v.arm != ref_arm]
        .pivot(index="bin", columns="arm", values="delta_pp")
        .rename(index=dict(enumerate(labels)))
    )
    if wide_d.empty:
        st.info("Nenhum braço além da referência selecionado.")
    else:
        st.bar_chart(wide_d, height=320)

# ── Veredito heurístico ────────────────────────────────────────────────────
st.subheader("Leitura")

central = set(range(n_bins // 3, n_bins - n_bins // 3))
tails = set(range(n_bins)) - central

verdicts = []
for arm in show_arms:
    if arm == ref_arm:
        continue
    d = agg_v[agg_v.arm == arm].set_index("bin")["delta_pp"]
    d_c = float(d[d.index.isin(central)].mean())
    d_t = float(d[d.index.isin(tails)].mean())
    gap = d_c - d_t
    verdicts.append(
        {
            "Braço": arm,
            "Δ médio faixas centrais (pp)": round(d_c, 2),
            "Δ médio caudas (pp)": round(d_t, 2),
            "Assimetria centro−cauda (pp)": round(gap, 2),
        }
    )

if verdicts:
    vdf = pd.DataFrame(verdicts)
    st.dataframe(vdf, hide_index=True, use_container_width=True)

    worst = max(verdicts, key=lambda v: v["Assimetria centro−cauda (pp)"])
    gap = worst["Assimetria centro−cauda (pp)"]
    if gap > 1.0:
        st.warning(
            f"**{worst['Braço']}** concentra o ganho nas faixas centrais "
            f"(assimetria de {gap:+.2f} pp). Padrão compatível com prior "
            "posicional — confirmar com o teste t pareado abaixo e com o "
            "stress test de translação vertical."
        )
    elif gap < -1.0:
        st.info(
            f"**{worst['Braço']}** ganha mais nas caudas do que no centro "
            f"({gap:+.2f} pp). Inconsistente com prior posicional."
        )
    else:
        st.success(
            "Ganho aproximadamente uniforme entre faixas verticais "
            f"(assimetria máxima {gap:+.2f} pp). A hipótese de prior posicional "
            "não se sustenta nesta métrica."
        )
    st.caption(
        "Assimetria = Δ médio nas faixas centrais menos Δ médio nas caudas. "
        "Com poucos bins e n=3 seeds, trate como indicativo, não conclusivo."
    )

# ── Tabela detalhada ───────────────────────────────────────────────────────
st.subheader("Tabela")

cols = ["arm", "bin", "y_lo", "y_hi", "n_gt", "n_seeds",
        f"{metric}_mean", f"{metric}_std"]
for extra in (f"d{metric}_pp", f"d{metric}_p"):
    if extra in agg_v.columns:
        cols.append(extra)

tbl = agg_v[cols].sort_values(["arm", "bin"]).copy()
tbl = tbl.rename(
    columns={
        "arm": "Braço", "bin": "Bin", "y_lo": "y min", "y_hi": "y máx",
        "n_gt": "n GT", "n_seeds": "seeds",
        f"{metric}_mean": f"{metric} média", f"{metric}_std": "dp",
        f"d{metric}_pp": "Δ vs ref (pp)", f"d{metric}_p": "p (t pareado)",
    }
)
st.dataframe(tbl, hide_index=True, use_container_width=True)

with st.expander("Valores por seed"):
    st.dataframe(
        flat[flat.arm.isin(show_arms)].sort_values(["arm", "seed", "bin"]),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Estratificação por y_center implementada via proxy no campo `area` do COCO, "
    "reproduzindo a semântica de `ignore` do pycocotools nos dois lados (GT fora "
    "da faixa sai do denominador; detecções não pareadas fora da faixa não contam "
    "como FP). AP_small/medium/large não são válidos neste pipeline."
)
