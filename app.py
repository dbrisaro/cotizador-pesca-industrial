import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Cotizador Pesca Industrial", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
BETA = -0.816  # OLS reference curve


@st.cache_data
def load_data():
    sst = pd.read_csv(DATA_DIR / "cotizador_sst_by_season.csv")
    baselines = pd.read_csv(DATA_DIR / "cotizador_company_baselines.csv")
    actuals = pd.read_csv(DATA_DIR / "cotizador_company_actuals.csv")
    return sst, baselines, actuals


def payout_frac(sst, entry, exit_):
    if sst <= entry:
        return 0.0
    return min(1.0, (sst - entry) / (exit_ - entry))


def ols_loss_frac(sst, beta):
    if sst <= 0:
        return 0.0
    return max(0.0, 1 - np.exp(beta * sst))


def fmt_k(n):
    if n is None or np.isnan(n):
        return "-"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return f"{n:.0f}"


def fmt_pct(n):
    return f"{n*100:.1f}%"


def baseline_period(company, season, actuals_df, all_label):
    if company == all_label:
        df = actuals_df
    else:
        df = actuals_df[actuals_df["company"] == company]

    if season == "both":
        sub = df
    else:
        sub = df[df["tipo"] == season]

    if sub.empty:
        return ""
    years = sorted(sub["year"].unique())
    n = len(years)
    unit = "anos" if season == "both" else "temporadas"
    freq = "por ano" if season == "both" else "por temporada"
    return f"media {years[0]}-{years[-1]}, {n} {unit} - {freq}"


# ── Load ─────────────────────────────────────────────────────────────────────
sst_df, baselines_df, actuals_df = load_data()

ALL_LABEL = "Todas las empresas"
companies = [ALL_LABEL] + sorted(baselines_df["company"].tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Parametros")

    company = st.selectbox("Empresa", companies)
    season = st.selectbox(
        "Temporada",
        ["both", "T1", "T2"],
        format_func=lambda x: {
            "both": "Ambas (T1 + T2)",
            "T1": "T1 - Primera (abr-jul)",
            "T2": "T2 - Segunda (nov-dic)",
        }[x],
    )

    st.divider()
    st.subheader("Trigger")
    entry = st.slider(
        "Anomalia SST entrada T_ent (°C)", 0.0, 2.0, 0.5, 0.1,
        help=(
            "Umbral a partir del cual el seguro empieza a pagar. "
            "Si la anomalia SST de la temporada no supera este valor, el pago es cero. "
            "Subirlo hace el contrato mas barato pero menos frecuente."
        ),
    )
    exit_ = st.slider(
        "Anomalia SST salida T_sal (°C)", 0.5, 5.0, 2.5, 0.1,
        help=(
            "Nivel de anomalia SST al que se alcanza el pago maximo (100%). "
            "Entre T_ent y T_sal el pago crece linealmente. "
            "Cuanto mas cerca este de T_ent, mas rapido satura el pago; "
            "cuanto mas lejos, mas gradual."
        ),
    )
    if exit_ <= entry:
        exit_ = round(entry + 0.1, 1)
        st.warning(f"T_sal ajustado a {exit_:.1f} °C")

    st.divider()
    st.subheader("Cobertura y precio")
    cov = st.slider(
        "Cobertura contratada (%)", 10, 100, 80, 5,
        help=(
            "Fraccion del baseline de captura cubierta por el contrato. "
            "El pago maximo es cobertura x baseline x precio. "
            "Bajarla reduce prima y pago maximo proporcionalmente."
        ),
    ) / 100
    price = st.number_input(
        "Precio referencia (USD/ton)", 50, 1000, 300, 10,
        help=(
            "Precio de la anchoveta usado para convertir toneladas a USD. "
            "Solo afecta los montos en dolares; no cambia frecuencias ni fracciones de pago."
        ),
    )

    st.divider()
    st.subheader("Gastos y margen")
    factor = st.number_input(
        "Factor de carga", 1.0, 3.0, 1.65, 0.05,
        help=(
            "Prima comercial = Prima pura x factor. "
            "Un factor de 1.65 implica un loss ratio del 60.6% "
            "(es decir, el 60.6% de la prima cubre siniestros y el resto gastos y margen)."
        ),
    )

    st.divider()
    with st.expander("Como funciona el cotizador"):
        st.markdown(
            """
**Indice:** la anomalia SST promedio de la temporada en la zona Centro Norte del Peru
(MODIS AQUA, 7.1°S-11°S). Positivo = mar mas calido que lo normal.

**Pago:** si SST < T_ent, no hay pago. Si SST >= T_sal, se paga el maximo.
Entre ambos el pago crece linealmente:

> pago = baseline x cobertura x (SST - T_ent) / (T_sal - T_ent)

**Prima pura (AAL):** promedio historico del pago anual sobre 2002-2025.

**Prima comercial:** Prima pura x factor de carga.
El factor cubre gastos operativos, comision y margen de la aseguradora.

**Que mueve el precio:**
- Subir T_ent → el contrato se activa menos seguido → prima baja
- Bajar T_sal → el pago satura mas rapido → prima sube
- Bajar cobertura → pago maximo menor → prima baja proporcionalmente
- Subir precio USD/ton → misma frecuencia, mayor monto en USD
            """
        )

# ── Compute ───────────────────────────────────────────────────────────────────
if company == ALL_LABEL:
    bl_t1 = baselines_df["baseline_t1"].sum()
    bl_t2 = baselines_df["baseline_t2"].sum()
    co_actuals = actuals_df.groupby(["year", "tipo"])["actual_ton"].sum().to_dict()
else:
    bl_row = baselines_df[baselines_df["company"] == company].iloc[0]
    bl_t1 = bl_row["baseline_t1"]
    bl_t2 = bl_row["baseline_t2"]
    co_actuals = (
        actuals_df[actuals_df["company"] == company]
        .set_index(["year", "tipo"])["actual_ton"]
        .to_dict()
    )

if season == "T1":
    baseline = bl_t1
elif season == "T2":
    baseline = bl_t2
else:
    baseline = bl_t1 + bl_t2

max_pay_ton = baseline * cov
max_pay_usd = max_pay_ton * price

# Build season rows
if season == "both":
    rows = []
    for _, r in sst_df.iterrows():
        bl_s = bl_t1 if r["tipo"] == "T1" else bl_t2
        rows.append({"year": int(r["year"]), "tipo": r["tipo"], "sst": r["sst"], "baseline_s": bl_s})
else:
    rows = []
    for _, r in sst_df[sst_df["tipo"] == season].iterrows():
        bl_s = bl_t1 if season == "T1" else bl_t2
        rows.append({"year": int(r["year"]), "tipo": r["tipo"], "sst": r["sst"], "baseline_s": bl_s})

for r in rows:
    r["f"] = payout_frac(r["sst"], entry, exit_)
    r["paton"] = r["baseline_s"] * r["f"] * cov
    r["pausd"] = r["paton"] * price
    r["actual"] = co_actuals.get((r["year"], r["tipo"]), None)

# AAL
if season == "both":
    years_uniq = sorted(set(r["year"] for r in rows))
    annual = [sum(r["paton"] for r in rows if r["year"] == yr) for yr in years_uniq]
    aal_ton = np.mean(annual)
else:
    aal_ton = np.mean([r["paton"] for r in rows])

aal_pct = aal_ton / (baseline or 1)
pure_prem_usd = aal_ton * price
comm_prem_usd = pure_prem_usd * factor
load_usd = comm_prem_usd - pure_prem_usd
load_pct = 1 - 1 / factor

# Baseline period label
period_label = baseline_period(company, season, actuals_df, ALL_LABEL)

# ── Baseline box ─────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='background:#f5f5f5;border-radius:6px;padding:10px 16px;margin-bottom:16px;"
    f"font-size:14px;color:#555'>Captura de referencia (baseline): "
    f"<strong style='font-size:18px;color:#111'>{baseline:,.0f} ton</strong>"
    f"<span style='margin-left:12px;font-size:12px;color:#999'>{period_label}</span></div>",
    unsafe_allow_html=True,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pago maximo", f"{fmt_k(max_pay_ton)} ton", f"USD {fmt_k(max_pay_usd)}")
k2.metric("Prima pura (AAL)", f"USD {fmt_k(pure_prem_usd)}/ano", f"{fmt_pct(aal_pct)} del baseline - {fmt_k(aal_ton)} ton")
k3.metric("Prima comercial", f"USD {fmt_k(comm_prem_usd)}/ano", f"tasa {fmt_pct(comm_prem_usd / (max_pay_usd or 1))} s/ suma asegurada")
k4.metric("Carga (gastos + margen)", f"USD {fmt_k(load_usd)}/ano", f"{fmt_pct(load_pct)} de la prima comercial")

st.divider()

# ── Chart + Table ─────────────────────────────────────────────────────────────
col_chart, col_table = st.columns([1, 1], gap="large")

with col_chart:
    st.subheader("Curva de pago vs anomalia SST")

    sst_range = np.arange(-1.5, 5.25, 0.05)
    ramp_y = [baseline * cov * payout_frac(s, entry, exit_) for s in sst_range]
    ols_y = [baseline * cov * ols_loss_frac(s, BETA) for s in sst_range]

    fig = go.Figure()

    # OLS reference
    fig.add_trace(go.Scatter(
        x=sst_range, y=ols_y,
        mode="lines", name="OLS referencia",
        line=dict(color="#aaaaaa", width=1.5, dash="dash"),
    ))

    # Ramp fill
    ramp_fill_x = [s for s in sst_range if s >= entry]
    ramp_fill_y = [baseline * cov * payout_frac(s, entry, exit_) for s in ramp_fill_x]
    fig.add_trace(go.Scatter(
        x=ramp_fill_x + ramp_fill_x[::-1],
        y=ramp_fill_y + [0] * len(ramp_fill_x),
        fill="toself", fillcolor="rgba(67,160,71,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Ramp curve
    fig.add_trace(go.Scatter(
        x=sst_range, y=ramp_y,
        mode="lines", name="Ramp lineal (pago)",
        line=dict(color="#43A047", width=2.5),
    ))

    # Historical dots
    nino_years = {2015, 2023}
    warm_x, warm_y, warm_txt = [], [], []
    cold_x, cold_y = [], []
    nino_x, nino_y, nino_txt = [], [], []

    for r in rows:
        label = f"{r['year']} {r['tipo']}"
        if r["year"] in nino_years:
            nino_x.append(r["sst"]); nino_y.append(r["paton"]); nino_txt.append(label)
        elif r["sst"] >= entry:
            warm_x.append(r["sst"]); warm_y.append(r["paton"]); warm_txt.append(label)
        else:
            cold_x.append(r["sst"]); cold_y.append(r["paton"])

    fig.add_trace(go.Scatter(
        x=cold_x, y=cold_y, mode="markers", name="Temporada fria",
        marker=dict(color="#90A4AE", size=8, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=warm_x, y=warm_y, mode="markers+text", name="Temporada calida",
        text=warm_txt, textposition="top right", textfont=dict(size=9),
        marker=dict(color="#FF8F00", size=8, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=nino_x, y=nino_y, mode="markers+text", name="El Nino (2015, 2023)",
        text=nino_txt, textposition="top right", textfont=dict(size=9),
        marker=dict(color="#D32F2F", size=10, line=dict(color="white", width=1.5)),
    ))

    fig.add_vline(x=entry, line=dict(color="#1976D2", width=1.5, dash="dash"),
                  annotation_text=f"T_ent {entry:.1f}°C", annotation_position="bottom")
    fig.add_vline(x=exit_, line=dict(color="#D32F2F", width=1.5, dash="dash"),
                  annotation_text=f"T_sal {exit_:.1f}°C", annotation_position="bottom")
    fig.add_vline(x=0, line=dict(color="#cccccc", width=1, dash="dot"))

    fig.update_layout(
        xaxis_title="Anomalia SST (°C)",
        yaxis_title="Pago (ton)",
        xaxis=dict(range=[-1.5, 5.2]),
        yaxis=dict(range=[0, max_pay_ton * 1.15]),
        legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("Temporadas historicas (SST 2002-2025)")

    sorted_rows = sorted(rows, key=lambda r: (-r["year"], r["tipo"]))
    table_data = []
    for r in sorted_rows:
        table_data.append({
            "Ano": r["year"],
            "Temp.": r["tipo"],
            "SST (°C)": round(r["sst"], 2),
            "f pago": f"{r['f']*100:.1f}%" if r["f"] > 0 else "0%",
            "Pago (ton)": f"{r['paton']:,.0f}" if r["f"] > 0 else "-",
            "Pago (USD)": f"USD {fmt_k(r['pausd'])}" if r["f"] > 0 else "-",
            "Captura real": f"{r['actual']:,.0f} ton" if r["actual"] is not None else "-",
        })

    st.dataframe(
        pd.DataFrame(table_data),
        hide_index=True,
        use_container_width=True,
        height=400,
    )
    st.caption("Captura real disponible 2015-2025. SST: MODIS AQUA Centro Norte.")
