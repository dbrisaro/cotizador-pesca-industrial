import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Cotizador Paramétrico - Anchoveta Perú", layout="wide")

DATA_DIR = Path(__file__).parent / "data"  # actuals updated 2025-04-24: added 2024-T1
BETA = -0.816

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
body {
    font-family: 'Source Sans 3', 'Source Sans Pro', Arial, sans-serif !important;
}
h3 { font-size: 1.3rem !important; margin-bottom: 0.2rem !important; }
[data-testid="stCaptionContainer"] p { font-size: 0.85rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.80rem !important; }
.block-container { padding-top: 3rem !important; padding-bottom: 1rem !important; }
header[data-testid="stHeader"] { height: 2.5rem !important; }
[data-testid="stAppViewContainer"] > section > div { gap: 0.5rem !important; }
div[data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


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
    df = actuals_df if company == all_label else actuals_df[actuals_df["company"] == company]
    sub = df if season == "both" else df[df["tipo"] == season]
    if sub.empty:
        return ""
    years = sorted(sub["year"].unique())
    n = len(years)
    unit = "años" if season == "both" else "temporadas"
    freq = "por año" if season == "both" else "por temporada"
    return f"media {years[0]}-{years[-1]}, {n} {unit} - {freq}"


# ── Load ─────────────────────────────────────────────────────────────────────
sst_df, baselines_df, actuals_df = load_data()

ALL_LABEL = "Todas las empresas"
companies = [ALL_LABEL] + sorted(baselines_df["company"].tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Empresa asegurada")
    company = st.selectbox("Empresa", companies, label_visibility="collapsed")

    st.markdown("#### Temporada")
    season = st.selectbox(
        "Temporada", ["both", "T1", "T2"], label_visibility="collapsed",
        format_func=lambda x: {
            "both": "Ambas (T1 + T2)",
            "T1": "T1 - Primera (abr-jul)",
            "T2": "T2 - Segunda (nov-dic)",
        }[x],
    )

    st.markdown("#### Parámetros del trigger")
    entry = st.slider(
        "Anomalía SST entrada T_ent (°C)", 0.0, 2.0, 0.5, 0.1,
        help="Temperatura a partir de la cual el seguro empieza a pagar. Si la anomalia de la temporada es menor a este valor, el pago es cero. Subirlo hace el contrato menos frecuente y mas barato.",
    )
    exit_ = st.slider(
        "Anomalía SST salida T_sal (°C)", 0.5, 5.0, 2.5, 0.1,
        help="Temperatura a la que el seguro paga el maximo. Entre T_ent y T_sal el pago crece linealmente. Acercarlo a T_ent hace que el seguro sature mas rapido.",
    )
    if exit_ <= entry:
        exit_ = round(entry + 0.1, 1)
        st.warning(f"T_sal ajustado a {exit_:.1f} °C")

    st.markdown("#### Cobertura y precio")
    cov = st.slider(
        "Cobertura contratada (%)", 10, 100, 80, 5,
        help="Que fraccion del baseline quiere asegurar el cliente. Al 80%, si el baseline es 100,000 ton el pago maximo es 80,000 ton. Reducirlo baja la prima y el pago maximo proporcionalmente.",
    ) / 100
    price = st.number_input(
        "Precio referencia (USD/ton)", 50, 1000, 300, 10,
        help="Precio de mercado de la anchoveta usado para convertir toneladas a USD. No afecta las frecuencias ni las fracciones de pago, solo la escala en dolares.",
    )

    st.markdown("#### Gastos y margen")
    factor = st.number_input(
        "Factor de carga", 1.0, 3.0, 1.65, 0.05,
        help="Multiplicador sobre la prima pura para cubrir gastos operativos y margen. Factor 1.65 = loss ratio 60.6% (de cada USD de prima, 61 centavos cubren siniestros y 39 son gastos y margen).",
    )

    with st.expander("¿Cómo funciona el cotizador?"):
        st.markdown("""
**Índice:** anomalía SST promedio de la temporada en Centro Norte (MODIS AQUA, 7.1°S-11°S).
Positivo = mar más cálido que lo normal.

**Pago:** si SST < T_ent no hay pago. Si SST >= T_sal se paga el máximo.
Entre ambos crece linealmente:
> pago = baseline × cobertura × (SST − T_ent) / (T_sal − T_ent)

**Prima pura (AAL):** promedio histórico del pago anual sobre 2002-2025.

**Prima comercial:** Prima pura × factor de carga.
        """)

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
        .set_index(["year", "tipo"])["actual_ton"].to_dict()
    )

baseline = bl_t1 if season == "T1" else bl_t2 if season == "T2" else bl_t1 + bl_t2
max_pay_ton = baseline * cov
max_pay_usd = max_pay_ton * price

if season == "both":
    rows = [
        {"year": int(r["year"]), "tipo": r["tipo"], "sst": r["sst"],
         "baseline_s": bl_t1 if r["tipo"] == "T1" else bl_t2}
        for _, r in sst_df.iterrows()
    ]
else:
    bl_s = bl_t1 if season == "T1" else bl_t2
    rows = [
        {"year": int(r["year"]), "tipo": r["tipo"], "sst": r["sst"], "baseline_s": bl_s}
        for _, r in sst_df[sst_df["tipo"] == season].iterrows()
    ]

for r in rows:
    r["f"]      = payout_frac(r["sst"], entry, exit_)
    r["paton"]  = r["baseline_s"] * r["f"] * cov
    r["pausd"]  = r["paton"] * price
    r["actual"] = co_actuals.get((r["year"], r["tipo"]), None)

if season == "both":
    years_uniq = sorted(set(r["year"] for r in rows))
    aal_ton = np.mean([sum(r["paton"] for r in rows if r["year"] == yr) for yr in years_uniq])
else:
    aal_ton = np.mean([r["paton"] for r in rows])

aal_pct       = aal_ton / (baseline or 1)
pure_prem_usd = aal_ton * price
comm_prem_usd = pure_prem_usd * factor
load_usd      = comm_prem_usd - pure_prem_usd
load_pct      = 1 - 1 / factor
period_label  = baseline_period(company, season, actuals_df, ALL_LABEL)

# ── Header ───────────────────────────────────────────────────────────────────
st.subheader("Cotizador Paramétrico - Seguro de Captura de Anchoveta")
st.caption("Centro Norte (11°S - 7.1°S)  ·  MODIS SST 2002-2025  ·  Datos IHMA 2015-2025")
st.divider()

# ── Baseline ─────────────────────────────────────────────────────────────────
st.metric(
    "Captura de referencia (baseline)",
    f"{baseline:,.0f} ton",
    help=(
        f"{period_label}. "
        "Promedio historico de captura de la empresa segun datos IHMA (2015-2025). "
        "Es el punto de referencia del contrato: define cuanto es 'captura normal'. "
        "Ojo: el seguro no paga por caidas en la captura real, paga segun la anomalia de temperatura del mar, independientemente de lo que haya pescado la empresa ese ano."
    )
)

# ── KPIs ─────────────────────────────────────────────────────────────────────
leverage = max_pay_usd / comm_prem_usd if comm_prem_usd > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prima pura (AAL)", f"USD {fmt_k(pure_prem_usd)}/año",
    help=f"Costo actuarial del contrato: promedio de lo que habria pagado el seguro por ano sobre el historico 2002-2025. Representa el {fmt_pct(aal_pct)} del baseline. No incluye gastos ni margen.")
c2.metric("Prima comercial", f"USD {fmt_k(comm_prem_usd)}/año",
    help=f"Lo que paga el cliente. Es la prima pura x factor de carga ({factor:.2f}). Equivale al {fmt_pct(comm_prem_usd / (max_pay_usd or 1))} de la suma asegurada.")
c3.metric("Suma asegurada", f"USD {fmt_k(max_pay_usd)}",
    help=f"Pago maximo que recibe el cliente si la SST supera T_sal. Equivale a {fmt_k(max_pay_ton)} toneladas al precio de referencia ingresado.")
c4.metric("Cobertura / prima", f"{leverage:.1f}x",
    help=f"Suma asegurada dividida la prima comercial. Por cada USD que paga el cliente, tiene hasta {leverage:.1f} USD de cobertura.")

st.divider()

# ── Chart + Table ─────────────────────────────────────────────────────────────
col_chart, col_table = st.columns([1, 1], gap="large")

with col_chart:
    st.subheader("Curva de pago vs anomalía SST")

    sst_range = np.arange(-1.5, 5.25, 0.05)
    ramp_y    = [baseline * cov * payout_frac(s, entry, exit_) for s in sst_range]
    ols_y     = [baseline * cov * ols_loss_frac(s, BETA) for s in sst_range]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sst_range, y=ols_y, mode="lines", name="OLS referencia",
        line=dict(color="#aaaaaa", width=1.5, dash="dash"),
    ))

    ramp_fill_x = [s for s in sst_range if s >= entry]
    ramp_fill_y = [baseline * cov * payout_frac(s, entry, exit_) for s in ramp_fill_x]
    fig.add_trace(go.Scatter(
        x=ramp_fill_x + ramp_fill_x[::-1],
        y=ramp_fill_y + [0] * len(ramp_fill_x),
        fill="toself", fillcolor="rgba(67,160,71,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=sst_range, y=ramp_y, mode="lines", name="Ramp lineal (pago)",
        line=dict(color="#43A047", width=2.5),
    ))

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
        x=cold_x, y=cold_y, mode="markers", name="Sin activacion (SST < T_ent)",
        marker=dict(color="#90A4AE", size=8, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=warm_x, y=warm_y, mode="markers+text", name="Con pago (SST >= T_ent)",
        text=warm_txt, textposition="top right", textfont=dict(size=9, color="#141414"),
        marker=dict(color="#FF8F00", size=8, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=nino_x, y=nino_y, mode="markers+text", name="El Niño (2015, 2023)",
        text=nino_txt, textposition="top right", textfont=dict(size=9, color="#141414"),
        marker=dict(color="#c62828", size=10, line=dict(color="white", width=1.5)),
    ))

    fig.add_vline(x=entry, line=dict(color="#1565c0", width=1.5, dash="dash"),
                  annotation_text=f"T_ent {entry:.1f}°C", annotation_position="bottom",
                  annotation_font_color="#1565c0", annotation_font_size=10)
    fig.add_vline(x=exit_, line=dict(color="#c62828", width=1.5, dash="dash"),
                  annotation_text=f"T_sal {exit_:.1f}°C", annotation_position="bottom",
                  annotation_font_color="#c62828", annotation_font_size=10)
    fig.add_vline(x=0, line=dict(color="#E0E0E0", width=1, dash="dot"))

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(title="Anomalía SST (°C)", range=[-1.5, 5.2],
                   gridcolor="#F0F0F0", linecolor="#E0E0E0", tickfont=dict(size=10)),
        yaxis=dict(title="Pago (ton)", range=[0, max_pay_ton * 1.15],
                   gridcolor="#F0F0F0", linecolor="#E0E0E0", tickfont=dict(size=10)),
        legend=dict(orientation="v", x=0.01, y=0.99,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E0E0E0", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color="#141414"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Time series chart ─────────────────────────────────────────────────────
    st.subheader("Pago paramétrico y pérdida real por temporada")

    ts_rows = sorted(rows, key=lambda r: (r["year"], r["tipo"]))
    x_labels = [f"{r['year']} {r['tipo']}" if season == "both" else str(r["year"]) for r in ts_rows]

    bar_colors = ["#43A047" if r["f"] > 0 else "#E0E0E0" for r in ts_rows]

    loss_x, loss_y = [], []
    for r, lbl in zip(ts_rows, x_labels):
        if r["actual"] is not None:
            loss = r["baseline_s"] - r["actual"]
            loss_x.append(lbl)
            loss_y.append(max(loss, 0))

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=x_labels,
        y=[r["paton"] if r["f"] > 0 else None for r in ts_rows],
        name="Pago paramétrico",
        marker_color="#43A047", marker_line_width=0,
    ))
    fig2.add_trace(go.Bar(
        x=x_labels,
        y=[r["paton"] if r["f"] == 0 else None for r in ts_rows],
        name="Sin activación",
        marker_color="#E0E0E0", marker_line_width=0,
    ))
    fig2.add_trace(go.Bar(
        x=loss_x, y=loss_y,
        name="Pérdida real",
        marker_color="#c62828", marker_line_width=0,
    ))

    fig2.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#F0F0F0", linecolor="#E0E0E0", tickfont=dict(size=9), tickangle=-45),
        yaxis=dict(title="Toneladas", gridcolor="#F0F0F0", linecolor="#E0E0E0", tickfont=dict(size=10)),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=10),
                    bgcolor="rgba(255,255,255,0)"),
        margin=dict(l=10, r=10, t=30, b=60),
        height=320,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color="#141414"),
        barmode="group",
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_table:
    st.subheader("Temporadas históricas (SST 2002-2025)")

    sorted_rows = sorted(rows, key=lambda r: (-r["year"], r["tipo"]))

    download_df = pd.DataFrame([{
        "año":              r["year"],
        "temporada":        r["tipo"],
        "sst_anomalia":     round(r["sst"], 2),
        "f_pago":           round(r["f"], 2),
        "pago_ton":         round(r["paton"], 1),
        "pago_usd":         round(r["pausd"], 0),
        "captura_real_ton": r["actual"] if r["actual"] is not None else "",
    } for r in sorted_rows])

    display_df = download_df.rename(columns={
        "año": "Año", "temporada": "Temp.", "sst_anomalia": "SST (°C)",
        "f_pago": "f pago", "pago_ton": "Pago (ton)",
        "pago_usd": "Pago (USD)", "captura_real_ton": "Captura real",
    })

    def color_sst(val):
        try:
            v = float(val)
            if v >= entry:
                return "color: #c62828; font-weight: bold"
            return "color: #1565c0"
        except (ValueError, TypeError):
            return ""

    def color_pago(val):
        try:
            if float(val) > 0:
                return "color: #43A047; font-weight: bold"
        except (ValueError, TypeError):
            pass
        return "color: #9E9E9E"

    styled = (
        display_df.style
        .format({
            "SST (°C)":   "{:.2f}",
            "f pago":     "{:.2f}",
            "Pago (ton)": "{:,.0f}",
            "Pago (USD)": "{:,.0f}",
            "Captura real": lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else v,
        })
        .map(color_sst, subset=["SST (°C)"])
        .map(color_pago, subset=["Pago (ton)", "Pago (USD)"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    st.caption("Captura real disponible 2015-2025. SST: MODIS AQUA Centro Norte.")
