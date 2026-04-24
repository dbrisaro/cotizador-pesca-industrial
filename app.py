import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Cotizador Paramétrico — Anchoveta Perú", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
BETA = -0.816

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page */
.stApp { background-color: #F7F7F5; font-family: 'Helvetica Neue', Arial, sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-size: 11px !important;
    color: #555555 !important;
}

/* Sidebar section titles */
.section-title {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    color: #9E9E9E; letter-spacing: 0.8px;
    margin: 16px 0 8px; padding-bottom: 4px;
    border-bottom: 1px solid #E0E0E0;
}

/* Slider accent color */
.stSlider [data-baseweb="slider"] [role="slider"] { background: #43A047 !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color: #43A047 !important; }

/* Main content top padding */
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* Header */
.app-header {
    background: #111111;
    color: white;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 3px solid #43A047;
    margin: -1rem -1rem 1rem -1rem;
}
.app-header h1 { font-size: 15px; font-weight: 600; letter-spacing: 0.3px; color: white; margin: 0; }
.app-header span { font-size: 11px; color: #9E9E9E; }
.app-header .brand { font-size: 12px; font-weight: 700; color: #43A047; margin-left: auto; letter-spacing: 1px; }

/* Baseline box */
.baseline-box {
    background: #F7F7F5;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 16px;
    border-left: 3px solid #43A047;
}
.baseline-box .bl-label { font-size: 10px; color: #9E9E9E; margin-bottom: 2px; }
.baseline-box .bl-val { font-size: 18px; font-weight: 700; color: #141414; }
.baseline-box .bl-period { font-size: 11px; color: #9E9E9E; margin-top: 2px; }

/* KPI strip */
.kpi-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: #E0E0E0; border: 1px solid #E0E0E0; border-radius: 6px; overflow: hidden; margin-bottom: 16px; }
.kpi { background: #FFFFFF; padding: 12px 16px; }
.kpi .kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.6px; color: #9E9E9E; margin-bottom: 4px; }
.kpi .kpi-val { font-size: 20px; font-weight: 700; color: #141414; }
.kpi .kpi-val.green { color: #43A047; }
.kpi .kpi-sub { font-size: 11px; color: #555555; margin-top: 2px; }

/* Panel titles */
.panel-title {
    font-size: 11px; font-weight: 700; color: #555555;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 10px;
}

/* Custom table */
.styled-table { width: 100%; border-collapse: collapse; font-size: 11px; border: 1px solid #E0E0E0; border-radius: 4px; overflow: hidden; }
.styled-table thead th {
    background: #111111; color: white; padding: 6px 8px;
    text-align: left; font-weight: 600; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.4px;
}
.styled-table tbody tr:nth-child(even) { background: #F7F7F5; }
.styled-table tbody tr.loss-row { background: #fff8f0 !important; }
.styled-table tbody td { padding: 5px 8px; border-bottom: 1px solid #E0E0E0; color: #141414; }
.styled-table .sst-warm { color: #c62828; font-weight: 600; }
.styled-table .sst-cool { color: #1565c0; }
.styled-table .payout-pos { color: #43A047; font-weight: 700; }
.styled-table .payout-zero { color: #9E9E9E; }

.note-text { font-size: 10px; color: #9E9E9E; margin-top: 8px; }
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
    unit = "anos" if season == "both" else "temporadas"
    freq = "por ano" if season == "both" else "por temporada"
    return f"media {years[0]}-{years[-1]}, {n} {unit} - {freq}"


# ── Load ─────────────────────────────────────────────────────────────────────
sst_df, baselines_df, actuals_df = load_data()

ALL_LABEL = "Todas las empresas"
companies = [ALL_LABEL] + sorted(baselines_df["company"].tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">Empresa asegurada</div>', unsafe_allow_html=True)
    company = st.selectbox("Empresa", companies, label_visibility="collapsed")

    st.markdown('<div class="section-title">Temporada</div>', unsafe_allow_html=True)
    season = st.selectbox(
        "Temporada", ["both", "T1", "T2"], label_visibility="collapsed",
        format_func=lambda x: {
            "both": "Ambas (T1 + T2)",
            "T1": "T1 - Primera (abr-jul)",
            "T2": "T2 - Segunda (nov-dic)",
        }[x],
    )

    st.markdown('<div class="section-title">Parametros del trigger</div>', unsafe_allow_html=True)
    entry = st.slider(
        "Anomalia SST entrada T_ent (°C)", 0.0, 2.0, 0.5, 0.1,
        help="Umbral a partir del cual el seguro empieza a pagar. Si la anomalia SST no supera este valor, el pago es cero. Subirlo hace el contrato mas barato pero menos frecuente.",
    )
    exit_ = st.slider(
        "Anomalia SST salida T_sal (°C)", 0.5, 5.0, 2.5, 0.1,
        help="Nivel de anomalia SST al que se alcanza el pago maximo. Entre T_ent y T_sal el pago crece linealmente. Cuanto mas cerca de T_ent, mas rapido satura.",
    )
    if exit_ <= entry:
        exit_ = round(entry + 0.1, 1)
        st.warning(f"T_sal ajustado a {exit_:.1f} °C")

    st.markdown('<div class="section-title">Cobertura y precio</div>', unsafe_allow_html=True)
    cov = st.slider(
        "Cobertura contratada (%)", 10, 100, 80, 5,
        help="Fraccion del baseline cubierta. El pago maximo es cobertura x baseline x precio. Bajarla reduce prima y pago maximo proporcionalmente.",
    ) / 100
    price = st.number_input(
        "Precio referencia (USD/ton)", 50, 1000, 300, 10,
        help="Precio de la anchoveta para convertir toneladas a USD. No cambia frecuencias ni fracciones de pago.",
    )

    st.markdown('<div class="section-title">Gastos y margen</div>', unsafe_allow_html=True)
    factor = st.number_input(
        "Factor de carga", 1.0, 3.0, 1.65, 0.05,
        help="Prima comercial = Prima pura x factor. Un factor de 1.65 implica un loss ratio del 60.6%.",
    )

    with st.expander("Como funciona el cotizador"):
        st.markdown("""
**Indice:** anomalia SST promedio de la temporada en Centro Norte (MODIS AQUA, 7.1°S-11°S).
Positivo = mar mas calido que lo normal.

**Pago:** si SST < T_ent no hay pago. Si SST >= T_sal se paga el maximo.
Entre ambos crece linealmente:
> pago = baseline × cobertura × (SST − T_ent) / (T_sal − T_ent)

**Prima pura (AAL):** promedio historico del pago anual sobre 2002-2025.

**Prima comercial:** Prima pura × factor de carga.

**Que mueve el precio:**
- Subir T_ent → menos activaciones → prima baja
- Bajar T_sal → satura mas rapido → prima sube
- Bajar cobertura → prima baja proporcionalmente
- Subir precio USD/ton → misma frecuencia, mayor monto en USD
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
st.markdown(f"""
<div class="app-header">
  <div>
    <h1>Cotizador Parametrico — Seguro de Captura de Anchoveta</h1>
    <span>Centro Norte (11°S – 7.1°S) · MODIS SST 2002–2025 · Datos IHMA 2015–2025</span>
  </div>
  <div class="brand">SUYANA</div>
</div>
""", unsafe_allow_html=True)

# ── Baseline box ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="baseline-box">
  <div class="bl-label">Captura de referencia (baseline)</div>
  <div class="bl-val">{baseline:,.0f} ton</div>
  <div class="bl-period">{period_label}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Pago maximo</div>
    <div class="kpi-val">{fmt_k(max_pay_ton)} ton</div>
    <div class="kpi-sub">USD {fmt_k(max_pay_usd)}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Prima pura (AAL)</div>
    <div class="kpi-val green">USD {fmt_k(pure_prem_usd)}/ano</div>
    <div class="kpi-sub">{fmt_pct(aal_pct)} del baseline · {fmt_k(aal_ton)} ton</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Prima comercial</div>
    <div class="kpi-val">USD {fmt_k(comm_prem_usd)}/ano</div>
    <div class="kpi-sub">tasa {fmt_pct(comm_prem_usd / (max_pay_usd or 1))} s/ suma asegurada</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Carga (gastos + margen)</div>
    <div class="kpi-val">USD {fmt_k(load_usd)}/ano</div>
    <div class="kpi-sub">{fmt_pct(load_pct)} de la prima comercial</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chart + Table ─────────────────────────────────────────────────────────────
col_chart, col_table = st.columns([1, 1], gap="large")

with col_chart:
    st.markdown('<div class="panel-title">Curva de pago vs anomalia SST</div>', unsafe_allow_html=True)

    sst_range  = np.arange(-1.5, 5.25, 0.05)
    ramp_y     = [baseline * cov * payout_frac(s, entry, exit_) for s in sst_range]
    ols_y      = [baseline * cov * ols_loss_frac(s, BETA) for s in sst_range]

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
        x=cold_x, y=cold_y, mode="markers", name="Temporada fria",
        marker=dict(color="#90A4AE", size=8, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=warm_x, y=warm_y, mode="markers+text", name="Temporada calida",
        text=warm_txt, textposition="top right", textfont=dict(size=9, color="#141414"),
        marker=dict(color="#FF8F00", size=8, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=nino_x, y=nino_y, mode="markers+text", name="El Nino (2015, 2023)",
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
        xaxis=dict(title="Anomalia SST (°C)", range=[-1.5, 5.2],
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

with col_table:
    st.markdown('<div class="panel-title">Temporadas historicas (SST 2002-2025)</div>', unsafe_allow_html=True)

    sorted_rows = sorted(rows, key=lambda r: (-r["year"], r["tipo"]))

    rows_html = ""
    for r in sorted_rows:
        loss_cls = "loss-row" if r["f"] > 0 else ""
        sst_cls  = "sst-warm" if r["sst"] >= entry else "sst-cool"
        f_str    = f"{r['f']*100:.1f}%" if r["f"] > 0 else '<span class="payout-zero">0%</span>'
        ton_str  = f'<span class="payout-pos">{r["paton"]:,.0f}</span>' if r["f"] > 0 else '<span class="payout-zero">-</span>'
        usd_str  = f'<span class="payout-pos">USD {fmt_k(r["pausd"])}</span>' if r["f"] > 0 else '<span class="payout-zero">-</span>'
        act_str  = f'{r["actual"]:,.0f} ton' if r["actual"] is not None else "-"
        rows_html += f"""
        <tr class="{loss_cls}">
          <td>{r['year']}</td><td>{r['tipo']}</td>
          <td class="{sst_cls}">{r['sst']:.2f}</td>
          <td>{f_str}</td><td>{ton_str}</td><td>{usd_str}</td><td>{act_str}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
      <thead>
        <tr>
          <th>Ano</th><th>Temp.</th><th>SST (°C)</th>
          <th>f pago</th><th>Pago (ton)</th><th>Pago (USD)</th><th>Captura real</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="note-text">Captura real disponible 2015-2025. SST: MODIS AQUA Centro Norte.</div>
    """, unsafe_allow_html=True)
