"""
Generate regional CSVs for the cotizador app.
Produces:
  - cotizador_sst_by_season.csv    (year, tipo, region, sst)
  - cotizador_company_actuals.csv  (company, year, tipo, region, actual_ton)
  - cotizador_company_baselines.csv (company, region, baseline_t1, baseline_t2)
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("/home/jupyter-daniela/suyana/peru_production/features")
CALAS_CSV    = Path("/home/jupyter-daniela/suyana/peru_production/outputs/calas_all_data.csv")
OUT_DIR      = Path("/home/jupyter-daniela/cotizador_pesca_industrial/data")
SST_YEARS    = list(range(2002, 2027))

# ── Region definitions ────────────────────────────────────────────────────────
REGIONS = {
    "Norte":        (-7.1,   None),   # lat > -7.1
    "Centro Norte": (-11.0, -7.1),    # -11.0 < lat <= -7.1
    "Centro Sur":   (-15.8, -11.0),   # -15.8 < lat <= -11.0
    "Todas":        (-15.8,  None),   # lat > -15.8
}

# Season DOY windows
T1_DOYS = (91,  212)   # Apr 1 – Jul 31
T2_DOYS = (305, 365)   # Nov 1 – Dec 31
MIN_DAYS = 10

# ── Company name normalization ─────────────────────────────────────────────────
NORMALIZE = {
    "DIAMANTE":                "PESQUERA DIAMANTE S.A.",
    "Pesquera Diamante S.A.":  "PESQUERA DIAMANTE S.A.",
    "PESQUERA DIAMANTE S.A.":  "PESQUERA DIAMANTE S.A.",
    "EXALMAR-CENTINELA":       "PESQUERA CENTINELA S.A.C",
    "TASA ":                   "TASA",
}

COTIZADOR_COMPANIES = {
    "AUSTRAL GROUP SAA",
    "CFG-COPEINCA",
    "HAYDUK",
    "PESQUERA CENTINELA S.A.C",
    "PESQUERA DIAMANTE S.A.",
    "PESQUERA EXALMAR S.A.A.",
    "TASA",
}

# ── Helper: filter calas by region ───────────────────────────────────────────
def filter_lat(df, lat_min, lat_max):
    mask = df["latitud"] > lat_min
    if lat_max is not None:
        mask &= (df["latitud"] <= lat_max)
    return df[mask]


# ── Load calas data ───────────────────────────────────────────────────────────
print("Loading calas_all_data.csv ...")
calas_raw = pd.read_csv(CALAS_CSV, low_memory=False,
                        usecols=["empresa", "fecha_cala", "declarado_tm", "latitud", "longitud"])
calas_raw["latitud"]     = pd.to_numeric(calas_raw["latitud"],     errors="coerce")
calas_raw["longitud"]    = pd.to_numeric(calas_raw["longitud"],    errors="coerce")
calas_raw["declarado_tm"]= pd.to_numeric(calas_raw["declarado_tm"],errors="coerce")
calas_raw["fecha_cala"]  = pd.to_datetime(calas_raw["fecha_cala"], errors="coerce")
calas_raw = calas_raw.dropna(subset=["latitud", "longitud", "fecha_cala", "declarado_tm"])

# Normalize company names
calas_raw["empresa"] = calas_raw["empresa"].map(lambda x: NORMALIZE.get(x, x))

# Add year, month, DOY
calas_raw["year"]  = calas_raw["fecha_cala"].dt.year
calas_raw["month"] = calas_raw["fecha_cala"].dt.month
calas_raw["doy"]   = calas_raw["fecha_cala"].dt.dayofyear

# Season label
def season_label(month):
    if 4 <= month <= 7:
        return "T1"
    elif month in (11, 12):
        return "T2"
    return None

calas_raw["tipo"] = calas_raw["month"].map(season_label)
calas_raw = calas_raw.dropna(subset=["tipo"])

print(f"  {len(calas_raw):,} cala rows after cleaning")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Build fishing polygons per region (5th-95th pct lon per 1-deg lat band)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding fishing polygons ...")

def fishing_polygon(df):
    """Return dict: lat_floor -> (lon_min, lon_max) using 5th-95th pct."""
    df = df.copy()
    df["lat_floor"] = np.floor(df["latitud"]).astype(int)
    poly = {}
    for lat_floor, grp in df.groupby("lat_floor"):
        lo5  = grp["longitud"].quantile(0.05)
        lo95 = grp["longitud"].quantile(0.95)
        poly[lat_floor] = (lo5, lo95)
    return poly

# Build polygon for each region
polygons = {}
for region, (lat_min, lat_max) in REGIONS.items():
    sub = filter_lat(calas_raw, lat_min, lat_max)
    polygons[region] = fishing_polygon(sub)
    print(f"  {region}: {len(polygons[region])} lat bands, {len(sub):,} calas")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: SST by season per region
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing SST by season and region ...")

def mask_polygon(da_2d, lat_coords, lon_coords, poly):
    """
    Returns a 2D boolean mask (lat x lon) where pixels are inside the polygon.
    poly: dict lat_floor -> (lon_min, lon_max)
    """
    mask = np.zeros((len(lat_coords), len(lon_coords)), dtype=bool)
    for i, la in enumerate(lat_coords):
        lf = int(np.floor(la))
        if lf in poly:
            lo_min, lo_max = poly[lf]
            for j, lo in enumerate(lon_coords):
                if lo_min <= lo <= lo_max:
                    mask[i, j] = True
    return mask


def compute_sst_season(year, doy_start, doy_end, lat_min, lat_max, poly):
    """Compute mean SST anomaly for a season window using the fishing polygon."""
    nc_path = FEATURES_DIR / f"sst_anomaly_daily_{year}.nc"
    if not nc_path.exists():
        return np.nan

    ds = xr.open_dataset(nc_path)
    # Filter to region lat band
    lat_mask = ds.lat.values > lat_min
    if lat_max is not None:
        lat_mask &= (ds.lat.values <= lat_max)
    ds_reg = ds.isel(lat=np.where(lat_mask)[0])

    # Filter to season DOYs
    doy_arr = ds_reg.time.dt.dayofyear.values
    t_mask  = (doy_arr >= doy_start) & (doy_arr <= doy_end)
    ds_seas = ds_reg.isel(time=np.where(t_mask)[0])

    n_days = int(t_mask.sum())
    ds.close()

    if n_days < MIN_DAYS:
        return np.nan

    # Build spatial mask from polygon
    lat_vals = ds_seas.lat.values
    lon_vals = ds_seas.lon.values
    sp_mask  = mask_polygon(None, lat_vals, lon_vals, poly)

    if sp_mask.sum() == 0:
        return np.nan

    # Mean over valid pixels across all days
    sst_data = ds_seas["sst_anomaly"].values  # (time, lat, lon)
    # Apply spatial mask
    sp_mask_3d = np.broadcast_to(sp_mask[np.newaxis], sst_data.shape)
    valid = np.where(sp_mask_3d, sst_data, np.nan)
    daily_means = np.nanmean(valid.reshape(n_days, -1), axis=1)
    return float(np.nanmean(daily_means))


sst_rows = []
for region, (lat_min, lat_max) in REGIONS.items():
    poly = polygons[region]
    for year in SST_YEARS:
        for tipo, (d0, d1) in [("T1", T1_DOYS), ("T2", T2_DOYS)]:
            val = compute_sst_season(year, d0, d1, lat_min, lat_max, poly)
            sst_rows.append({"year": year, "tipo": tipo, "region": region, "sst": round(val, 3) if not np.isnan(val) else np.nan})
        print(f"  {region} {year} done")

sst_df = pd.DataFrame(sst_rows)
sst_df = sst_df.dropna(subset=["sst"])
out_path = OUT_DIR / "cotizador_sst_by_season.csv"
sst_df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}  ({len(sst_df)} rows)")
print(sst_df.groupby(["region", "tipo"]).size().to_string())
print()
print("Sample SST values:")
print(sst_df[sst_df["year"].isin([2015, 2023])].to_string())


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Company actuals per region
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing company actuals by region ...")

# Only cotizador companies
calas_co = calas_raw[calas_raw["empresa"].isin(COTIZADOR_COMPANIES)].copy()

actuals_rows = []
for region, (lat_min, lat_max) in REGIONS.items():
    if region == "Todas":
        # Sum across all three sub-regions - compute from full filter
        sub = filter_lat(calas_co, lat_min, lat_max)
    else:
        sub = filter_lat(calas_co, lat_min, lat_max)

    grp = (sub
           .groupby(["empresa", "year", "tipo"])["declarado_tm"]
           .sum()
           .reset_index()
           .rename(columns={"empresa": "company", "declarado_tm": "actual_ton"}))
    grp["region"] = region
    actuals_rows.append(grp)

actuals_df = pd.concat(actuals_rows, ignore_index=True)
actuals_df = actuals_df[["company", "year", "tipo", "region", "actual_ton"]]
out_path = OUT_DIR / "cotizador_company_actuals.csv"
actuals_df.to_csv(out_path, index=False)
print(f"Saved {out_path}  ({len(actuals_df)} rows)")
print(actuals_df.groupby("region").size().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Company baselines per region
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing company baselines by region ...")

baseline_rows = []
for region in REGIONS:
    sub = actuals_df[actuals_df["region"] == region]
    for company in COTIZADOR_COMPANIES:
        co_sub = sub[sub["company"] == company]
        if co_sub.empty:
            bl_t1 = 0.0
            bl_t2 = 0.0
        else:
            t1 = co_sub[co_sub["tipo"] == "T1"]
            t2 = co_sub[co_sub["tipo"] == "T2"]
            bl_t1 = t1["actual_ton"].mean() if len(t1) > 0 else 0.0
            bl_t2 = t2["actual_ton"].mean() if len(t2) > 0 else 0.0
        baseline_rows.append({
            "company":     company,
            "region":      region,
            "baseline_t1": round(bl_t1, 4),
            "baseline_t2": round(bl_t2, 4),
        })

baselines_df = pd.DataFrame(baseline_rows)
out_path = OUT_DIR / "cotizador_company_baselines.csv"
baselines_df.to_csv(out_path, index=False)
print(f"Saved {out_path}  ({len(baselines_df)} rows)")
print()
print("Baselines sample (Centro Norte):")
print(baselines_df[baselines_df["region"] == "Centro Norte"].to_string(index=False))
print()
print("Baselines sample (Todas):")
print(baselines_df[baselines_df["region"] == "Todas"].to_string(index=False))

print("\nDone.")
