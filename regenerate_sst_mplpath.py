"""
Regenerate cotizador_sst_by_season.csv using the MplPath polygon method
to match pipeline's 18_pricing_document.py approach exactly.

Method (per region):
  1. Build MplPath fishing polygon from calas lat/lon (5th–95th pct per 1° band)
  2. For each year: filter SST anomaly to region lat bounds + inside polygon
  3. Resample daily → monthly means
  4. T1 = mean of months 4,5,6,7 (require all 4); T2 = mean of months 11,12 (require both)
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from matplotlib.path import Path as MplPath

FEATURES_DIR = Path("/home/jupyter-daniela/suyana/peru_production/features")
CALAS_CSV    = Path("/home/jupyter-daniela/suyana/peru_production/outputs/calas_all_data.csv")
OUT_CSV      = Path("/home/jupyter-daniela/cotizador_pesca_industrial/data/cotizador_sst_by_season.csv")

SST_YEARS = list(range(2002, 2026))  # 2002–2025
T1_MONTHS = [4, 5, 6, 7]
T2_MONTHS = [11, 12]
LON_W, LON_E = -82.0, -74.0
MIN_CALAS_PER_BAND = 20

REGIONS = {
    "Norte":        {"lat_min": -7.1,  "lat_max": None},
    "Centro Norte": {"lat_min": -11.0, "lat_max": -7.1},
    "Centro Sur":   {"lat_min": -15.8, "lat_max": -11.0},
    "Todas":        {"lat_min": -15.8, "lat_max": None},
}


def build_polygon(df, lat_min, lat_max):
    """Build MplPath fishing corridor polygon from calas data for a region."""
    sub = df[(df["lat"] >= lat_min)]
    if lat_max is not None:
        sub = sub[sub["lat"] <= lat_max]
    sub = sub.dropna(subset=["lat", "lon"])

    real_lat_min = sub["lat"].min()
    real_lat_max = sub["lat"].max()
    band_edges   = np.arange(int(np.floor(real_lat_min)), int(np.ceil(real_lat_max)), 1.0)

    west_lons, east_lons, valid_lats = [], [], []
    for lo in band_edges:
        band = sub[(sub["lat"] >= lo) & (sub["lat"] < lo + 1.0)]
        if len(band) < MIN_CALAS_PER_BAND:
            continue
        west_lons.append(np.percentile(band["lon"], 5))
        east_lons.append(np.percentile(band["lon"], 95))
        valid_lats.append(lo + 0.5)

    if not valid_lats:
        return None

    valid_lats = np.array(valid_lats)
    west_lons  = np.array(west_lons)
    east_lons  = np.array(east_lons)

    lat_lo = lat_min
    lat_hi = lat_max if lat_max is not None else real_lat_max

    lat_full  = np.concatenate([[lat_lo], valid_lats, [lat_hi]])
    west_full = np.concatenate([[west_lons[0]], west_lons, [west_lons[-1]]])
    east_full = np.concatenate([[east_lons[0]], east_lons, [east_lons[-1]]])
    poly_lons = np.concatenate([west_full, east_full[::-1], [west_full[0]]])
    poly_lats = np.concatenate([lat_full,  lat_full[::-1],  [lat_full[0]]])
    verts = list(zip(poly_lons, poly_lats))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def compute_region_sst(region_name, lat_min, lat_max, polygon, years):
    rows = []
    for yr in years:
        f = FEATURES_DIR / f"sst_anomaly_daily_{yr}.nc"
        if not f.exists():
            print(f"  {region_name} {yr}: file not found, skipping")
            continue

        ds = xr.open_dataset(f)
        lat_v = ds["lat"].values
        lon_v = ds["lon"].values

        lat_m = lat_v >= lat_min
        if lat_max is not None:
            lat_m &= (lat_v <= lat_max)
        lon_m = (lon_v >= LON_W) & (lon_v <= LON_E)

        lat_s = lat_v[lat_m]
        lon_s = lon_v[lon_m]

        if len(lat_s) == 0 or len(lon_s) == 0:
            ds.close()
            print(f"  {region_name} {yr}: empty lat/lon slice, skipping")
            continue

        G_lon, G_lat = np.meshgrid(lon_s, lat_s)
        pts     = np.column_stack([G_lon.ravel(), G_lat.ravel()])
        mask_2d = polygon.contains_points(pts).reshape(lat_s.size, lon_s.size)
        mask_da = xr.DataArray(mask_2d, dims=["lat", "lon"],
                               coords={"lat": lat_s, "lon": lon_s})

        s = (ds["sst_anomaly"]
             .isel(lat=np.where(lat_m)[0], lon=np.where(lon_m)[0])
             .where(mask_da)
             .mean(dim=["lat", "lon"])
             .to_series())
        s.index = pd.to_datetime(s.index)
        ds.close()

        monthly = s.dropna().resample("ME").mean()
        yr_mon  = monthly[monthly.index.year == yr]

        t1 = yr_mon[yr_mon.index.month.isin(T1_MONTHS)]
        t2 = yr_mon[yr_mon.index.month.isin(T2_MONTHS)]

        t1_ok = len(t1) == len(T1_MONTHS)
        t2_ok = len(t2) == len(T2_MONTHS)

        if t1_ok:
            rows.append({"year": yr, "tipo": "T1", "region": region_name,
                         "sst": round(float(t1.mean()), 3)})
        if t2_ok:
            rows.append({"year": yr, "tipo": "T2", "region": region_name,
                         "sst": round(float(t2.mean()), 3)})

        print(f"  {region_name} {yr}: "
              f"T1={'OK' if t1_ok else f'{len(t1)}/{len(T1_MONTHS)}'} "
              f"T2={'OK' if t2_ok else f'{len(t2)}/{len(T2_MONTHS)}'}")
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

print("Loading calas for polygon construction...")
calas = (pd.read_csv(CALAS_CSV, usecols=["latitud", "longitud"], low_memory=False)
         .rename(columns={"latitud": "lat", "longitud": "lon"})
         .apply(pd.to_numeric, errors="coerce")
         .dropna())
print(f"  {len(calas):,} cala points loaded")

all_rows = []
for region_name, bounds in REGIONS.items():
    lat_min = bounds["lat_min"]
    lat_max = bounds["lat_max"]
    print(f"\nBuilding polygon for {region_name} ...")
    polygon = build_polygon(calas, lat_min, lat_max)
    if polygon is None:
        print(f"  WARNING: no polygon for {region_name}, skipping")
        continue
    print(f"  Polygon built. Computing SST for years {SST_YEARS[0]}–{SST_YEARS[-1]} ...")
    rows = compute_region_sst(region_name, lat_min, lat_max, polygon, SST_YEARS)
    all_rows.extend(rows)
    print(f"  {len(rows)} SST records for {region_name}")

sst_df = pd.DataFrame(all_rows)
sst_df = sst_df[["year", "tipo", "region", "sst"]]
sst_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}  ({len(sst_df)} rows)")

# ── Verification: print Centro Norte percentiles to compare with report ──────
cn = sst_df[(sst_df["region"] == "Centro Norte") & sst_df["year"].between(2005, 2024)]
print("\nCentro Norte 2005–2024 pooled percentiles (T1+T2):")
for p in [80, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(cn['sst'].values, p):.3f}°C")

print("\nCentro Norte 2005–2024 T1 percentiles:")
cn_t1 = cn[cn["tipo"] == "T1"]
for p in [80, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(cn_t1['sst'].values, p):.3f}°C")

print("\nCentro Norte recent years:")
print(sst_df[sst_df["region"] == "Centro Norte"].tail(8).to_string(index=False))
print("\nDone.")
