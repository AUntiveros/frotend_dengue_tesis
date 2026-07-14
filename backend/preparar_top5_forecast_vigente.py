"""
Genera pronosticos vigentes SE27-SE30 para modelos top 5 cuando hay artefactos
reales disponibles.

Incluye:
  - INLA-BYM2 + lags actualizado desde tesis_presentacion_julio2026.
  - ZIP M7a/M7c entrenados en try_julio_2026, aplicados a la AMT actualizada.

No inventa INLA Multinivel si la corrida actualizada no esta disponible.

Salida:
  frontend/backend/data/top5_forecast_vigente.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRY = ROOT / "try_julio_2026"
TESIS = ROOT / "tesis_presentacion_julio2026"
FRONTEND = ROOT / "frontend"
PROC = TRY / "data" / "procesados"
OUT = FRONTEND / "backend" / "data" / "top5_forecast_vigente.csv"


def ubigeo6(s: object) -> str:
    return str(s).strip().split(".")[0].zfill(6)


def zip_forecast() -> pd.DataFrame:
    import xarray as xr

    print("ZIP vigente: preparando scaler desde try_julio...", flush=True)
    train_df = pd.read_parquet(TRY / "data" / "AMT_final.parquet")
    with open(TRY / "data" / "AMT_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    train_df["fecha"] = pd.to_datetime(train_df["fecha"])
    ct = pd.Timestamp(meta["corte_train"])
    feat_meta = [f for f in meta["all_features"] if f != "flag_anomalia"]

    base = train_df[train_df["fecha"] <= ct].copy()
    base["obs_weight"] = 1.0
    base.loc[(base["fecha"] >= "2020-01-01") & (base["fecha"] <= "2021-12-31"), "obs_weight"] = 0.01
    base.loc[base["flag_anomalia"] == 1, "obs_weight"] = np.minimum(
        base.loc[base["flag_anomalia"] == 1, "obs_weight"], 0.05
    )

    rng = np.random.default_rng(42)
    hh = base[base["cluster_code"] == 4]
    ns_ub = base[base["cluster_code"] == 0]["ubigeo"].unique()
    ns = base[base["ubigeo"].isin(rng.choice(ns_ub, min(200, len(ns_ub)), replace=False))]
    ll_ub = base[base["cluster_code"] == 1]["ubigeo"].unique()
    ll = base[base["ubigeo"].isin(rng.choice(ll_ub, min(100, len(ll_ub)), replace=False))]
    samp = pd.concat([hh, ns, ll], ignore_index=True)

    feat_ok = [f for f in feat_meta if f in samp.columns]
    samp = samp.dropna(subset=feat_ok + ["target_h4"]).copy()
    if len(samp) > 15000:
        pos = samp[samp["target_h4"] > 0]
        neg = samp[samp["target_h4"] == 0].sample(
            min(15000 - len(pos), (samp["target_h4"] == 0).sum()), random_state=42
        )
        samp = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    scaler = StandardScaler().fit(samp[feat_ok].fillna(0))

    current = pd.read_parquet(TESIS / "data_processed" / "AMT_final.parquet")
    current["fecha"] = pd.to_datetime(current["fecha"])
    current["ubigeo"] = current["ubigeo"].map(ubigeo6)
    ref = current["fecha"].max()
    base_dates = [ref - pd.Timedelta(weeks=k) for k in [3, 2, 1, 0]]
    current = current[current["fecha"].isin(base_dates)].dropna(subset=feat_ok).copy()
    current["target_fecha"] = current["fecha"] + pd.Timedelta(weeks=4)
    X = scaler.transform(current[feat_ok].fillna(0)).astype(np.float32)
    cluster = current["cluster_code"].values.astype(int)

    def point_pred_nc(model_dir: str, Xmat: np.ndarray, cluster_arr: np.ndarray) -> np.ndarray:
        ds = xr.open_dataset(PROC / model_dir / "trace.nc", group="posterior")
        beta = ds["beta"].mean(("chain", "draw")).values
        inter = float(ds["intercept"].mean(("chain", "draw")).values)
        phh = float(ds["psi_hh"].mean(("chain", "draw")).values)
        pll = float(ds["psi_ll"].mean(("chain", "draw")).values)
        pns = float(ds["psi_ns"].mean(("chain", "draw")).values)
        ds.close()
        log_lam = inter + np.einsum("ij,j->i", Xmat, beta, optimize=False)
        lam = np.exp(np.clip(log_lam, -10, 12))
        psi = np.where(cluster_arr == 4, phh, np.where(cluster_arr == 1, pll, pns))
        return np.clip((1 - psi) * lam, 0, None)

    print("ZIP vigente: M7a...", flush=True)
    pred_a = point_pred_nc("m7a_zip_lineal", X, cluster)

    print("ZIP vigente: M7c...", flush=True)
    spline_vars = [
        v
        for v in ["tmean_clima_lag_depto", "ptot_clima_lag_depto", "humr_clima_lag_depto"]
        if v in samp.columns
    ]
    cols = []
    for v in spline_vars:
        ref_vals = samp[v].dropna().values
        mu, sd = ref_vals.mean(), ref_vals.std() + 1e-8
        vals = (current[v].fillna(ref_vals.mean()).values - mu) / sd
        knots = (np.quantile(ref_vals, [0.25, 0.5, 0.75]) - mu) / sd
        for k in knots:
            cols.append(np.maximum(0, vals - k))
    basis = np.column_stack(cols).astype(np.float32) if cols else np.empty((len(current), 0), np.float32)
    pred_c = point_pred_nc("m7c_zip_splines", np.hstack([X, basis]), cluster)

    base_cols = ["ubigeo", "departamento", "provincia", "distrito", "fecha", "target_fecha"]
    a = current[base_cols].copy()
    a["model_id"] = "ZIP_M7A"
    a["pred"] = pred_a
    c = current[base_cols].copy()
    c["model_id"] = "ZIP_M7C"
    c["pred"] = pred_c
    return pd.concat([a, c], ignore_index=True)


def inla_bym2_forecast() -> pd.DataFrame:
    p = TESIS / "predicciones" / "inla_actualizado" / "forecast_inla_h4_distrital.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["target_fecha"], dtype={"ubigeo": str})
    df = df[(df["modelo"] == "M4_bym2_lags") & (df["familia"] == "poisson")].copy()
    df["ubigeo"] = df["ubigeo"].map(ubigeo6)
    df["fecha"] = df["target_fecha"] - pd.Timedelta(weeks=4)
    df["model_id"] = "INLA_BYM2"
    df = df.rename(columns={"pred": "pred"})
    return df[["model_id", "ubigeo", "departamento", "provincia", "distrito", "fecha", "target_fecha", "pred", "q025", "q975"]]


def main() -> None:
    print("== Preparando forecast vigente top5 ==", flush=True)
    parts = [inla_bym2_forecast(), zip_forecast()]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out["pred"] = out["pred"].clip(lower=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby("model_id").agg(n=("pred", "size"), desde=("target_fecha", "min"), hasta=("target_fecha", "max"), total=("pred", "sum")))
    print(f"OK: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
