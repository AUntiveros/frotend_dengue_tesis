"""
Reconstruye series historicas H4 de los modelos top 5 guardados en try_julio_2026.

No reentrena modelos ni inventa forecast nuevo. Solo toma las predicciones ya
guardadas y las alinea al eje H4 como fecha_objetivo_h4 = fecha + 4 semanas.

Salida:
  frontend/backend/data/top5_try_julio_series.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRY = ROOT / "try_julio_2026"
SCRIPTS = TRY / "validacion_pipelines" / "scripts"
RESULTS = TRY / "validacion_pipelines" / "resultados"
PROC = TRY / "data" / "procesados"
OUT = ROOT / "frontend" / "backend" / "data" / "top5_try_julio_series.csv"

sys.path.insert(0, str(SCRIPTS))


def ubigeo6(s: object) -> str:
    return str(s).strip().split(".")[0].zfill(6)


def inla_with_dates(filename: str, model_id: str, pred_col: str) -> pd.DataFrame:
    import t1_data as D

    df, _ = D.load_amt()
    df = df.assign(ubigeo=df["ubigeo"].map(ubigeo6))

    nodes = pd.read_csv(PROC / "spatial" / "nodes.csv", dtype={"ubigeo": str})["ubigeo"].map(ubigeo6).tolist()
    sidx = {u: k + 1 for k, u in enumerate(nodes)}

    dd = df.dropna(subset=["target_h4", "casos_lag1", "casos_lag4"]).copy()
    dd = dd[dd["ubigeo"].isin(sidx)]
    times = sorted(pd.to_datetime(dd["fecha"]).unique())
    fecha_of = {k + 1: pd.Timestamp(t) for k, t in enumerate(times)}

    mod = pd.read_csv(PROC / "inla_input" / "modeling.csv", dtype={"ubigeo": str})
    modt = mod[mod["split"] == "test"].reset_index(drop=True)
    modt["ubigeo"] = modt["ubigeo"].map(ubigeo6)

    pred = pd.read_csv(RESULTS / filename, dtype={"ubigeo": str}).reset_index(drop=True)
    pred["ubigeo"] = pred["ubigeo"].map(ubigeo6)
    if len(modt) != len(pred):
        raise RuntimeError(f"{filename}: filas modeling test={len(modt)} != pred={len(pred)}")
    match = float((modt["ubigeo"].values == pred["ubigeo"].values).mean())
    if match < 0.999:
        raise RuntimeError(f"{filename}: alineacion ubigeo insuficiente ({match:.4f})")

    out = pd.DataFrame(
        {
            "model_id": model_id,
            "ubigeo": pred["ubigeo"].values,
            "fecha": pd.to_datetime(modt["time_idx"].map(fecha_of).values),
            "actual": pred["actual"].astype(float).values,
            "pred": pred[pred_col].astype(float).values,
        }
    )
    out["fecha_objetivo_h4"] = out["fecha"] + pd.Timedelta(weeks=4)
    return out[["model_id", "ubigeo", "fecha", "fecha_objetivo_h4", "actual", "pred"]]


def zip_series() -> pd.DataFrame:
    import json
    import xarray as xr

    print("  ZIP: leyendo AMT...", flush=True)
    df = pd.read_parquet(TRY / "data" / "AMT_final.parquet")
    with open(TRY / "data" / "AMT_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    df["fecha"] = pd.to_datetime(df["fecha"])
    ct, cv = pd.Timestamp(meta["corte_train"]), pd.Timestamp(meta["corte_val"])
    feat_meta = [f for f in meta["all_features"] if f != "flag_anomalia"]

    print("  ZIP: preparando muestra...", flush=True)
    base = df[df["fecha"] <= ct].copy()
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

    print(f"  ZIP: fit scaler samp={len(samp)} feats={len(feat_ok)}", flush=True)
    scaler = StandardScaler().fit(samp[feat_ok].fillna(0))
    test_full = df[df["fecha"] > cv].dropna(subset=feat_ok + ["target_h4"]).copy()
    print(f"  ZIP: test={len(test_full)}", flush=True)
    Xte = scaler.transform(test_full[feat_ok].fillna(0)).astype(np.float32)
    clte = test_full["cluster_code"].values.astype(int)
    yte = test_full["target_h4"].clip(0).values

    test = test_full[["ubigeo", "fecha"]].copy()
    test["ubigeo"] = test["ubigeo"].map(ubigeo6)
    test["fecha"] = pd.to_datetime(test["fecha"])
    test["fecha_objetivo_h4"] = test["fecha"] + pd.Timedelta(weeks=4)
    test["actual"] = np.asarray(yte, dtype=float)

    def point_pred_nc(model_dir: str, X: np.ndarray, cluster: np.ndarray) -> np.ndarray:
        print(f"  ZIP: leyendo {model_dir}...", flush=True)
        ds = xr.open_dataset(PROC / model_dir / "trace.nc", group="posterior")
        beta = ds["beta"].mean(("chain", "draw")).values
        inter = float(ds["intercept"].mean(("chain", "draw")).values)
        phh = float(ds["psi_hh"].mean(("chain", "draw")).values)
        pll = float(ds["psi_ll"].mean(("chain", "draw")).values)
        pns = float(ds["psi_ns"].mean(("chain", "draw")).values)
        ds.close()
        print(f"  ZIP: prediciendo {model_dir} X={X.shape}", flush=True)
        log_lam = inter + np.einsum("ij,j->i", X, beta, optimize=False)
        lam = np.exp(np.clip(log_lam, -10, 12))
        psi = np.where(cluster == 4, phh, np.where(cluster == 1, pll, pns))
        return np.clip((1 - psi) * lam, 0, None)

    print("  ZIP: M7a...", flush=True)
    pred_a = point_pred_nc("m7a_zip_lineal", Xte, clte)

    # Replica la base spline de t2_figs_bayes.py para M7c.
    spline_vars = [
        v
        for v in ["tmean_clima_lag_depto", "ptot_clima_lag_depto", "humr_clima_lag_depto"]
        if v in samp.columns
    ]
    cols = []
    for v in spline_vars:
        ref = samp[v].dropna().values
        mu, sd = ref.mean(), ref.std() + 1e-8
        vals = (test_full[v].fillna(ref.mean()).values - mu) / sd
        knots = (np.quantile(ref, [0.25, 0.5, 0.75]) - mu) / sd
        for k in knots:
            cols.append(np.maximum(0, vals - k))
    basis_te = np.column_stack(cols).astype(np.float32) if cols else np.empty((len(test), 0), np.float32)
    Xte_c = np.hstack([Xte, basis_te])
    print("  ZIP: M7c...", flush=True)
    pred_c = point_pred_nc("m7c_zip_splines", Xte_c, clte)

    a = test.copy()
    a["model_id"] = "ZIP_M7A"
    a["pred"] = pred_a
    c = test.copy()
    c["model_id"] = "ZIP_M7C"
    c["pred"] = pred_c
    return pd.concat([a, c], ignore_index=True)[
        ["model_id", "ubigeo", "fecha", "fecha_objetivo_h4", "actual", "pred"]
    ]


def main() -> None:
    print("== Reconstruyendo top 5 desde try_julio_2026 ==")
    parts = [
        inla_with_dates("inla_pred_Multinivel_poisson.csv", "INLA_MULTI", "pred"),
        inla_with_dates("inla_pred_M4_bym2_lags_poisson.csv", "INLA_BYM2", "pred"),
        zip_series(),
    ]
    out = pd.concat(parts, ignore_index=True)
    out["pred"] = out["pred"].clip(lower=0)
    out["actual"] = out["actual"].clip(lower=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby("model_id").agg(n=("pred", "size"), desde=("fecha_objetivo_h4", "min"), hasta=("fecha_objetivo_h4", "max")))
    print(f"OK: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
    print("  ZIP: basis M7c...", flush=True)
