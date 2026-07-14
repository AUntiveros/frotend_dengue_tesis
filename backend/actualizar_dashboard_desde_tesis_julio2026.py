"""
Actualiza el JSON estatico del dashboard con los resultados reales de
tesis_presentacion_julio2026.

Entrada principal:
  - tesis_presentacion_julio2026/data_processed/AMT_final.parquet
  - tesis_presentacion_julio2026/predicciones/*.csv

Salida:
  - frontend/backend/data/predicciones_semana.json
  - frontend/public/data/predicciones_semana.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TESIS = ROOT / "tesis_presentacion_julio2026"
FRONTEND = ROOT / "frontend"
PRED = TESIS / "predicciones"
AMT_PATH = TESIS / "data_processed" / "AMT_final.parquet"

AVAILABLE_MODELS = {
    "ENS_H4": {
        "column": "pred_calibrated_h4",
        "description": "Mejor modelo oficial: ensemble enrutado por estrato, con inferencia operativa cargada para SE26/2026.",
        "map": True,
        "has_shap": False,
    },
    "XGB_POI_RT": {
        "column": "pred_xgb_poisson_retrain",
        "name": "XGBoost Poisson reentrenado",
        "family": "Gradient Boosting",
        "description": "Modelo tabular reentrenado para conteos con objetivo Poisson.",
        "map": True,
        "has_shap": False,
    },
    "STACK_RT": {
        "column": "pred_meta_retrain",
        "name": "Stacking tabular reentrenado",
        "family": "ML Ensamble",
        "description": "Stacking LightGBM + XGB log1p + XGB Poisson reentrenado.",
        "map": True,
        "has_shap": False,
    },
    "LGBM_RT": {
        "column": "pred_lgbm_retrain",
        "name": "LightGBM Optuna corregido reentrenado",
        "family": "Gradient Boosting",
        "description": "LightGBM corregido y reentrenado con la AMT actualizada.",
        "map": True,
        "has_shap": False,
    },
    "XGB_LOG_RT": {
        "column": "pred_xgb_log_retrain",
        "name": "XGBoost log1p reentrenado",
        "family": "Gradient Boosting",
        "description": "XGBoost reentrenado sobre target log1p.",
        "map": True,
        "has_shap": False,
    },
}

LEGACY_TOP5_SERIES_IDS = {"INLA_MULTI", "INLA_BYM2", "ZIP_M7C", "ZIP_M7A"}

OFFICIAL_TO_DASHBOARD_ID = {
    "Ensemble enrutado por estrato": "ENS_H4",
    "INLA Multinivel_poisson": "INLA_MULTI",
    "INLA M4_bym2_lags_poisson": "INLA_BYM2",
    "M7c ZIP splines": "ZIP_M7C",
    "M7a ZIP lineal": "ZIP_M7A",
    "Stacking XGB+LGB+RF": "STACK_RT",
    "LightGBM Optuna (fix)": "LGBM_RT",
    "XGBoost Poisson": "XGB_POI_RT",
    "XGBoost log1p": "XGB_LOG_RT",
}

DISPLAY_NAMES = {
    "INLA Multinivel_poisson": "INLA Multinivel (Poisson)",
    "INLA M4_bym2_lags_poisson": "INLA-BYM2 + lags (Poisson)",
    "M7a ZIP lineal": "ZIP M7a log-lineal",
    "M7c ZIP splines": "ZIP M7c splines",
    "M7b ZIP Horseshoe": "ZIP M7b Horseshoe",
    "Sebast-Ensemble(RF stack)": "Sebastianelli replicado (ensemble RF)",
    "Hurdle-NB": "Hurdle Negative Binomial",
    "STGNN resid-gate": "ST-GNN resid-gate",
    "STGNN resid-gate v2 cluster": "ST-GNN resid-gate v2 (cluster-aware)",
    "STGNN gated": "ST-GNN gated",
    "LightGBM Optuna (fix)": "LightGBM Optuna (corregido)",
    "GNN resid-gate (LGBM+g)": "ST-GNN resid-gate",
    "BMA pseudo tabular": "BMA (pseudo) tabular",
    "LGBM + vecinos v2": "LightGBM + vecinos v2",
    "LGBM Tweedie p=1.3": "LightGBM Tweedie (p = 1,3)",
    "LGBM default": "LightGBM default",
    "XGB Tweedie p=1.2": "XGBoost Tweedie (p = 1,2)",
    "LGBM + vecinos espaciales": "LightGBM + vecinos espaciales (v1)",
    "Random Forest": "Random Forest",
    "XGBoost Poisson": "XGBoost Poisson",
    "XGB Tweedie p=1.5": "XGBoost Tweedie (p = 1,5)",
    "NGBoost": "NGBoost",
    "XGBoost log1p": "XGBoost log1p",
    "CatBoost log1p": "CatBoost log1p",
    "Sebast-CatBoost": "Sebastianelli replicado (CatBoost)",
    "CatBoost Poisson": "CatBoost Poisson",
    "BART Poisson": "BART Poisson",
    "M7d ZIP XGB-offset": "ZIP M7d XGB-offset",
    "M7e ZIP NegBin": "ZIP M7e NegBin",
    "HMM Poisson 3 estados": "HMM Poisson (3 estados)",
    "Hawkes auto-excitante": "Hawkes auto-excitante",
    "Sebast-SVR Nystroem": "Sebastianelli replicado (SVR-Nystroem)",
    "Prophet rolling nacional": "Prophet (rolling-origin, nacional)",
    "GAMM NegBin": "GAMM (replicado, NegBin)",
}

DISPLAY_FAMILIES = {
    "Bayes/conteos": "Bayes espacial",
    "Deep learning": "Deep Learning",
    "Ensamble": "Ensamble",
}


def ubigeo6(s: object) -> str:
    return str(s).strip().split(".")[0].zfill(6)


def r(value: object, ndigits: int = 3):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def label_se(fecha: pd.Timestamp, se_by_date: dict[pd.Timestamp, int], ref_date: pd.Timestamp, ref_se: int) -> str:
    fecha = pd.Timestamp(fecha)
    if fecha in se_by_date:
        se = int(se_by_date[fecha])
    else:
        se = ref_se + int(round((fecha - ref_date).days / 7))
    return f"SE{se:02d}/{fecha.year}"


def poisson_interval(p: float) -> tuple[float, float]:
    p = max(float(p), 0.0)
    half = 1.96 * math.sqrt(max(p, 1e-9))
    return max(0.0, p - half), p + half


def build_model_registry(map_capable_ids: set[str]) -> list[dict]:
    modelos: list[dict] = []

    ranking_path = PRED / "comparacion_41_modelos_multinivel_metricas.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        for i, row in ranking.iterrows():
            official_name = str(row["modelo"])
            mid = OFFICIAL_TO_DASHBOARD_ID.get(official_name, f"BM_{i + 1:02d}")
            available_cfg = AVAILABLE_MODELS.get(mid)
            is_available = available_cfg is not None or mid in LEGACY_TOP5_SERIES_IDS
            family = str(row.get("familia", "Comparativa"))
            desc = "Benchmark oficial H4 del paquete comparativo de 41 modelos; sin serie distrital cargada en el mapa."
            if available_cfg is not None:
                desc = available_cfg["description"]
            elif mid in LEGACY_TOP5_SERIES_IDS:
                desc = "Serie historica H4 reconstruida desde try_julio_2026; usar como validacion/modelado, no como reentrenamiento nuevo SE26."
            modelos.append(
                {
                    "id": mid,
                    "name": DISPLAY_NAMES.get(official_name, official_name.replace("_", " ")),
                    "family": DISPLAY_FAMILIES.get(family, family),
                    "description": desc,
                    "n": int(row.get("n", 0)) if pd.notna(row.get("n", np.nan)) else None,
                    "rmse": r(row.get("rmse")),
                    "mae": r(row.get("mae")),
                    "r2": r(row.get("r2")),
                    "map": bool(mid in map_capable_ids),
                    "scope": "distrital",
                    "has_shap": bool(available_cfg is not None and available_cfg["has_shap"]),
                    "disponible": bool(is_available),
                }
            )
    return modelos


def main() -> None:
    print("== Actualizando dashboard desde tesis_presentacion_julio2026 ==")

    amt_cols = [
        "ubigeo",
        "fecha",
        "departamento",
        "provincia",
        "distrito",
        "semana_epi",
        "casos",
        "poblacion",
        "lisa_cluster",
        "endemico",
    ]
    amt = pd.read_parquet(AMT_PATH, columns=amt_cols)
    amt["fecha"] = pd.to_datetime(amt["fecha"])
    amt["ubigeo"] = amt["ubigeo"].map(ubigeo6)
    amt = amt.sort_values(["ubigeo", "fecha"])

    ref_date = amt["fecha"].max()
    ref_rows = amt[amt["fecha"] == ref_date].drop_duplicates("ubigeo", keep="last").set_index("ubigeo")
    hist_dates = sorted(amt.loc[amt["fecha"] <= ref_date, "fecha"].unique())
    hist_dates = [pd.Timestamp(d) for d in hist_dates]
    future_dates = [ref_date + pd.Timedelta(weeks=k) for k in range(1, 5)]
    all_dates = hist_dates + future_dates
    se_by_date = (
        amt[["fecha", "semana_epi"]]
        .dropna()
        .drop_duplicates("fecha")
        .set_index("fecha")["semana_epi"]
        .astype(int)
        .to_dict()
    )
    ref_se = int(se_by_date[ref_date])

    series = pd.read_csv(PRED / "series_operativas_recalibradas_h4_distrital.csv", parse_dates=["fecha_objetivo_h4"])
    series["ubigeo"] = series["ubigeo"].map(ubigeo6)
    series = series.sort_values(["ubigeo", "fecha_objetivo_h4"])
    target_dates = [d for d in all_dates if d >= series["fecha_objetivo_h4"].min()]
    pred_offset = all_dates.index(target_dates[0])
    last_future = future_dates[-1]

    forecast = pd.read_csv(PRED / "forecast_h4_continuo_calibrado_distrital.csv", parse_dates=["fecha_objetivo_h4"])
    forecast["ubigeo"] = forecast["ubigeo"].map(ubigeo6)
    fc_last = forecast[forecast["fecha_objetivo_h4"] == last_future].set_index("ubigeo")

    obs = amt.pivot_table(index="ubigeo", columns="fecha", values="casos", aggfunc="sum").reindex(columns=hist_dates)
    obs = obs.sort_index()
    ubigeos = list(obs.index)

    pred_tabs: dict[str, pd.DataFrame] = {}
    lo_tabs: dict[str, pd.DataFrame] = {}
    hi_tabs: dict[str, pd.DataFrame] = {}
    for mid, cfg in AVAILABLE_MODELS.items():
        tab = (
            series.pivot_table(index="ubigeo", columns="fecha_objetivo_h4", values=cfg["column"], aggfunc="last")
            .reindex(index=ubigeos, columns=target_dates)
        )
        pred_tabs[mid] = tab

    legacy_path = FRONTEND / "backend" / "data" / "top5_try_julio_series.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path, parse_dates=["fecha_objetivo_h4"], dtype={"ubigeo": str})
        legacy["ubigeo"] = legacy["ubigeo"].map(ubigeo6)
        legacy = legacy[legacy["fecha_objetivo_h4"] <= ref_date].copy()
        for mid, g in legacy.groupby("model_id"):
            tab = (
                g.pivot_table(index="ubigeo", columns="fecha_objetivo_h4", values="pred", aggfunc="last")
                .reindex(index=ubigeos, columns=target_dates)
            )
            pred_tabs[mid] = tab

    current_top5_path = FRONTEND / "backend" / "data" / "top5_forecast_vigente.csv"
    if current_top5_path.exists():
        current_top5 = pd.read_csv(current_top5_path, parse_dates=["target_fecha"], dtype={"ubigeo": str})
        current_top5["ubigeo"] = current_top5["ubigeo"].map(ubigeo6)
        for mid, g in current_top5.groupby("model_id"):
            tab = (
                g.pivot_table(index="ubigeo", columns="target_fecha", values="pred", aggfunc="last")
                .reindex(index=ubigeos, columns=target_dates)
            )
            if mid in pred_tabs:
                pred_tabs[mid] = pred_tabs[mid].combine_first(tab)
                pred_tabs[mid].update(tab)
            else:
                pred_tabs[mid] = tab
            if "q025" in g.columns:
                lo_tabs[mid] = (
                    g.pivot_table(index="ubigeo", columns="target_fecha", values="q025", aggfunc="last")
                    .reindex(index=ubigeos, columns=target_dates)
                )
            if "q975" in g.columns:
                hi_tabs[mid] = (
                    g.pivot_table(index="ubigeo", columns="target_fecha", values="q975", aggfunc="last")
                    .reindex(index=ubigeos, columns=target_dates)
                )

    map_capable_ids = {
        mid
        for mid, tab in pred_tabs.items()
        if last_future in tab.columns and tab[last_future].notna().any()
    }

    info_cols = ["departamento", "provincia", "distrito", "semana_epi", "poblacion", "lisa_cluster", "endemico"]
    info = ref_rows.reindex(columns=info_cols)
    lisa_ok = {"HH", "LL", "LH", "HL"}
    districts = {}

    print(f"Referencia: {ref_date.date()} ({label_se(ref_date, se_by_date, ref_date, ref_se)})")
    print(f"Distritos : {len(ubigeos)}")
    print(f"Predicho  : {target_dates[0].date()} -> {target_dates[-1].date()} ({len(target_dates)} semanas)")

    for ub in ubigeos:
        row = info.loc[ub] if ub in info.index else pd.Series(dtype=object)
        hist_vals = obs.loc[ub].fillna(0).values
        pred_block = {}
        for mid, tab in pred_tabs.items():
            if ub not in tab.index:
                continue
            vals = tab.loc[ub]
            p = vals.get(last_future)
            has_any_series = vals.notna().any()
            has_current = p is not None and np.isfinite(p)
            if not has_any_series:
                continue
            if mid in lo_tabs and mid in hi_tabs and has_current:
                lo = lo_tabs[mid].loc[ub].get(last_future) if ub in lo_tabs[mid].index else None
                hi = hi_tabs[mid].loc[ub].get(last_future) if ub in hi_tabs[mid].index else None
                if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = poisson_interval(float(p))
            elif mid == "ENS_H4" and has_current and ub in fc_last.index:
                lo = fc_last.loc[ub, "ic95_inf"]
                hi = fc_last.loc[ub, "ic95_sup"]
            elif has_current:
                lo, hi = poisson_interval(float(p))
            else:
                lo, hi = None, None
            pred_block[mid] = {
                "p": r(p) if has_current else None,
                "lo": r(lo),
                "hi": r(hi),
                "series": [r(v) for v in vals.values],
            }

        lisa = str(row.get("lisa_cluster", "ns"))
        districts[ub] = {
            "name": str(row.get("distrito", ub)).title(),
            "dep": str(row.get("departamento", "")).title(),
            "prov": str(row.get("provincia", "")).title(),
            "ccdd": ub[:2],
            "lisa": lisa if lisa in lisa_ok else "ns",
            "pop": int(row.get("poblacion", 0) or 0),
            "hist": [r(v, 0) for v in hist_vals],
            "pred": pred_block,
        }

    nacional = {"hist": [r(v, 0) for v in obs.sum(axis=0).values], "pred": {}}
    for mid, tab in pred_tabs.items():
        s = tab.sum(axis=0, min_count=1)
        p = s.get(last_future)
        has_current = p is not None and np.isfinite(p)
        if not s.notna().any():
            continue
        lo_series = []
        hi_series = []
        for v in s.values:
            if v is None or not np.isfinite(v):
                lo_series.append(None)
                hi_series.append(None)
            else:
                lo, hi = poisson_interval(float(v))
                lo_series.append(r(lo))
                hi_series.append(r(hi))
        nacional["pred"][mid] = {
            "p": r(p) if has_current else None,
            "series": [r(v) for v in s.values],
            "lo": lo_series,
            "hi": hi_series,
        }

    salida = {
        "meta": {
            "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "semana_ref": ref_date.strftime("%Y-%m-%d"),
            "se_label": label_se(ref_date, se_by_date, ref_date, ref_se),
            "n_distritos": len(districts),
            "n_hist_semanas": len(hist_dates),
            "pred_offset": pred_offset,
            "n_pred_semanas": len([d for d in target_dates if d <= ref_date]),
            "n_future_semanas": len(future_dates),
            "future_weeks": [label_se(d, se_by_date, ref_date, ref_se) for d in future_dates],
            "future_fechas": [d.strftime("%Y-%m-%d") for d in future_dates],
            "hist_desde": hist_dates[0].strftime("%Y-%m-%d"),
            "corte_train": "2022-12-31",
            "corte_val": "2023-12-31",
            "horizonte_real": "H4",
            "horizontes_no_entrenados": ["H1", "H8", "H12", "H16", "H24"],
            "modelo_default": "ENS_H4",
            "umbrales_riesgo": {"sin": 0, "bajo": 5, "moderado": 20, "alto": 100},
        },
        "models": build_model_registry(map_capable_ids),
        "hist_weeks": [label_se(d, se_by_date, ref_date, ref_se) for d in hist_dates],
        "hist_fechas": [d.strftime("%Y-%m-%d") for d in hist_dates],
        "districts": districts,
        "nacional": nacional,
    }

    for path in [
        FRONTEND / "backend" / "data" / "predicciones_semana.json",
        FRONTEND / "public" / "data" / "predicciones_semana.json",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(salida, f, ensure_ascii=False, separators=(",", ":"))
        print(f"OK: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
