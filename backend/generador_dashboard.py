"""
generador_dashboard.py — Pre-cálculo semanal del dashboard de dengue.

Orquesta la inferencia de TODOS los modelos sobre la AMT en la semana de
referencia (última semana epidemiológica con señal real) y exporta un único
archivo estático `predicciones_semana.json`, consolidado por UBIGEO, que el
frontend (React + Vite) consume sin cómputo en el cliente.

Uso:
    python generador_dashboard.py

Salida:
    backend/data/predicciones_semana.json
    public/data/predicciones_semana.json   (copia servida por Vite)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from inferencia import comun
from inferencia.modelos import (
    ZipM7b, StackingModel, BiLSTMModel, ProphetNac, cargar_arboles,
)

# ── Tabla de métricas reportada en la tesis (test set, H = 4 semanas) ────────
# map=True  -> produce predicción distrital seleccionable en el mapa
# scope     -> 'distrital' | 'nacional'
REGISTRO_MODELOS = [
    dict(id="M7b",     name="ZIP Horseshoe",          family="Bayesiano",
         rmse=4.89,  mae=0.62, r2=0.942, map=True,  scope="distrital", has_shap=True,
         description="Zero-Inflated Poisson con prior Horseshoe. Mejor desempeño global."),
    dict(id="M7a",     name="ZIP log-lineal",         family="Bayesiano",
         rmse=4.93,  mae=0.62, r2=0.941, map=False, scope="distrital", has_shap=False,
         description="ZIP con predictor log-lineal. Variante comparativa."),
    dict(id="M7c",     name="ZIP splines",            family="Bayesiano",
         rmse=5.09,  mae=0.63, r2=0.937, map=False, scope="distrital", has_shap=False,
         description="ZIP con splines truncados sobre covariables clave."),
    dict(id="STACK",   name="Stacking XGB+LGB+RF",    family="ML Ensamble",
         rmse=11.94, mae=0.90, r2=0.544, map=True,  scope="distrital", has_shap=True,
         description="Meta-modelo lineal positivo sobre XGBoost, LightGBM y RF."),
    dict(id="BILSTM",  name="BiLSTM Huber",           family="Deep Learning",
         rmse=12.03, mae=0.73, r2=0.372, map=True,  scope="distrital", has_shap=False,
         description="Red BiLSTM de 2 capas, ventana de 12 semanas, pérdida Huber."),
    dict(id="LGB_OPT", name="LightGBM Optuna",        family="Gradient Boosting",
         rmse=14.05, mae=0.95, r2=0.369, map=True,  scope="distrital", has_shap=True,
         description="LightGBM con hiperparámetros optimizados por Optuna."),
    dict(id="M7d",     name="ZIP XGB-offset",         family="Bayesiano",
         rmse=14.08, mae=0.91, r2=0.519, map=False, scope="distrital", has_shap=False,
         description="ZIP con offset log-λ de un XGBoost surrogate."),
    dict(id="LGB_GLB", name="LightGBM global",        family="Gradient Boosting",
         rmse=14.35, mae=0.96, r2=0.342, map=True,  scope="distrital", has_shap=True,
         description="LightGBM global sin optimización bayesiana."),
    dict(id="RF",      name="Random Forest",          family="Bagging",
         rmse=14.58, mae=0.98, r2=0.320, map=True,  scope="distrital", has_shap=True,
         description="500 árboles de decisión sobre el conjunto completo de features."),
    dict(id="XGB_LOG", name="XGBoost log1p",          family="Gradient Boosting",
         rmse=14.67, mae=0.98, r2=0.312, map=True,  scope="distrital", has_shap=True,
         description="XGBoost con target log1p y pérdida cuadrática."),
    dict(id="XGB_POI", name="XGBoost Poisson",        family="Gradient Boosting",
         rmse=14.69, mae=1.05, r2=0.310, map=True,  scope="distrital", has_shap=True,
         description="XGBoost con objetivo count:poisson."),
    dict(id="M7e",     name="ZIP NegBin",             family="Bayesiano",
         rmse=15.24, mae=1.01, r2=0.436, map=False, scope="distrital", has_shap=False,
         description="ZIP-NegativeBinomial con offset XGBoost (sobredispersión)."),
    dict(id="PROPHET", name="Prophet (mediana HH)",   family="Series de tiempo",
         rmse=6912.0, mae=3131.0, r2=-0.042, map=False, scope="nacional", has_shap=False,
         description="Prophet nacional con regresores satelitales. Opera a nivel país."),
]


def pivot_pred(panel, valores, target_weeks, ref, forward):
    """
    Reorganiza predicciones por (ubigeo × semana) y las alinea a las semanas
    objetivo `target_weeks` (periodo de test + cola de pronóstico).

    Para modelos de horizonte (forward), la predicción PARA la semana `t` se hizo
    con las features de `t-4`; los modelos contemporáneos (ZIP) usan `t`. Esto
    hace que la cola de pronóstico (SE+1..SE+4 tras la última semana observada)
    quede como predicho-sin-observado, y que el extremo de la serie coincida con
    el valor H4 del mapa.

    Devuelve (tabla_series[ubigeo×target_weeks], headline[ubigeo] en ref).
    """
    tmp = panel[["ubigeo", "fecha"]].copy()
    tmp["v"] = np.asarray(valores, dtype=float)
    tab = tmp.pivot(index="ubigeo", columns="fecha", values="v")
    headline = tab.get(ref)                        # pred hecha en ref (forecast a ref+4)
    delta = pd.Timedelta(weeks=4) if forward else pd.Timedelta(0)
    fuente = [pd.Timestamp(t) - delta for t in target_weeks]
    series = tab.reindex(columns=fuente)
    series.columns = list(target_weeks)            # renombrar a la semana objetivo
    return series, headline


def main():
    t0 = time.time()
    print("== Generador de dashboard de dengue ==")
    meta = comun.cargar_metadata()
    feats_arboles = meta["all_features"]          # 40 features para árboles/BiLSTM

    df = comun.cargar_amt()
    ref = comun.resolver_semana_ref(df)
    corte_train = pd.Timestamp(meta["corte_train"])
    corte_val = pd.Timestamp(meta["corte_val"])

    # Historia observada COMPLETA desde 2018; predicción out-of-sample solo en
    # el periodo de test (post corte_val) para coincidir con las métricas.
    hist_weeks = comun.semanas_rango(df, ref)
    pred_weeks = [w for w in hist_weeks if pd.Timestamp(w) > corte_val]
    pred_offset = len(hist_weeks) - len(pred_weeks)
    # Cola de pronóstico: 4 semanas posteriores a la última observada (solo predicho)
    future_weeks = [pd.Timestamp(ref) + pd.Timedelta(weeks=k) for k in (1, 2, 3, 4)]
    target_weeks = list(pred_weeks) + future_weeks      # eje de la serie predicha
    print(f"Semana de referencia : {pd.Timestamp(ref).date()} ({comun.etiqueta_se(ref)})")
    print(f"Distritos            : {df['ubigeo'].nunique()}")
    print(f"Historia observada   : {len(hist_weeks)} semanas ({pd.Timestamp(hist_weeks[0]).date()} → {pd.Timestamp(ref).date()})")
    print(f"Predicción           : {len(pred_weeks)} test + {len(future_weeks)} pronóstico (offset {pred_offset})")

    # Panel con buffer hacia atrás (cubre shift de 4 y ventana LSTM de 12)
    buffer_ini = pd.Timestamp(pred_weeks[0]) - pd.Timedelta(weeks=18)
    panel = df[(df["fecha"] >= buffer_ini) & (df["fecha"] <= ref)].copy()
    panel = panel.sort_values(["ubigeo", "fecha"])

    ref_rows = panel[panel["fecha"] == ref].drop_duplicates("ubigeo", keep="last")
    ref_rows = ref_rows.set_index("ubigeo")

    # ── Instanciar modelos ───────────────────────────────────────────────────
    print("\nCargando modelos...")
    zip_m = ZipM7b()
    arboles = cargar_arboles(feats_arboles)
    try:
        arboles["STACK"] = StackingModel(feats_arboles)
    except Exception as e:
        print(f"  ! STACK omitido: {e}")
    try:
        bilstm = BiLSTMModel(feats_arboles)
    except Exception as e:
        print(f"  ! BILSTM omitido: {e}"); bilstm = None
    try:
        prophet = ProphetNac()
    except Exception as e:
        print(f"  ! PROPHET omitido: {e}"); prophet = None
    print(f"  Árboles/ensamble cargados: {list(arboles)}")

    # ── Calcular series y headline por modelo ────────────────────────────────
    series_por_modelo = {}     # id -> DataFrame (ubigeo × hist_weeks)
    headline_por_modelo = {}   # id -> Series (ubigeo -> p en ref)

    # ZIP M7b (contemporáneo)
    print("\nInfiriendo ZIP M7b...")
    series_por_modelo["M7b"], headline_por_modelo["M7b"] = pivot_pred(
        panel, zip_m.pred_raw(panel), target_weeks, ref, forward=False)
    ey_ref, lo_ref, hi_ref = zip_m.pred_intervalo(ref_rows.reset_index())
    zip_ci = pd.DataFrame({"p": ey_ref, "lo": lo_ref, "hi": hi_ref},
                          index=ref_rows.index)

    # Árboles + stacking (horizonte)
    for mid, m in arboles.items():
        print(f"Infiriendo {mid}...")
        series_por_modelo[mid], headline_por_modelo[mid] = pivot_pred(
            panel, m.pred_raw(panel), target_weeks, ref, forward=True)

    # BiLSTM (horizonte, secuencial)
    if bilstm is not None:
        print("Infiriendo BiLSTM...")
        pred_lstm = bilstm.pred_panel(panel)
        series_por_modelo["BILSTM"], headline_por_modelo["BILSTM"] = pivot_pred(
            panel, pred_lstm.values, target_weeks, ref, forward=True)

    modelos_mapa = [m for m in series_por_modelo]   # ids con predicción distrital
    print(f"\nModelos distritales activos: {modelos_mapa}")

    # ── Historia observada COMPLETA desde 2018 (casos reales) ────────────────
    obs = df[df["fecha"].isin(hist_weeks)][["ubigeo", "fecha", "casos"]]
    hist_tab = (obs.pivot(index="ubigeo", columns="fecha", values="casos")
                .reindex(columns=hist_weeks))

    # Metadatos descriptivos por distrito (de la fila ref)
    info_cols = ["distrito", "departamento", "provincia", "ccdd",
                 "lisa_cluster", "poblacion"]
    info = ref_rows.reindex(columns=[c for c in info_cols if c in ref_rows.columns])

    lisa_map = {"HH": "HH", "LL": "LL", "LH": "LH", "HL": "HL"}

    # ── Ensamblar JSON por distrito ──────────────────────────────────────────
    print("Ensamblando JSON...")
    districts = {}
    ubigeos = list(hist_tab.index)
    for ub in ubigeos:
        row = info.loc[ub] if ub in info.index else None
        hist_vals = hist_tab.loc[ub].fillna(0).values
        pred_block = {}
        ref_row_df = ref_rows.loc[[ub]] if ub in ref_rows.index else None
        for mid in modelos_mapa:
            serie = series_por_modelo[mid].loc[ub] if ub in series_por_modelo[mid].index else None
            p = headline_por_modelo[mid].get(ub) if headline_por_modelo[mid] is not None else None
            if p is None or not np.isfinite(p):
                continue
            if mid == "M7b" and ub in zip_ci.index:
                lo, hi = zip_ci.loc[ub, "lo"], zip_ci.loc[ub, "hi"]
            else:
                lo_a, hi_a = comun.banda_poisson(np.array([p]))
                lo, hi = lo_a[0], hi_a[0]
            entry = {
                "p": comun.r(p),
                "lo": comun.r(lo),
                "hi": comun.r(hi),
                "series": [None if not np.isfinite(v) else comun.r(v)
                           for v in (serie.values if serie is not None else [])],
            }
            # Importancia de variables (solo donde el modelo la expone)
            if ref_row_df is not None:
                model_obj = zip_m if mid == "M7b" else arboles.get(mid)
                if model_obj is not None and hasattr(model_obj, "shap_local"):
                    sh = model_obj.shap_local(ref_row_df)
                    if sh:
                        entry["shap"] = sh
            pred_block[mid] = entry

        lisa = str(row["lisa_cluster"]) if row is not None and "lisa_cluster" in row else "ns"
        districts[ub] = {
            "name": str(row["distrito"]).title() if row is not None else ub,
            "dep": str(row["departamento"]).title() if row is not None else "",
            "prov": str(row["provincia"]).title() if row is not None else "",
            "ccdd": str(row["ccdd"]) if row is not None and "ccdd" in row else ub[:2],
            "lisa": lisa_map.get(lisa, "ns"),
            "pop": int(row["poblacion"]) if row is not None and pd.notna(row["poblacion"]) else 0,
            "hist": [comun.r(v, 0) for v in hist_vals],
            "pred": pred_block,
        }

    # ── Bloque nacional (serie agregada + Prophet) ───────────────────────────
    print("Agregando bloque nacional...")
    nac_hist = hist_tab.sum(axis=0).reindex(hist_weeks).fillna(0)
    nacional = {
        "hist": [comun.r(v, 0) for v in nac_hist.values],
        "pred": {},
    }
    for mid in modelos_mapa:
        s = series_por_modelo[mid].reindex(columns=target_weeks).sum(axis=0, min_count=1)
        p = float(headline_por_modelo[mid].sum())
        nacional["pred"][mid] = {
            "p": comun.r(p),
            "series": [None if not np.isfinite(v) else comun.r(v) for v in s.values],
        }
    if prophet is not None:
        try:
            fc = prophet.predecir(df, target_weeks)
            fc = fc.set_index("ds").reindex([pd.Timestamp(w) for w in target_weeks])
            nacional["pred"]["PROPHET"] = {
                "p": comun.r(fc["yhat"].iloc[-1]),
                "series": [comun.r(v) for v in fc["yhat"].values],
                "lo": [comun.r(v) for v in fc["yhat_lower"].clip(lower=0).values],
                "hi": [comun.r(v) for v in fc["yhat_upper"].clip(lower=0).values],
            }
            print("  Prophet nacional OK")
        except Exception as e:
            print(f"  ! Prophet nacional omitido: {e}")

    # ── Registro de modelos (marca disponibilidad real de predicción) ────────
    modelos_out = []
    for m in REGISTRO_MODELOS:
        disponible = (m["id"] in modelos_mapa) or (m["scope"] == "nacional" and prophet is not None)
        modelos_out.append({**m, "disponible": bool(disponible),
                            "map": bool(m["map"] and m["id"] in modelos_mapa)})

    salida = {
        "meta": {
            "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "semana_ref": pd.Timestamp(ref).strftime("%Y-%m-%d"),
            "se_label": comun.etiqueta_se(ref),
            "n_distritos": len(districts),
            "n_hist_semanas": len(hist_weeks),
            "pred_offset": pred_offset,
            "n_pred_semanas": len(pred_weeks),
            "n_future_semanas": len(future_weeks),
            "future_weeks": [comun.etiqueta_se(w) for w in future_weeks],
            "future_fechas": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in future_weeks],
            "hist_desde": pd.Timestamp(hist_weeks[0]).strftime("%Y-%m-%d"),
            "corte_train": corte_train.strftime("%Y-%m-%d"),
            "corte_val": corte_val.strftime("%Y-%m-%d"),
            "horizonte_real": "H4",
            "horizontes_no_entrenados": ["H1", "H8", "H12", "H24"],
            "modelo_default": "M7b",
            "umbrales_riesgo": comun.UMBRALES_RIESGO,
        },
        "models": modelos_out,
        "hist_weeks": [comun.etiqueta_se(w) for w in hist_weeks],
        "hist_fechas": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in hist_weeks],
        "districts": districts,
        "nacional": nacional,
    }

    # ── Escribir salida (backend + public) ───────────────────────────────────
    out_backend = comun.DATA_DIR / "predicciones_semana.json"
    out_public = comun.BACKEND_DIR.parent / "public" / "data" / "predicciones_semana.json"
    for path in (out_backend, out_public):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(salida, f, ensure_ascii=False, separators=(",", ":"))
    mb = out_backend.stat().st_size / 1e6
    print(f"\nOK  {len(districts)} distritos · {len(modelos_mapa)} modelos · {mb:.1f} MB")
    print(f"    {out_backend}")
    print(f"    {out_public}")
    print(f"    {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
