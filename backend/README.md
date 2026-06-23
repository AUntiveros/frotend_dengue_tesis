# Backend — Pre-cálculo del dashboard de dengue

Convierte los modelos entrenados (notebooks `03*`) en un único archivo
estático que el frontend (React + Vite) consume sin cómputo en el cliente.

## Uso

```bash
python -m pip install -r requirements.txt
python generador_dashboard.py
```

Genera `data/predicciones_semana.json` y lo copia a `../public/data/` (servido por Vite).
Re-ejecutar cada semana epidemiológica al actualizar la AMT.

## Qué hace

1. Carga `data/AMT_final.parquet` + `data/AMT_metadata.json`.
2. Resuelve la **semana de referencia** = última semana con casos reales
   (las posteriores en la AMT son relleno para sostener los lags).
3. Corre la inferencia de cada modelo y consolida por UBIGEO.
4. Exporta predicción puntual H4 + IC95% + serie observado/predicho (52 sem)
   + importancia de variables, por modelo y distrito.

## Modelos

| id        | Modelo                | Escala     | Artefacto                          |
|-----------|-----------------------|------------|------------------------------------|
| `M7b`     | ZIP Horseshoe (mejor) | distrital  | `zip_beta_mean/intercept/psi` + `scaler_zip` |
| `XGB_LOG` | XGBoost log1p         | distrital  | `xgb_log.json`                     |
| `XGB_POI` | XGBoost Poisson       | distrital  | `xgb_poisson.json`                 |
| `LGB_OPT` | LightGBM Optuna       | distrital  | `lgb_optuna.txt`                   |
| `RF`      | Random Forest         | distrital  | `rf_model.pkl`                     |
| `STACK`   | Stacking XGB+LGB+RF   | distrital  | `stack_coef.npy` + bases           |
| `BILSTM`  | BiLSTM Huber          | distrital  | `bilstm_best.pt` + `scaler_lstm`   |
| `PROPHET` | Prophet multi-satelital | nacional | `prophet_full.pkl`                 |

`M7a/M7c/M7d/M7e` y `LGB global` aparecen en la comparativa de métricas pero
no producen predicción distrital (sin arrays de inferencia guardados o con
subconjunto de features incompatible). Solo **H4** tiene modelo entrenado.

## Estructura

```
inferencia/
  comun.py     carga AMT, semana de referencia, helpers numéricos
  modelos.py   cargadores e inferencia de cada familia
generador_dashboard.py   orquestador → predicciones_semana.json
```

## Notas de inferencia (ZIP M7b)

```
λ = exp(intercept + X_escalado · β)          # X = 45 covariables (scaler_zip)
ψ = ψ_posterior[cluster_LISA]                # [HH, LL, ns]
E[Y] = (1 − ψ) · λ                           # predicción puntual
```

Validado: corr(real, predicho) ≈ 0.93 en la semana de referencia.
