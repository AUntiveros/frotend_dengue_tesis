# Modelos e inferencia

Métricas reales reportadas (test set, H = 4 semanas). `map` = produce
predicción distrital seleccionable; `scope` = distrital/nacional.

| id        | Modelo               | Familia          | RMSE  | MAE  | R²     | map | artefacto |
|-----------|----------------------|------------------|------:|-----:|-------:|-----|-----------|
| `M7b`     | ZIP Horseshoe        | Bayesiano        | 4.89  | 0.62 | 0.942  | ✅  | `zip_beta_mean/intercept/psi_posterior.npy` + `scaler_zip.pkl` |
| `M7a`     | ZIP log-lineal       | Bayesiano        | 4.93  | 0.62 | 0.941  | ❌  | solo métricas (sin arrays) |
| `M7c`     | ZIP splines          | Bayesiano        | 5.09  | 0.63 | 0.937  | ❌  | solo métricas |
| `STACK`   | Stacking XGB+LGB+RF  | ML Ensamble      | 11.94 | 0.90 | 0.544  | ✅  | `stack_coef.npy` + bases |
| `BILSTM`  | BiLSTM Huber         | Deep Learning    | 12.03 | 0.73 | 0.372  | ✅  | `bilstm_best.pt` + `scaler_lstm.pkl` |
| `LGB_OPT` | LightGBM Optuna      | Gradient Boosting| 14.05 | 0.95 | 0.369  | ✅  | `lgb_optuna.txt` |
| `M7d`     | ZIP XGB-offset       | Bayesiano        | 14.08 | 0.91 | 0.519  | ❌  | solo métricas |
| `LGB_GLB` | LightGBM global      | Gradient Boosting| 14.35 | 0.96 | 0.342  | ❌  | features incompatibles (20≠41) |
| `RF`      | Random Forest        | Bagging          | 14.58 | 0.98 | 0.320  | ✅  | `rf_model.pkl` |
| `XGB_LOG` | XGBoost log1p        | Gradient Boosting| 14.67 | 0.98 | 0.312  | ✅  | `xgb_log.json` |
| `XGB_POI` | XGBoost Poisson      | Gradient Boosting| 14.69 | 1.05 | 0.310  | ✅  | `xgb_poisson.json` |
| `M7e`     | ZIP NegBin           | Bayesiano        | 15.24 | 1.01 | 0.436  | ❌  | solo métricas |
| `PROPHET` | Prophet (mediana HH) | Series de tiempo | 6912  | 3131 | -0.042 | ❌ (nacional) | `prophet_full.pkl` |

## Features

`AMT_metadata.json → all_features` = **41 columnas** (autorregresivas, climáticas,
satelitales, interacciones, espaciales LISA, topográficas, cíclicas, tasas).
Árboles y BiLSTM usan las 41; el ZIP M7b usa **45** (las 41 metadata sin
`flag_anomalia` + `evi, agua_expuesta, agua_en_vegetacion, seq_lst_norm, mes`),
con el orden exacto guardado en `scaler_zip.feature_names_in_`.

## Recetas de inferencia

### ZIP M7b (mejor modelo — inferencia rápida en numpy, sin PyMC)

```python
X = scaler_zip.transform(df[feature_names_in_].fillna(0))   # 45 features
lam = exp(intercept + X @ beta_mean)                        # λ = tasa esperada
psi = psi_posterior[cluster_LISA]   # [HH=0.0012, LL=0.815, ns=0.0145]
E[Y] = (1 - psi) * lam                                      # predicción puntual
Var  = (1-psi)*lam*(1+psi*lam)  →  IC95% = E ± 1.96·√Var
```
`cluster_code`: 4=HH, 1=LL, otro=ns. **Validado: corr(real, predicho) ≈ 0.93**
en la semana de referencia (top distritos = Pucallpa, Yurimaguas, Sullana,
Iquitos — focos endémicos reales).

### Árboles (XGBoost / LightGBM / RF)

Target = `target_h4_log` (casos 4 semanas adelante, log1p) salvo XGBoost
Poisson (`target_h4` crudo, `objective=count:poisson`).
```python
raw = modelo.predict(df[all_features].fillna(0))
pred = expm1(clip(raw, 0))        # log → casos   (Poisson: clip directo)
```
Importancia: `feature_importances_` (XGB/RF) / `feature_importance("gain")` (LGB).

### Stacking

Meta-modelo `LinearRegression(positive=True)` sobre las predicciones EN LOG de
`[xgb_log, lgb_optuna, rf]`; `stack_coef.npy` = pesos.
```python
base_log = [xgb_log.predict(X), lgb_optuna.predict(X), rf.predict(X)]  # log
pred = expm1(clip(base_log @ stack_coef, 0))
```
Importancia = pesos relativos de los 3 modelos base.

### BiLSTM (PyTorch)

`BiLSTMForecaster(n_feat=41, h1=64, h2=32)`, 2 capas bidireccionales + BatchNorm.
Secuencia de 12 semanas (`scaler_lstm` MinMax), target `target_h4_log`.
```python
seq = scaler_lstm.transform(ultimas_12_semanas[all_features])  # (12, 41)
pred = expm1(clip(model(seq), 0))
```

### Prophet (nacional)

`prophet_full.pkl` entrenado sobre la **serie nacional agregada** (suma de casos)
con regresores `ndvi_lag4, precip_lag4, lst_lag4, tmean_x_ndvi` (medias nacionales).
No es distrital → métricas pobres (R²=-0.042). Se muestra como serie nacional.

## Notas

- `scikit-learn` debe ser **1.6.1** para deserializar los `.pkl` (se serializaron
  con 1.8 → solo warning, no rompe).
- Los artefactos `*_global` (lgb/xgb/rf) usan subconjuntos de 20-30 features
  incompatibles con las 41 de `all_features`; el generador los descarta
  automáticamente por validación de `n_features`.
- Las variantes ZIP M7a/c/d/e no guardaron arrays de inferencia rápida (solo
  M7b). Sus traces `.nc` requieren PyMC/ArviZ (no instalados) y rutas de
  inferencia más complejas (splines, offset XGB). Aparecen como comparativa.
