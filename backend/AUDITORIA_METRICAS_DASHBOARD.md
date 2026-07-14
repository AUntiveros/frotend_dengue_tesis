# Auditoria de metricas del dashboard

Fecha: 2026-07-13

## Problema detectado

El primer JSON actualizado del dashboard recalculo RMSE, MAE y R2 directamente
desde `series_operativas_recalibradas_h4_distrital.csv`. Esa tabla sirve para
visualizar series operativas continuas, pero no debe usarse como fuente de
ranking porque mezcla una ventana distinta a la validacion oficial.

Resultado erroneo observado:

| Modelo | R2 erroneo en dashboard |
|---|---:|
| Ensemble operativo calibrado H4 | -0.371 |
| LightGBM Optuna corregido reentrenado | 0.914 |

La causa fue evaluar sobre `n=240157` filas disponibles en la serie operativa,
mientras que la validacion reentrenada oficial reporta `n=92659`.

## Fuentes correctas finales

Para el ranking y los numeros que se muestran al usuario:

`tesis_presentacion_julio2026/predicciones/comparacion_41_modelos_multinivel_metricas.csv`

Para las series y mapas operativos actualizados a SE26/2026:

`tesis_presentacion_julio2026/predicciones/series_operativas_recalibradas_h4_distrital.csv`
`tesis_presentacion_julio2026/predicciones/forecast_h4_continuo_calibrado_distrital.csv`

Para las series historicas de los modelos top 5 heredados desde `try_julio_2026`:

`frontend/backend/data/top5_try_julio_series.csv`

## Metricas corregidas en el dashboard

El dashboard debe reflejar la tabla final de 41 modelos. Los primeros puestos son:

| Modelo | n | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Ensemble enrutado por estrato | 296074 | 10.869 | 0.943 | 0.622 |
| INLA Multinivel (Poisson) | 296074 | 11.366 | 1.037 | 0.587 |
| INLA-BYM2 + lags (Poisson) | 296074 | 11.368 | 1.038 | 0.587 |
| ZIP M7c splines | 207008 | 14.196 | 1.531 | 0.536 |
| ZIP M7a log-lineal | 207008 | 14.326 | 1.527 | 0.527 |

La disponibilidad de serie/mapa no debe alterar el orden visual del ranking.
INLA Multinivel, INLA-BYM2, ZIP M7c y ZIP M7a se cargan como series historicas
de validacion reconstruidas desde `try_julio_2026`; no se muestran como
pronostico operativo SE26 si no fueron reentrenados con la AMT actualizada.

## Pronostico vigente SE30

Se genero `frontend/backend/data/top5_forecast_vigente.csv` para habilitar mapa
y p vigente cuando existe inferencia real:

| Modelo | p SE30 nacional | distritos con p vigente | Nota |
|---|---:|---:|---|
| Ensemble enrutado por estrato | 3072.756 | 1891 | Inferencia operativa calibrada SE26 |
| INLA-BYM2 + lags (Poisson) | 1359.026 | 1891 | R-INLA actualizado, forecast future |
| ZIP M7c splines | 703.678 | 1738 | Trazas de try_julio aplicadas a AMT actualizada |
| ZIP M7a log-lineal | 700.459 | 1738 | Trazas de try_julio aplicadas a AMT actualizada |
| INLA Multinivel (Poisson) | NA | 0 | La corrida R-INLA actualizada en modo final volvio a crashear |

El INLA Multinivel queda en el ranking oficial y con serie historica, pero sin
mapa/pronostico vigente hasta resolver el crash del ajuste actualizado.

## Correccion aplicada

`frontend/backend/actualizar_dashboard_desde_tesis_julio2026.py` ahora arma el
registro de modelos desde la tabla final de 41 modelos. Los modelos que ademas
tienen inferencia distrital operativa vigente se marcan con `map=true`, pero no
se colocan por encima de la tabla oficial.

El modelo predeterminado vuelve a ser `ENS_H4`, correspondiente a `Ensemble
enrutado por estrato`.
