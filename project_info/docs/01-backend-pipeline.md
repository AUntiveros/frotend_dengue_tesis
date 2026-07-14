# Backend — Pipeline de pre-cálculo

## Archivos

```
backend/
  generador_dashboard.py      orquestador → predicciones_semana.json
  inferencia/
    comun.py                  carga AMT, semana de referencia, helpers
    modelos.py                cargadores e inferencia de cada modelo
  requirements.txt
  data/AMT_final.parquet      AMT (insumo)
  data/AMT_metadata.json      features, cortes train/val, horizontes
  modelos/                    artefactos entrenados (.json/.txt/.pkl/.npy/.pt/.nc)
```

## Flujo de `generador_dashboard.py`

1. **Carga** `AMT_final.parquet` (dedup por `ubigeo,fecha`) + `AMT_metadata.json`.
2. **Semana de referencia** = última fecha con casos nacionales > 0
   (`comun.resolver_semana_ref`). Las filas posteriores en la AMT son relleno
   (ceros) para sostener los lags. Hoy = **2026-03-09** (≈ SE11/2026).
3. **Ejes temporales**:
   - `hist_weeks` = historia observada COMPLETA desde **2018** (426 semanas).
   - `pred_weeks` = periodo de test post `corte_val` (114 semanas, 2024→ref).
   - `pred_offset` = índice en `hist_weeks` donde empieza la predicción (312).
4. **Inferencia** de cada modelo sobre un panel con buffer de 18 semanas
   (cubre el shift de 4 y la ventana LSTM de 12).
5. **Alineación** (`pivot_pred`): para modelos de horizonte (árboles, BiLSTM)
   la salida en la semana `w` es el forecast hecho en `w` para `w+4`; se
   desplaza la columna +4 para que cada semana muestre la predicción que le
   corresponde. El ZIP es contemporáneo (sin shift).
6. **Ensamblado** del JSON por UBIGEO + bloque nacional + registro de modelos.
7. **Escritura** en `backend/data/` y copia a `public/data/`.

Tiempo ≈ 3 min (domina el BiLSTM secuencial). Re-ejecutar cada semana al
actualizar la AMT.

## Estructura de `predicciones_semana.json`

```jsonc
{
  "meta": {
    "semana_ref": "2026-03-09", "se_label": "SE11/2026",
    "n_distritos": 1891,
    "n_hist_semanas": 426,        // observado desde 2018
    "pred_offset": 312,           // predicción empieza aquí en hist_weeks
    "n_pred_semanas": 114,        // periodo de test
    "hist_desde": "2018-01-01",
    "corte_train": "2022-12-31", "corte_val": "2023-12-31",
    "horizonte_real": "H4",
    "modelo_default": "M7b",
    "umbrales_riesgo": { "sin":0,"bajo":5,"moderado":20,"alto":100 }
  },
  "models": [ { "id","name","family","rmse","mae","r2","map","scope","disponible","has_shap","description" }, ... ],
  "hist_weeks":  ["SE01/2018", ..., "SE11/2026"],   // 426 etiquetas
  "hist_fechas": ["2018-01-01", ..., "2026-03-09"], // ISO, para líneas de corte
  "districts": {
    "250101": {
      "name","dep","prov","ccdd","lisa","pop",
      "hist": [ ...426 casos observados... ],
      "pred": {
        "M7b": { "p", "lo", "hi", "series":[...114...], "shap":[["feat",val],...] },
        "XGB_LOG": { ... }, ...
      }
    }, ...
  },
  "nacional": {
    "hist": [ ...426... ],
    "pred": { "M7b": {"p","series"}, ..., "PROPHET": {"p","series","lo","hi"} }
  }
}
```

### Convenciones clave para el frontend

- `pred.series` tiene longitud `n_pred_semanas` (114). Para alinearla al eje
  completo, anteponer `pred_offset` nulos: semana `i` → si `i < pred_offset`
  no hay predicción; si no, `series[i - pred_offset]`.
- `pred.p` = forecast puntual H4 para `ref+4` (el número de la tarjeta y el mapa).
- `district.hist[last]` = casos reportados en la semana de referencia
  (lo que pinta el toggle "Reportado").
