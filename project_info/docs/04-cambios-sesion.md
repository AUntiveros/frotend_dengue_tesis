# Historial de cambios

## Sesión 1 — De mockup a backend funcional

Punto de partida: frontend bonito con datos mock (`mockData.ts`), sin backend.

**Backend nuevo**
- `backend/generador_dashboard.py` + paquete `inferencia/` (`comun.py`, `modelos.py`).
- Instaladas libs: `xgboost`, `lightgbm`, `torch` (CPU), `prophet`.
- Inferencia real de 7 modelos distritales + Prophet nacional → `predicciones_semana.json`.
- ZIP M7b validado: corr(real, predicho) ≈ 0.93.

**Frontend recableado**
- `src/data/predictions.ts` (capa de datos) reemplaza `mockData.ts` + `types.ts`.
- Selector de modelo ahora mueve mapa + serie + métricas (antes solo métricas).
- Métricas reales de los 13 modelos.
- Serie temporal observado vs predicho + slider Brush.
- Importancia de variables real (β·x del ZIP, ganancia en árboles).
- Horizontes H1/H8/H12/H24 deshabilitados (solo H4 entrenado).

## Sesión 2 — Correcciones de visualización y temporalidad

Feedback del usuario: mapa demasiado azul, historia recortada a SE12/2025,
falta toggle reportado/predicho, realce de departamento perdido.

1. **Colormap** — de azul-frío (`#1e40af→#dc2626`, hacía parecer todo cero) a
   secuencial cálido **YlOrBr** (`#f8fafc → #8c2d04`). Cero ≈ blanco; focos
   endémicos en rojo-marrón. Leyenda actualizada.

2. **Temporalidad** — antes solo 52 semanas (desde SE12/2025). Ahora:
   - Historia observada **completa desde 2018** (426 semanas).
   - Predicción out-of-sample sobre el periodo de test (114 semanas, post `corte_val`).
   - `pred_offset` en el JSON para alinear ambas en el frontend.
   - Líneas de referencia `corte_train` (2022-12-31) y `corte_val` (2023-12-31).
   - Brush por defecto en el periodo reciente; rango completo disponible.

3. **Toggle mapa Reportado / Predicho (H4)** — nuevo control arriba-derecha.
   "Reportado" pinta casos observados en la semana de referencia;
   "Predicho" pinta la predicción H4 del modelo activo.

4. **Realce de departamento** — borde índigo grueso (`#312e81`, weight 3),
   visible ahora que el mapa ya no es azul. Distritos fuera del departamento
   seleccionado se atenúan a gris claro.

**Tamaño JSON**: 5.8 MB → 12 MB (por la historia completa). Gzip ≈ 2-3 MB en
producción. Si se requiere reducir: redondear `series`/`hist` a enteros.

## Sesión 3 — Detalles profesionales + deploy

1. **Máscara gris en modo departamento** — los departamentos no seleccionados ya
   no quedan blancos; se atenúan con gris suave (`#94a3b8` opacity 0.35),
   manteniendo el contexto mientras resalta el departamento seleccionado.

2. **Cola de pronóstico (SE12–SE15)** — la serie temporal se extiende 4 semanas
   tras la última observada, como predicho-sin-observado. Resuelve el desajuste
   "mapa H4 vs serie": ahora el **extremo de la serie = valor H4 del mapa** por
   modelo. Forward (XGB/LGB/RF/Stacking/BiLSTM) pronostican SE15; ZIP es
   contemporáneo (nowcast SE11, cola null). Zona de pronóstico sombreada + nuevos
   campos `future_weeks`/`future_fechas` en el JSON.

3. **Scrubber de semana en el mapa (modo Reportado)** — slider sobre las 426
   semanas; al moverlo el mapa pinta los casos reportados de esa semana
   (navegación histórica estilo CDC). El selector de fechas ahora "dibuja" sobre
   el mapa lo seleccionado.

4. **Deploy** — `.gitignore`, `vercel.json`, repo en GitHub
   (`AUntiveros/frotend_dengue_tesis`). Vercel sirve frontend + `public/` estático.

## Sesión 4 — Unificación del control temporal

- Se **eliminó el slider redundante sobre el mapa**. La semana reportada que pinta
  el mapa (modo Reportado) ahora la controla el **selector de rango bajo la serie
  temporal** (`<Brush>`): su borde derecho fija la semana del mapa. Estado `obsWeek`
  elevado a `App` y compartido entre `MapPanel` (lectura) y `TimeSeriesChart`
  (escritura vía `onWeekChange`).
- **Deploy queda en H4** (único horizonte entrenado). Pendientes en `05-*.md`.
