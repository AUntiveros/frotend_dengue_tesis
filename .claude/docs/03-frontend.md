# Frontend — Componentes y flujo

React 18 + Vite. Un solo fetch del JSON estático; todo el estado es lookup en
memoria.

## Archivos

```
src/
  data/predictions.ts     tipos + loadDashboard() + riskColor/riskLabel
  App.tsx                 carga datos, estado global (modelo, distrito)
  components/
    MapPanel.tsx          mapa Leaflet (coropletas por modelo/fuente)
    DetailPanel.tsx       panel lateral (tarjeta H4, serie, SHAP, métricas)
    TimeSeriesChart.tsx   serie observado vs predicho + Brush + cortes
    ShapChart.tsx         importancia de variables (barras)
    ModelSelector.tsx     dropdown de 13 modelos con métricas
```
(`mockData.ts` y `types.ts` se eliminaron — reemplazados por `predictions.ts`.)

## Flujo de datos

```
App: loadDashboard() ──► data: Dashboard
     estado: activeModelId, selectedUbigeo
       │
       ├─► MapPanel   (data, activeModelId, modelHasMap, onDistrictClick, selectedUbigeo)
       └─► DetailPanel(data, selectedUbigeo, activeModel)
              ├─► TimeSeriesChart
              └─► ShapChart
```

## Lógica clave

### MapPanel
- Pinta cada distrito con `riskColor(valor)` donde el valor depende de:
  - **Fuente**: toggle `Reportado` (= `district.hist[last]`, casos en semana ref)
    vs `Predicho (H4)` (= `district.pred[modelId].p`).
  - **Métrica**: `Casos / SE` vs `Tasa ×100k` (divide por población).
- Si el modelo activo no es distrital (`map=false`), pinta M7b y muestra un
  banner ("modelo sin desagregación distrital").
- Click en distrito → selección + zoom al departamento (realce con borde
  índigo grueso `#312e81`).

### Colormap (`riskColor`, ColorBrewer YlOrBr secuencial)
```
<0  gris #e2e8f0   (sin predicción para el modelo)
=0  #f8fafc        (casi blanco → la mayoría del mapa queda silencioso)
≤2  #fee391  ≤5 #fec44f  ≤20 #fe9929  ≤50 #ec7014  ≤100 #cc4c02  >100 #8c2d04
```
Diseño: cero ≈ blanco para que los focos endémicos (HH) destaquen en rojo-marrón,
en vez del azul anterior que hacía parecer todo "cero".

### TimeSeriesChart
- Eje X = 426 semanas desde 2018. Línea azul = observado (full).
- Línea naranja = predicho (solo periodo test), alineada con `pred_offset`:
  semana `i` usa `series[i - pred_offset]` (nulo antes del offset).
- Líneas de referencia verticales en `corte_train` y `corte_val` (zona de test).
- `<Brush>` inferior = slider de rango de fechas (por defecto desde validación).

### DetailPanel
- Distrito seleccionado → `data.districts[ubigeo]`; sin selección → agregado
  nacional (`data.nacional`).
- Tarjeta H4 (p + IC95% + etiqueta de riesgo). H1/H8/H12/H24 deshabilitados.
- SHAP: `pred.shap` si existe; si no, aviso "en proceso".

## Tipos (`predictions.ts`)

`Dashboard`, `ModelInfo`, `DistrictRec`, `PredEntry`, `NacionalPred` — ver el
archivo. `loadDashboard()` hace `fetch('/data/predicciones_semana.json')`.
