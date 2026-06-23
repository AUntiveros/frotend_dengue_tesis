# Dashboard de Dengue — Visión general

Sistema de visualización de predicción de brotes epidemiológicos de dengue a
nivel **distrito × semana epidemiológica** (Perú, 1 891 distritos). Tesis PUCP.

## Stack real (verificado)

| Capa      | Tecnología                                            |
|-----------|-------------------------------------------------------|
| Frontend  | **React 18 + Vite 6** (NO Next.js) + Leaflet + Recharts + Tailwind |
| Backend   | Python (pandas/numpy/sklearn/xgboost/lightgbm/torch/prophet) |
| Mapas     | TopoJSON distrital INEI 2025 (`public/data/distritos.topojson`, clave `UBIGEO`) |
| Datos     | `backend/data/AMT_final.parquet` (908 980 filas, 2018→2026) |

## Arquitectura: pre-cálculo semanal

```
AMT_final.parquet ──► backend/generador_dashboard.py ──► predicciones_semana.json (estático)
                                                              │
                                          public/data/  ◄─────┘ (servido por Vite)
                                                              │
                          React: 1 fetch → todo en memoria ──►  mapa + panel + serie
```

El cliente hace **un solo fetch**. Cambiar modelo, distrito o rango temporal
son lookups en memoria → reaccionan en milisegundos (sin recálculo, sin red).

## Estado funcional

- ✅ 7 modelos con predicción **distrital** real en el mapa
  (M7b ZIP, XGBoost log1p, XGBoost Poisson, LightGBM Optuna, RF, Stacking, BiLSTM).
- ✅ Prophet como modelo **nacional** (serie agregada).
- ✅ Selector de modelo mueve mapa + serie + métricas.
- ✅ Serie temporal observado vs predicho desde **2018**, con slider de rango.
- ✅ Toggle mapa **Reportado / Predicho (H4)**.
- ✅ Importancia de variables real (β·x del ZIP, ganancia en árboles).
- ✅ Métricas reales de los 13 modelos en el selector.

## Documentos

- `01-backend-pipeline.md` — generador y estructura del JSON.
- `02-modelos-inferencia.md` — recetas de inferencia por modelo.
- `03-frontend.md` — componentes y flujo de datos.
- `04-cambios-sesion.md` — historial de cambios.
- `05-limitaciones-y-pendientes.md` — qué falta / decisiones abiertas.

## Cómo correr

```bash
cd backend && python -m pip install -r requirements.txt
python generador_dashboard.py          # genera predicciones_semana.json (~3 min)
cd .. && npm run dev                    # http://localhost:5173
```
