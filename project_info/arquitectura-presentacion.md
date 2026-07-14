# Arquitectura del Sistema — Dashboard de Predicción de Dengue

**Proyecto:** Tesis PUCP — Predicción de brotes epidemiológicos de dengue a nivel distrital  
**Alcance:** 1 891 distritos · 426 semanas epidemiológicas · Perú 2018–2026

---

## 1. Descripcion del sistema

Sistema de visualización interactiva para la predicción de casos de dengue por semana epidemiológica y distrito. Combina modelos de machine learning (ZIP Bayesiano, XGBoost, LightGBM, Random Forest, BiLSTM, Prophet) con un mapa coroplético distrital y series temporales desde 2018.

---

## 2. Atributos de calidad

| Atributo | Descripcion | Escenario de validacion |
|----------|-------------|------------------------|
| **Rendimiento** | Toda interaccion (cambio de modelo, seleccion de distrito, rango temporal) responde en menos de 100 ms | Un solo `fetch` al inicio; cambios son lookups en memoria RAM |
| **Disponibilidad** | Sin dependencia de servidor en tiempo de ejecucion | JSON estatico servido por CDN (Vercel); sin base de datos ni API activa |
| **Mantenibilidad** | Actualizacion semanal reducida a re-ejecutar un script Python | Pipeline encapsulado en `generador_dashboard.py`; frontend no cambia |
| **Portabilidad** | Ejecutable en cualquier hosting estatico | Vite genera artefactos `dist/` autocontenidos; sin runtime de servidor |
| **Exactitud** | Predicciones distritales con correlacion real/predicho >= 0.93 (modelo M7b) | Validado en semana de referencia SE11/2026 contra casos NETLAB |
| **Escalabilidad** | Agregar nuevo modelo = agregar columna al JSON; el frontend lo detecta sin modificacion | `models[]` dinamico; componentes usan `activeModelId` como clave |
| **Usabilidad** | Usuario no epidemiologo puede interpretar riesgo sin leer valores numericos | Escala semantica: sin riesgo / bajo / moderado / alto / muy alto con colormap YlOrBr |
| **Trazabilidad** | Cada prediccion es atribuible a un modelo, horizonte y semana especifica | Metadatos `meta.semana_ref`, `meta.horizonte_real`, `model.id` en el JSON |

---

## 3. Diagrama de arquitectura — Conexion frontend y backend

```mermaid
flowchart TD
    subgraph FUENTES["Fuentes de datos"]
        NETLAB["NETLAB\nCasos confirmados de dengue"]
        SENAMHI["SENAMHI\nPrecipitacion y temperatura"]
        MODIS["MODIS / Landsat\nNDVI, EVI, LST"]
    end

    subgraph BACKEND["Backend — Pre-calculo semanal (Python)"]
        AMT["AMT_final.parquet\n908 980 filas · 41+ features"]
        META["AMT_metadata.json\nFeatures, cortes, horizontes"]
        GEN["generador_dashboard.py\nOrquestador de inferencia"]
        subgraph MODELOS["Modelos entrenados (artefactos .pkl / .json / .npy / .pt / .nc)"]
            M7B["M7b ZIP Horseshoe\nR2=0.942 · RMSE=4.89"]
            XGB["XGBoost log1p / Poisson"]
            LGB["LightGBM Optuna"]
            RF["Random Forest"]
            STACK["Stacking XGB+LGB+RF"]
            BILSTM["BiLSTM Huber\n2 capas bidireccionales"]
            PROPHET["Prophet nacional"]
        end
        JSON["predicciones_semana.json\n5.8 MB · 1891 distritos\n7 modelos distritales + Prophet"]
    end

    subgraph FRONTEND["Frontend — React 18 + Vite 6"]
        LOAD["predictions.ts\nloadDashboard() · tipos TS"]
        APP["App.tsx\nEstado global:\nactiveModelId · selectedUbigeo · obsWeek"]
        MAP["MapPanel\nLeaflet · coropletas · toggle"]
        DETAIL["DetailPanel\nTarjeta H4 · IC95% · riesgo"]
        SERIES["TimeSeriesChart\nRecharts · Brush · 426 semanas"]
        SHAP["ShapChart\nImportancia de variables"]
        MODELS["ModelSelector\n13 modelos · metricas"]
    end

    subgraph DESPLIEGUE["Despliegue — Vercel (temporal)"]
        CDN["CDN Vercel\nSirve dist/ + public/data/"]
    end

    NETLAB --> AMT
    SENAMHI --> AMT
    MODIS --> AMT
    AMT --> GEN
    META --> GEN
    M7B --> GEN
    XGB --> GEN
    LGB --> GEN
    RF --> GEN
    STACK --> GEN
    BILSTM --> GEN
    PROPHET --> GEN
    GEN --> JSON

    JSON -->|"npm run build\nVite copia public/ a dist/"| CDN
    CDN -->|"1 fetch HTTP GET\nal cargar la pagina"| LOAD
    LOAD --> APP
    APP --> MAP
    APP --> DETAIL
    APP --> MODELS
    DETAIL --> SERIES
    DETAIL --> SHAP
```

---

## 4. Diagrama de casos de uso

```mermaid
flowchart LR
    EP["Epidemiologo\nMINSA / DIRESA"]
    INV["Investigador\n(Tesista)"]
    GES["Gestor de salud\npublica"]

    subgraph UC["Sistema — Dashboard de Dengue"]
        UC1["Visualizar mapa de riesgo distrital"]
        UC2["Seleccionar modelo de prediccion"]
        UC3["Consultar serie temporal por distrito"]
        UC4["Comparar casos reportados vs predichos"]
        UC5["Analizar importancia de variables"]
        UC6["Filtrar por rango temporal"]
        UC7["Evaluar metricas de precision del modelo"]
        UC8["Identificar distritos de alto riesgo"]
    end

    EP --> UC1
    EP --> UC3
    EP --> UC4
    EP --> UC6
    EP --> UC8

    INV --> UC1
    INV --> UC2
    INV --> UC3
    INV --> UC5
    INV --> UC6
    INV --> UC7

    GES --> UC1
    GES --> UC8
    GES --> UC4
```

---

## 5. Diagrama de flujo de uso

```mermaid
sequenceDiagram
    actor Usuario
    participant Browser as Navegador
    participant Vite as Servidor Vite / CDN Vercel
    participant Mem as Memoria del cliente

    Usuario->>Browser: Abre la URL del dashboard
    Browser->>Vite: GET /data/predicciones_semana.json
    Vite-->>Browser: 5.8 MB JSON (unico fetch)
    Browser->>Mem: Carga Dashboard en memoria

    Note over Browser,Mem: A partir de aqui, sin red

    Usuario->>Browser: Hace clic en un distrito del mapa
    Browser->>Mem: lookup districts[ubigeo]
    Mem-->>Browser: DistrictRec con hist[] y pred{}
    Browser->>Browser: Renderiza serie temporal y tarjeta H4

    Usuario->>Browser: Cambia modelo (ej. XGBoost)
    Browser->>Mem: lookup pred[XGB_LOG].p por distrito
    Mem-->>Browser: Valores actualizados
    Browser->>Browser: Repinta coropletas del mapa

    Usuario->>Browser: Activa toggle "Reportado"
    Browser->>Mem: lookup district.hist[obsWeek]
    Mem-->>Browser: Casos observados en semana seleccionada
    Browser->>Browser: Repinta mapa con datos reales

    Usuario->>Browser: Mueve el Brush (rango temporal)
    Browser->>Mem: Recalcula obsWeek por posicion del brush
    Mem-->>Browser: Nueva semana de referencia
    Browser->>Browser: Actualiza mapa y serie
```

---

## 6. Flujo del pipeline de actualizacion semanal

```mermaid
flowchart TD
    START["Nueva semana epidemiologica disponible"]
    UPD["Analista actualiza\nAMT_final.parquet\n(NETLAB + SENAMHI + MODIS)"]
    RUN["python generador_dashboard.py\nDuracion: ~3 minutos"]

    subgraph INFERENCIA["Inferencia por modelo"]
        INF1["ZIP M7b: numpy puro\nintercept + X @ beta_mean"]
        INF2["XGBoost / LightGBM / RF:\npredict + expm1"]
        INF3["Stacking: bases en log\n@ stack_coef"]
        INF4["BiLSTM: secuencia 12 sem\nPyTorch inference"]
        INF5["Prophet: serie nacional\nagregate + regresores"]
    end

    PIVOT["Alineacion temporal (pivot_pred)\nShift +4 sem para modelos de horizonte"]
    SERIAL["Serializacion JSON\npor UBIGEO + bloque nacional"]
    COPY["Copia a public/data/\npara Vite"]
    DEPLOY["git push / Vercel deploy\nCDN actualizado"]

    START --> UPD --> RUN
    RUN --> INF1
    RUN --> INF2
    RUN --> INF3
    RUN --> INF4
    RUN --> INF5
    INF1 --> PIVOT
    INF2 --> PIVOT
    INF3 --> PIVOT
    INF4 --> PIVOT
    INF5 --> PIVOT
    PIVOT --> SERIAL --> COPY --> DEPLOY
```

---

## 7. Despliegue en Vercel (temporal)

El sistema se despliega como sitio estatico en Vercel durante la fase de evaluacion de la tesis.

```mermaid
flowchart LR
    subgraph LOCAL["Maquina local"]
        SRC["src/ · backend/\ncodigo fuente"]
        BUILD["npm run build\nVite compila a dist/"]
        PY["python generador_dashboard.py\nGenera JSON"]
    end

    subgraph REPO["Repositorio Git"]
        MAIN["rama main\ndist/ + public/data/ incluidos"]
    end

    subgraph VERCEL["Vercel (CDN global)"]
        EDGE["Edge Network\nTiempo de respuesta < 50ms"]
        STATIC["Archivos estaticos\ndist/ · predicciones_semana.json"]
    end

    subgraph USUARIO["Usuario final"]
        BROWSER["Navegador\nChrome / Firefox / Edge"]
    end

    PY -->|genera| SRC
    SRC --> BUILD
    BUILD --> MAIN
    MAIN -->|"Vercel detecta push\nauto-deploy"| STATIC
    STATIC --> EDGE
    EDGE -->|HTTPS| BROWSER
```

**Caracteristicas del despliegue:**

| Parametro | Valor |
|-----------|-------|
| Tipo de hosting | Estatico (sin servidor) |
| Tiempo de build | < 30 segundos (Vite) |
| Actualizacion | Re-deploy manual post-pipeline Python |
| Costo | Plan gratuito Vercel (hobby) |
| Dominio | `*.vercel.app` (temporal para tesis) |
| HTTPS | Automatico via Vercel |
| Cache | CDN distribuido; JSON con `Cache-Control` configurable |

---

## 8. Stack tecnologico completo

| Capa | Tecnologia | Version | Funcion |
|------|-----------|---------|---------|
| Framework UI | React | 18 | Componentes y estado reactivo |
| Build tool | Vite | 6 | Dev server, bundler, sirve `public/` |
| Lenguaje | TypeScript | 5 | Tipado estatico del JSON |
| Mapas | Leaflet / react-leaflet | — | Coropletas distritales, click, zoom |
| Graficos | Recharts | — | Serie temporal, Brush, barras SHAP |
| Estilos | Tailwind CSS | 3 | Utility-first, sin CSS personalizado |
| Datos geo | TopoJSON INEI 2025 | — | Geometria 1 891 distritos, clave UBIGEO |
| Backend ML | Python 3.11 | — | Pipeline de pre-calculo |
| Modelos | sklearn 1.6.1 / XGBoost / LightGBM / PyTorch / Prophet | — | Inferencia de 13 modelos |
| Datos fuente | Parquet (pandas) | — | AMT_final.parquet, 908 980 filas |
| Despliegue | Vercel | — | Hosting estatico CDN global |

---

## 9. Decisiones arquitectonicas

| Decision | Alternativa considerada | Razon de eleccion |
|----------|------------------------|-------------------|
| JSON estatico pre-calculado | API REST en tiempo real | Sin servidor dedicado; 3 min de inferencia es inaceptable por request |
| Un solo fetch al inicio | Fetch por interaccion | Lookups en memoria = 100x mas rapido que red |
| Vite sin SSR | Next.js con SSR | No se requiere renderizado en servidor; SPA es suficiente |
| ColorBrewer YlOrBr | Escala fria azul | La escala azul hacia parecer "cero" a distritos con transmision moderada |
| Modelo M7b como default | Seleccion manual siempre | Mejor R2 (0.942); usuario tiene punto de partida de alta calidad |
| TopoJSON INEI 2025 | GeoJSON / Shapefile | Peso 10 veces menor; clave UBIGEO estandarizada |
| Horizonte H4 unico activo | Multiples horizontes | Solo H4 esta entrenado y validado; los demas se deshabilitan para no engañar |
