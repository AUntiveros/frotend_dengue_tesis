// Capa de datos del dashboard: consume el archivo estático pre-calculado por
// backend/generador_dashboard.py. Un único fetch; todos los cambios de estado
// (modelo, distrito, rango temporal) son lookups en memoria → reaccionan en ms.

export type Lisa = 'HH' | 'LL' | 'LH' | 'HL' | 'ns'
export type Scope = 'distrital' | 'nacional'

export interface ModelInfo {
  id: string
  name: string
  family: string
  description: string
  n?: number | null
  rmse: number
  mae: number
  r2: number
  map: boolean          // tiene predicción distrital seleccionable en el mapa
  scope: Scope
  has_shap: boolean
  disponible: boolean
}

export interface PredEntry {
  p: number | null      // predicción puntual H4 vigente; null si solo hay serie histórica
  lo: number | null
  hi: number | null
  series: number[]      // serie alineada a hist_weeks (predicho)
  shap?: [string, number][]
}

export interface DistrictRec {
  name: string
  dep: string
  prov: string
  ccdd: string
  lisa: Lisa
  pop: number
  hist: number[]                       // casos observados, alineado a hist_weeks
  pred: Record<string, PredEntry>      // por id de modelo
}

export interface NacionalPred {
  p: number | null
  series: number[]
  lo?: number[]
  hi?: number[]
}

export interface Dashboard {
  meta: {
    generado: string
    semana_ref: string
    se_label: string
    n_distritos: number
    n_hist_semanas: number
    pred_offset: number          // índice en hist_weeks donde empieza la predicción
    n_pred_semanas: number
    n_future_semanas: number     // semanas de pronóstico tras la última observada
    future_weeks: string[]       // etiquetas SExx de la cola de pronóstico
    future_fechas: string[]
    hist_desde: string
    corte_train: string
    corte_val: string
    horizonte_real: string
    horizontes_no_entrenados: string[]
    modelo_default: string
    umbrales_riesgo: Record<string, number>
  }
  models: ModelInfo[]
  hist_weeks: string[]
  hist_fechas: string[]
  districts: Record<string, DistrictRec>
  nacional: { hist: number[]; pred: Record<string, NacionalPred> }
}

export async function loadDashboard(): Promise<Dashboard> {
  const res = await fetch('/data/predicciones_semana.json')
  if (!res.ok) throw new Error(`No se pudo cargar predicciones_semana.json (${res.status})`)
  return res.json()
}

// Escala de riesgo secuencial cálida (ColorBrewer YlOrBr).
// Cero ≈ blanco (la mayoría del mapa, sierra sin transmisión queda silenciosa);
// los focos endémicos (HH) destacan en marrón-rojo intenso.
// Valor negativo = distrito sin predicción para el modelo activo (gris neutro).
export function riskColor(value: number, metric: 'cases' | 'rate' = 'cases'): string {
  if (value < 0) return '#e2e8f0'
  const displayValue = metric === 'cases' ? Math.round(value) : Number(value.toFixed(1))
  if (displayValue <= 0) return '#f8fafc'
  if (metric === 'rate') {
    if (displayValue <= 50) return '#fee391'
    if (displayValue <= 200) return '#fec44f'
    if (displayValue <= 1000) return '#fe9929'
    if (displayValue <= 5000) return '#cc4c02'
    return '#8c2d04'
  }
  if (displayValue <= 5) return '#fee391'
  if (displayValue <= 20) return '#fec44f'
  if (displayValue <= 100) return '#fe9929'
  if (displayValue <= 500) return '#cc4c02'
  return '#8c2d04'
}

export function riskLabel(value: number, metric: 'cases' | 'rate' = 'cases'): string {
  const displayValue = metric === 'cases' ? Math.round(value) : Number(value.toFixed(1))
  if (displayValue <= 0) return 'Sin riesgo'
  if (metric === 'rate') {
    if (displayValue <= 50) return 'Bajo'
    if (displayValue <= 200) return 'Moderado'
    if (displayValue <= 1000) return 'Alto'
    return 'Muy alto'
  }
  if (displayValue <= 5) return 'Bajo'
  if (displayValue <= 20) return 'Moderado'
  if (displayValue <= 100) return 'Alto'
  return 'Muy alto'
}
