import { useEffect, useMemo, useState } from 'react'
import type { Dashboard, ModelInfo } from './data/predictions'
import { loadDashboard } from './data/predictions'
import MapPanel from './components/MapPanel'
import DetailPanel from './components/DetailPanel'
import ModelSelector from './components/ModelSelector'

// Solo H4 tiene modelo entrenado; los demás horizontes quedan deshabilitados.
const HORIZONS = ['H1', 'H4', 'H8', 'H12', 'H24'] as const
const HORIZON_LABELS: Record<string, string> = {
  H1: '1 semana', H4: '4 semanas', H8: '8 semanas', H12: '12 semanas', H24: '24 semanas',
}

export default function App() {
  const [data, setData] = useState<Dashboard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeModelId, setActiveModelId] = useState<string>('M7b')
  const [selectedUbigeo, setSelectedUbigeo] = useState<string | null>(null)
  // Semana reportada en el mapa — controlada por el selector de rango de la serie
  const [obsWeek, setObsWeek] = useState<number>(0)

  useEffect(() => {
    loadDashboard().then(d => {
      setData(d)
      setObsWeek(d.meta.n_hist_semanas - 1)   // por defecto: última semana observada
    }).catch(e => setError(String(e)))
  }, [])

  const activeModel: ModelInfo | undefined = useMemo(
    () => data?.models.find(m => m.id === activeModelId),
    [data, activeModelId],
  )

  if (error) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-50 p-8">
        <div className="max-w-md text-center">
          <p className="text-sm font-semibold text-red-600">Error al cargar predicciones</p>
          <p className="mt-2 text-xs text-slate-500">{error}</p>
          <p className="mt-3 text-xs text-slate-400">
            Ejecuta <code className="rounded bg-slate-100 px-1">python backend/generador_dashboard.py</code> para
            generar <code className="rounded bg-slate-100 px-1">public/data/predicciones_semana.json</code>.
          </p>
        </div>
      </div>
    )
  }

  if (!data || !activeModel) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 bg-slate-50">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-200 border-t-blue-500" />
        <p className="text-sm text-slate-500">Cargando predicciones...</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-slate-50">

      {/* Header */}
      <header className="flex items-center gap-4 px-5 py-3 bg-white border-b border-slate-200 flex-shrink-0 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 flex items-center justify-center flex-shrink-0">
            <img src="/pucp.png" alt="PUCP Logo" className="w-full h-full object-contain" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-900 leading-tight">
              Sistema de Vigilancia Epidemiologica
            </h1>
            <p className="text-xs text-slate-400 leading-tight">
              Dengue Peru · {data.meta.n_distritos.toLocaleString()} distritos
            </p>
          </div>
        </div>

        <div className="h-6 w-px bg-slate-200 mx-1" />

        {/* Horizon selector — solo H4 entrenado */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
          {HORIZONS.map(h => {
            const enabled = h === 'H4'
            return (
              <button
                key={h}
                disabled={!enabled}
                title={enabled ? `Horizonte ${HORIZON_LABELS[h]}` : 'Horizonte no entrenado todavía'}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  enabled
                    ? 'bg-white text-blue-700 shadow-sm'
                    : 'text-slate-300 cursor-not-allowed line-through decoration-slate-300'
                }`}
              >
                {h}
                {enabled && (
                  <span className="hidden sm:inline ml-1 font-normal text-slate-400">
                    · {HORIZON_LABELS[h]}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-slate-400">{data.meta.se_label}</span>
          </div>
          <div className="h-6 w-px bg-slate-200" />
          
          <div className="relative group flex items-center">
            <span className="text-xs font-semibold text-slate-500 border border-slate-200 px-2 py-0.5 rounded-md cursor-help flex items-center gap-1 hover:bg-slate-50 transition-colors">
              PUCP
              <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            {/* Tooltip con créditos */}
            <div className="absolute right-0 top-full mt-2 w-64 bg-white text-slate-700 text-xs rounded-xl p-3 shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[9999] pointer-events-none border border-slate-200">
              <p className="font-bold text-sm mb-1 text-blue-700">Proyecto de Tesis</p>
              <p className="text-slate-500 mb-2">Ingeniería Biomédica</p>
              <div className="h-px w-full bg-slate-200 my-2" />
              <p className="font-medium text-slate-800">Alejandro Alvaro Untiveros Parra</p>
              <p className="text-slate-500 mt-1">Asesora: Dra. Rene Flores Clavo</p>
              <p className="text-slate-400 mt-1">2026</p>
            </div>
          </div>

          <ModelSelector
            models={data.models}
            activeModel={activeModel}
            onChange={m => setActiveModelId(m.id)}
          />
        </div>
      </header>

      {/* Main */}
      <main className="flex flex-1 overflow-hidden">
        {/* Map panel — 60% */}
        <div className="relative flex-[60]">
          <MapPanel
            data={data}
            activeModelId={activeModel.map ? activeModelId : 'M7b'}
            modelHasMap={activeModel.map}
            obsWeek={obsWeek}
            onDistrictClick={setSelectedUbigeo}
            selectedUbigeo={selectedUbigeo}
          />
        </div>

        {/* Divider */}
        <div className="w-px bg-slate-200 flex-shrink-0" />

        {/* Detail panel — 40% */}
        <div className="flex-[40] bg-white overflow-hidden flex flex-col">
          <DetailPanel
            data={data}
            selectedUbigeo={selectedUbigeo}
            activeModel={activeModel}
            onWeekChange={setObsWeek}
          />
        </div>
      </main>
    </div>
  )
}
