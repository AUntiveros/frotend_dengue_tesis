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

  useEffect(() => {
    loadDashboard().then(setData).catch(e => setError(String(e)))
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
          <span className="text-xs font-semibold text-slate-500 border border-slate-200 px-2 py-0.5 rounded-md">PUCP</span>
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
          />
        </div>
      </main>
    </div>
  )
}
