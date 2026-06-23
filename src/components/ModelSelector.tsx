import { useState } from 'react'
import type { ModelInfo } from '../data/predictions'

interface Props {
  models: ModelInfo[]
  activeModel: ModelInfo
  onChange: (m: ModelInfo) => void
}

export default function ModelSelector({ models, activeModel, onChange }: Props) {
  const [open, setOpen] = useState(false)

  // Ordenar por R² descendente para la comparativa
  const ordered = [...models].sort((a, b) => b.r2 - a.r2)

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 transition-colors text-sm font-medium text-slate-700 shadow-sm"
      >
        <span className="w-2 h-2 rounded-full bg-violet-500 flex-shrink-0" />
        <span>{activeModel.name}</span>
        <span className="text-slate-400 ml-1 font-mono text-xs">R²={activeModel.r2.toFixed(3)}</span>
        <svg className={`w-4 h-4 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-[460px] max-h-[70vh] overflow-y-auto bg-white border border-slate-200 rounded-xl shadow-xl z-[9999]">
          <div className="px-4 py-3 border-b border-slate-100 bg-slate-50 sticky top-0">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Seleccionar modelo predictivo</p>
            <p className="text-[10px] text-slate-400 mt-0.5">13 modelos · métricas reales (test, H=4)</p>
          </div>
          <div className="divide-y divide-slate-100">
            {ordered.map(m => {
              const active = m.id === activeModel.id
              return (
                <button
                  key={m.id}
                  onClick={() => { onChange(m); setOpen(false) }}
                  className={`w-full text-left px-4 py-3 flex items-start gap-3 transition-colors ${active ? 'bg-violet-50' : 'hover:bg-slate-50'}`}
                >
                  <div className={`mt-0.5 w-4 h-4 rounded-full border-2 flex-shrink-0 flex items-center justify-center ${active ? 'border-violet-500 bg-violet-500' : 'border-slate-300'}`}>
                    {active && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className={`text-sm font-semibold ${active ? 'text-violet-700' : 'text-slate-800'}`}>{m.name}</span>
                      <div className="flex items-center gap-1 flex-shrink-0">
                        {m.map
                          ? <span className="text-[8px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-600">MAPA</span>
                          : m.scope === 'nacional'
                            ? <span className="text-[8px] font-bold px-1.5 py-0.5 rounded bg-sky-100 text-sky-600">NACIONAL</span>
                            : <span className="text-[8px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-400">COMPARATIVA</span>}
                        <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${active ? 'bg-violet-100 text-violet-600' : 'bg-slate-100 text-slate-500'}`}>{m.id}</span>
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5 truncate">{m.description}</p>
                    <div className="flex gap-4 mt-1.5">
                      <Metric label="R²" value={m.r2.toFixed(3)} good={m.r2 >= 0.9} />
                      <Metric label="RMSE" value={m.rmse.toLocaleString(undefined, { maximumFractionDigits: 2 })} good={m.rmse < 6} />
                      <Metric label="MAE" value={m.mae.toLocaleString(undefined, { maximumFractionDigits: 2 })} good={m.mae < 1} />
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {open && <div className="fixed inset-0 z-[9998]" onClick={() => setOpen(false)} />}
    </div>
  )
}

function Metric({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <span className="flex items-center gap-1">
      <span className="text-xs text-slate-400">{label}</span>
      <span className={`text-xs font-semibold font-mono ${good ? 'text-emerald-600' : 'text-amber-600'}`}>{value}</span>
    </span>
  )
}
