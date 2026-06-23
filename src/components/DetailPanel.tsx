import { useMemo } from 'react'
import type { Dashboard, ModelInfo, PredEntry } from '../data/predictions'
import { riskColor, riskLabel } from '../data/predictions'
import TimeSeriesChart from './TimeSeriesChart'
import ShapChart from './ShapChart'

interface Props {
  data: Dashboard
  selectedUbigeo: string | null
  activeModel: ModelInfo
}

const LISA_CONFIG: Record<string, { label: string; bg: string; text: string; border: string }> = {
  HH: { label: 'Alto-Alto', bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
  LL: { label: 'Bajo-Bajo', bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
  LH: { label: 'Bajo-Alto', bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  HL: { label: 'Alto-Bajo', bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
  ns: { label: 'No significativo', bg: 'bg-slate-50', text: 'text-slate-500', border: 'border-slate-200' },
}

const DISABLED_HORIZONS = ['H1', 'H8', 'H12', 'H24']

export default function DetailPanel({ data, selectedUbigeo, activeModel }: Props) {
  const view = useMemo(() => buildView(data, selectedUbigeo, activeModel.id), [data, selectedUbigeo, activeModel.id])

  const lisa = LISA_CONFIG[view.lisa] ?? LISA_CONFIG.ns
  const pred = view.pred
  const colorRef = pred ? riskColor(view.isNational ? pred.p / 100 : pred.p) : '#94a3b8'

  return (
    <div className="flex flex-col h-full overflow-y-auto detail-slide-in">

      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-100 flex-shrink-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-900 leading-tight">{view.name}</h2>
            <p className="text-xs text-slate-500 mt-0.5">{view.sub}</p>
          </div>
          {!view.isNational && (
            <span className={`flex-shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full border ${lisa.bg} ${lisa.text} ${lisa.border}`}>
              {view.lisa} &middot; {lisa.label}
            </span>
          )}
        </div>

        {/* H4 result + disabled horizons */}
        <div className="mt-3 flex gap-2 items-stretch">
          <div className="flex-1 rounded-xl border border-blue-500 bg-blue-50 ring-1 ring-blue-500 px-3 py-2 flex flex-col justify-center">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-blue-700">H4 · 4 semanas</span>
              {pred && (
                <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded" style={{ backgroundColor: colorRef + '22', color: colorRef }}>
                  {riskLabel(view.isNational ? pred.p / 100 : pred.p)}
                </span>
              )}
            </div>
            {pred ? (
              <>
                <span className="text-2xl font-bold mt-0.5" style={{ color: colorRef }}>
                  {Math.round(pred.p).toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-400">
                  IC95%: {Math.round(pred.lo)}–{Math.round(pred.hi)} casos
                </span>
              </>
            ) : (
              <span className="text-sm text-slate-400 mt-1">Sin predicción distrital para {activeModel.id}</span>
            )}
          </div>

          <div className="flex flex-col gap-1 justify-center">
            <div className="grid grid-cols-2 gap-1">
              {DISABLED_HORIZONS.map(h => (
                <span
                  key={h}
                  title="Horizonte no entrenado todavía"
                  className="text-[9px] font-semibold text-slate-300 line-through decoration-slate-300 border border-slate-100 rounded-md px-2 py-1 text-center"
                >
                  {h}
                </span>
              ))}
            </div>
            <span className="text-[8px] text-slate-300 text-center">no entrenados</span>
          </div>
        </div>
      </div>

      {/* Time series */}
      <div className="px-5 py-4 border-b border-slate-100 flex-shrink-0">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Serie temporal · observado vs predicho
        </p>
        <TimeSeriesChart
          weeks={[...view.histWeeks, ...data.meta.future_weeks]}
          fechas={[...data.hist_fechas, ...data.meta.future_fechas]}
          hist={view.hist}
          predSeries={pred?.series ?? null}
          predOffset={data.meta.pred_offset}
          forecastStart={data.meta.n_hist_semanas}
          corteTrain={data.meta.corte_train}
          corteVal={data.meta.corte_val}
          modelName={activeModel.name}
        />
      </div>

      {/* SHAP / importancia */}
      <div className="px-5 py-4 border-b border-slate-100 flex-shrink-0">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
          Variables explicativas (importancia)
        </p>
        <p className="text-[10px] text-slate-400 mb-3">
          {activeModel.id === 'M7b'
            ? `Contribución local β·x al log-λ · ${activeModel.id}`
            : activeModel.id === 'STACK'
              ? `Pesos del meta-modelo · ${activeModel.id}`
              : `Importancia global de variables · ${activeModel.id}`}
        </p>
        {pred?.shap
          ? <ShapChart shap={pred.shap} />
          : (
            <div className="flex items-center gap-2 rounded-lg bg-slate-50 border border-slate-100 px-3 py-4">
              <span className="text-[11px] text-slate-400">
                Importancia no disponible para este modelo (SHAP/LIME en proceso).
              </span>
            </div>
          )}
      </div>

      {/* System info */}
      <div className="px-5 py-4 flex-shrink-0">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Estado del sistema</p>
        <div className="space-y-1.5">
          <Row label="Última actualización" value={data.meta.se_label} />
          <Row label="Modelo activo" value={`${activeModel.id} · ${activeModel.name}`} />
          <Row label="Familia" value={activeModel.family} />
          <Row label="R² del modelo" value={activeModel.r2.toFixed(3)} />
          <Row label="RMSE" value={activeModel.rmse.toLocaleString(undefined, { maximumFractionDigits: 2 })} />
          <Row label="MAE" value={activeModel.mae.toLocaleString(undefined, { maximumFractionDigits: 2 })} />
          <Row label={view.isNational ? 'Población total' : 'Ubigeo'} value={view.isNational ? view.pop.toLocaleString() : view.ubigeo} />
        </div>
      </div>
    </div>
  )
}

interface View {
  isNational: boolean
  name: string
  sub: string
  ubigeo: string
  lisa: string
  pop: number
  histWeeks: string[]
  hist: number[]
  pred: PredEntry | null
}

function buildView(data: Dashboard, ubigeo: string | null, modelId: string): View {
  if (ubigeo && data.districts[ubigeo]) {
    const rec = data.districts[ubigeo]
    return {
      isNational: false,
      name: rec.name,
      sub: `${rec.dep} · ${rec.prov}`,
      ubigeo,
      lisa: rec.lisa,
      pop: rec.pop,
      histWeeks: data.hist_weeks,
      hist: rec.hist,
      pred: rec.pred[modelId] ?? null,
    }
  }
  // Vista nacional agregada
  const nac = data.nacional.pred[modelId]
  const totalPop = Object.values(data.districts).reduce((a, d) => a + d.pop, 0)
  return {
    isNational: true,
    name: 'Perú (Agregado Nacional)',
    sub: 'Suma de 1,891 distritos',
    ubigeo: '000000',
    lisa: 'ns',
    pop: totalPop,
    histWeeks: data.hist_weeks,
    hist: data.nacional.hist,
    pred: nac
      ? { p: nac.p, lo: nac.lo?.[nac.lo.length - 1] ?? nac.p * 0.8, hi: nac.hi?.[nac.hi.length - 1] ?? nac.p * 1.2, series: nac.series }
      : null,
  }
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-700 font-medium">{value}</span>
    </div>
  )
}
