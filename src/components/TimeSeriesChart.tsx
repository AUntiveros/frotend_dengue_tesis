import { useMemo, useState } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Brush, ReferenceLine, ReferenceArea,
} from 'recharts'

interface Props {
  weeks: string[]              // eje completo: observado (desde 2018) + cola de pronóstico
  fechas: string[]             // ISO (para líneas de corte)
  hist: number[]               // casos observados (más corto que weeks → null al final)
  predSeries: number[] | null  // casos predichos (test + pronóstico)
  predOffset: number           // índice en weeks donde empieza la predicción
  forecastStart: number        // índice donde empieza la cola de pronóstico (sin observado)
  corteTrain: string           // ISO
  corteVal: string             // ISO
  modelName: string
}

export default function TimeSeriesChart({
  weeks, fechas, hist, predSeries, predOffset, forecastStart, corteTrain, corteVal, modelName,
}: Props) {
  const data = useMemo(() => weeks.map((w, i) => {
    const j = i - predOffset
    const pred = predSeries && j >= 0 ? predSeries[j] : null
    const p = pred == null || !isFinite(pred) ? null : pred
    return {
      week: w,
      obs: hist[i] ?? null,
      pred: p,
      hi: p == null ? null : p + 1.96 * Math.sqrt(p),
      lo: p == null ? null : Math.max(0, p - 1.96 * Math.sqrt(p)),
    }
  }), [weeks, hist, predSeries, predOffset])

  const idxTrain = useMemo(() => fechas.findIndex(f => f >= corteTrain), [fechas, corteTrain])
  const idxVal = useMemo(() => fechas.findIndex(f => f >= corteVal), [fechas, corteVal])

  // Rango por defecto: desde el inicio del periodo de validación (contexto reciente)
  const defaultStart = Math.max(0, (idxTrain >= 0 ? idxTrain : weeks.length - 156))
  const [range, setRange] = useState<[number, number]>([defaultStart, weeks.length - 1])

  return (
    <div>
      <ResponsiveContainer width="100%" height={210}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="week" tick={{ fontSize: 9, fill: '#94a3b8' }} interval="preserveStartEnd" minTickGap={28} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} tickFormatter={(v) => v.toLocaleString()} />
          <Tooltip
            contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)' }}
            formatter={(v: number, name: string) => {
              if (name === 'obs') return [Math.round(v).toLocaleString(), 'Observado']
              if (name === 'pred') return [Math.round(v).toLocaleString(), 'Predicho']
              return [Math.round(v).toLocaleString(), name]
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} formatter={(v) => v === 'obs' ? 'Observado' : v === 'pred' ? `Predicho (${modelName})` : v} />

          {idxTrain >= 0 && (
            <ReferenceLine x={weeks[idxTrain]} stroke="#cbd5e1" strokeDasharray="3 3"
              label={{ value: 'train', position: 'insideTopLeft', fontSize: 8, fill: '#94a3b8' }} />
          )}
          {idxVal >= 0 && (
            <ReferenceLine x={weeks[idxVal]} stroke="#fbbf24" strokeDasharray="3 3"
              label={{ value: 'test →', position: 'insideTopRight', fontSize: 8, fill: '#d97706' }} />
          )}
          {forecastStart < weeks.length && (
            <ReferenceArea x1={weeks[forecastStart]} x2={weeks[weeks.length - 1]}
              fill="#f97316" fillOpacity={0.08} stroke="none"
              label={{ value: 'pronóstico', position: 'insideTop', fontSize: 8, fill: '#ea580c' }} />
          )}

          <Area type="monotone" dataKey="hi" stroke="transparent" fill="#fed7aa" fillOpacity={0.5} connectNulls legendType="none" name="ic_hi" />
          <Area type="monotone" dataKey="lo" stroke="transparent" fill="white" fillOpacity={1} connectNulls legendType="none" name="ic_lo" />

          <Line type="monotone" dataKey="obs" stroke="#2563eb" strokeWidth={1.8} dot={false} connectNulls name="obs" />
          <Line type="monotone" dataKey="pred" stroke="#f97316" strokeWidth={1.8} strokeDasharray="5 3" dot={false} connectNulls name="pred" />

          <Brush
            dataKey="week"
            height={20}
            stroke="#cbd5e1"
            travellerWidth={8}
            startIndex={range[0]}
            endIndex={range[1]}
            onChange={(r: any) => {
              if (r && typeof r.startIndex === 'number' && typeof r.endIndex === 'number') {
                setRange([r.startIndex, r.endIndex])
              }
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="text-[10px] text-slate-400 mt-1">
        Observado desde {weeks[0]} (azul) · Predicho out-of-sample post-validación (naranja, {modelName}) · Arrastra el control inferior para acotar fechas.
      </p>
    </div>
  )
}
