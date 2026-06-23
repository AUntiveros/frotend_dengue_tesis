import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts'

interface Props {
  shap: [string, number][]
}

export default function ShapChart({ shap }: Props) {
  const data = shap.map(([name, value]) => ({ name, value }))

  return (
    <ResponsiveContainer width="100%" height={Math.max(110, data.length * 22)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
        <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#475569' }} tickLine={false} axisLine={false} width={110} />
        <Tooltip
          contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }}
          formatter={(v: number) => [v.toFixed(2), 'Importancia']}
        />
        <ReferenceLine x={0} stroke="#cbd5e1" strokeWidth={1} />
        <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={14}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.value >= 0 ? '#2563eb' : '#ef4444'} fillOpacity={0.75} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
