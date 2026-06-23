import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import * as topojson from 'topojson-client'
import type { Topology } from 'topojson-specification'
import type { Dashboard } from '../data/predictions'
import { riskColor } from '../data/predictions'

type MapMode = 'full' | 'department'
type MapMetric = 'cases' | 'rate'
type MapSource = 'pred' | 'obs'    // predicción H4 vs casos reportados (semana ref)

interface Props {
  data: Dashboard
  activeModelId: string         // modelo distrital usado para pintar el mapa
  modelHasMap: boolean          // false → el modelo activo no es distrital
  obsWeek: number               // semana reportada (control del selector de la serie)
  onDistrictClick: (ubigeo: string | null) => void
  selectedUbigeo: string | null
}

const PERU_CENTER: [number, number] = [-9.2, -74.5]
const PERU_ZOOM = 5

export default function MapPanel({ data, activeModelId, modelHasMap, obsWeek, onDistrictClick, selectedUbigeo }: Props) {
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const districtsLayerRef = useRef<L.GeoJSON | null>(null)
  const deptBordersLayerRef = useRef<L.GeoJSON | null>(null)
  const deptLabelsLayerRef = useRef<L.LayerGroup | null>(null)
  const selectedDeptLayerRef = useRef<L.GeoJSON | null>(null)

  const [mapMode, setMapMode] = useState<MapMode>('full')
  const [mapMetric, setMapMetric] = useState<MapMetric>('cases')
  const [mapSource, setMapSource] = useState<MapSource>('pred')
  const [selectedDeptCcdd, setSelectedDeptCcdd] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const dataRef = useRef(data)
  const modelRef = useRef(activeModelId)
  const selectedUbigeoRef = useRef(selectedUbigeo)
  const mapModeRef = useRef(mapMode)
  const mapMetricRef = useRef(mapMetric)
  const mapSourceRef = useRef(mapSource)
  const obsWeekRef = useRef(obsWeek)
  const selectedDeptCcddRef = useRef(selectedDeptCcdd)

  useEffect(() => { dataRef.current = data }, [data])
  useEffect(() => { modelRef.current = activeModelId }, [activeModelId])
  useEffect(() => { selectedUbigeoRef.current = selectedUbigeo }, [selectedUbigeo])
  useEffect(() => { mapModeRef.current = mapMode }, [mapMode])
  useEffect(() => { mapMetricRef.current = mapMetric }, [mapMetric])
  useEffect(() => { mapSourceRef.current = mapSource }, [mapSource])
  useEffect(() => { obsWeekRef.current = obsWeek }, [obsWeek])
  useEffect(() => { selectedDeptCcddRef.current = selectedDeptCcdd }, [selectedDeptCcdd])

  // valor base de un distrito: predicción H4 o casos reportados en la semana `wk`
  function districtCases(ubigeo: string, modelId: string, source: MapSource, wk: number): number | null {
    const rec = dataRef.current.districts[ubigeo]
    if (!rec) return null
    if (source === 'obs') return rec.hist.length ? (rec.hist[wk] ?? null) : null
    const pred = rec.pred[modelId]
    return pred ? pred.p : null
  }

  // value (casos o tasa) de un distrito bajo el modelo/fuente activos
  function districtValue(ubigeo: string, modelId: string, metric: MapMetric, source: MapSource, wk: number): number | null {
    const rec = dataRef.current.districts[ubigeo]
    const c = districtCases(ubigeo, modelId, source, wk)
    if (!rec || c == null) return null
    return metric === 'cases' ? c : (c / Math.max(rec.pop, 1)) * 100000
  }

  // Init map once
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    const map = L.map(mapContainerRef.current, {
      center: PERU_CENTER,
      zoom: PERU_ZOOM,
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: true,
    })

    L.control.zoom({ position: 'bottomright' }).addTo(map)
    const container = map.getContainer()
    container.addEventListener('wheel', (e) => e.stopPropagation(), { passive: false })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
    }).addTo(map)

    L.control.attribution({ prefix: false })
      .addAttribution('INEI CPV 2025 · CartoDB')
      .addTo(map)

    mapRef.current = map

    Promise.all([
      fetch('/data/distritos.topojson').then(r => r.json()),
      fetch('/data/departamentos.geojson').then(r => r.json()),
    ]).then(([topoData, deptos]) => {
      const objectName = Object.keys((topoData as Topology).objects)[0]
      const distritos = topojson.feature(topoData as Topology, (topoData as Topology).objects[objectName]) as unknown as GeoJSON.FeatureCollection
      const map = mapRef.current!

      const deptBorders = L.geoJSON(deptos, {
        style: {
          fillColor: 'transparent', fillOpacity: 0,
          color: '#64748b', weight: 1.8, opacity: 0.9,
        },
        onEachFeature: (feature, layer) => {
          layer.bindTooltip(feature.properties.DEPARTAMEN, {
            permanent: false, direction: 'center', className: 'map-tooltip',
          })
        },
      }).addTo(map)
      deptBordersLayerRef.current = deptBorders

      // Etiquetas con el nombre de cada departamento (centroide), solo vista completa
      const labels = L.layerGroup()
      deptBorders.eachLayer((l) => {
        const f = (l as L.GeoJSON).feature as GeoJSON.Feature
        const name = (f?.properties as Record<string, string>)?.DEPARTAMEN
        const b = (l as unknown as { getBounds?: () => L.LatLngBounds }).getBounds?.()
        if (!name || !b) return
        L.marker(b.getCenter(), {
          icon: L.divIcon({ className: 'dept-label', html: name, iconSize: [0, 0] }),
          interactive: false,
          keyboard: false,
        }).addTo(labels)
      })
      labels.addTo(map)
      deptLabelsLayerRef.current = labels

      const districtsLayer = L.geoJSON(distritos, {
        style: (feature) => getDistrictStyle(feature, dataRef.current, modelRef.current, mapMetricRef.current, mapSourceRef.current, obsWeekRef.current, selectedUbigeoRef.current, mapModeRef.current, selectedDeptCcddRef.current),
        onEachFeature: (feature, layer) => {
          const props = feature.properties as Record<string, string>
          const ubigeo = props.UBIGEO || ''
          const distritoName = props.DISTRITO || ''
          const provName = props.PROVINCIA || ''
          const depName = props.DEPARTAMEN || ''
          const ccdd = props.CCDD || ''

          layer.on('mouseover', function (e: L.LeafletMouseEvent) {
            const m = mapMetricRef.current
            const src = mapSourceRef.current
            const rec = dataRef.current.districts[ubigeo]
            const pred = rec?.pred[modelRef.current]
            const val = districtValue(ubigeo, modelRef.current, m, src, obsWeekRef.current)
            const unit = m === 'cases' ? 'casos' : 'x100k'
            const tag = src === 'obs' ? (dataRef.current.hist_weeks[obsWeekRef.current] ?? 'reportado') : 'H4'
            const displayVal = val == null ? '—' : (m === 'cases' ? Math.round(val).toString() : val.toFixed(1))
            const color = riskColor(val == null ? -1 : (m === 'cases' ? val : val / 10))
            const detalle = src === 'pred' && pred
              ? `<div style="font-size:10px;color:#94a3b8;margin-top:2px">IC95%: [${pred.lo}–${pred.hi}] &middot; Pob: ${(rec?.pop ?? 0).toLocaleString()}</div>`
              : `<div style="font-size:10px;color:#94a3b8;margin-top:2px">Pob: ${(rec?.pop ?? 0).toLocaleString()}</div>`

            const content = `
              <div class="map-tooltip">
                <div style="font-weight:700;font-size:13px;color:#1e293b;margin-bottom:2px">${distritoName}</div>
                <div style="font-size:11px;color:#64748b;margin-bottom:6px">${depName} &middot; ${provName}</div>
                ${val != null
                  ? `<div style="font-size:15px;font-weight:800;color:${color}">${displayVal} <span style="font-size:11px;font-weight:400;color:#94a3b8">${unit} (${tag})</span></div>${detalle}`
                  : `<div style="font-size:11px;color:#94a3b8">Sin predicción para este modelo &middot; ${ubigeo}</div>`
                }
              </div>`

            layer.bindPopup(content, { maxWidth: 240, autoPan: false, closeButton: false }).openPopup(e.latlng)

            if (selectedUbigeoRef.current !== ubigeo) {
              ;(layer as L.Path).setStyle({ weight: 1.5, color: '#475569', fillOpacity: 0.9 })
              ;(layer as L.Path).bringToFront()
            }
          })

          layer.on('mouseout', function () {
            layer.closePopup()
            if (selectedUbigeoRef.current !== ubigeo) districtsLayer.resetStyle(layer)
          })

          layer.on('click', function () {
            const rec = dataRef.current.districts[ubigeo]
            if (!rec) return

            const prevUbigeo = selectedUbigeoRef.current
            const prevDept = selectedDeptCcddRef.current
            const newDept = ccdd

            if (prevUbigeo) {
              const prevLayer = findLayerByUbigeo(districtsLayer, prevUbigeo)
              if (prevLayer) districtsLayer.resetStyle(prevLayer as L.Path)
            }

            const val = districtValue(ubigeo, modelRef.current, mapMetricRef.current, mapSourceRef.current, obsWeekRef.current)
            ;(layer as L.Path).setStyle({
              color: '#312e81', weight: 3,
              fillColor: riskColor(val == null ? -1 : (mapMetricRef.current === 'cases' ? val : val / 10)),
              fillOpacity: 0.95,
            })
            ;(layer as L.Path).bringToFront()

            onDistrictClick(ubigeo)

            if (newDept !== prevDept || mapModeRef.current === 'full') {
              setSelectedDeptCcdd(newDept)
              setMapMode('department')
              if (deptLabelsLayerRef.current) map.removeLayer(deptLabelsLayerRef.current)
              zoomToDepartment(districtsLayer, deptBorders, newDept, map)
            }
          })
        },
      }).addTo(map)

      districtsLayerRef.current = districtsLayer
      setLoading(false)
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, []) // eslint-disable-line

  // Re-pintar cuando cambian modelo / métrica / fuente / semana / selección
  useEffect(() => {
    const layer = districtsLayerRef.current
    if (!layer) return
    layer.setStyle((feature) =>
      getDistrictStyle(feature, data, activeModelId, mapMetric, mapSource, obsWeek, selectedUbigeo, mapMode, selectedDeptCcdd)
    )
    if (selectedUbigeo) {
      const sel = findLayerByUbigeo(layer, selectedUbigeo)
      if (sel) {
        const val = districtValue(selectedUbigeo, activeModelId, mapMetric, mapSource, obsWeek)
        ;(sel as L.Path).setStyle({
          color: '#312e81', weight: 3,
          fillColor: riskColor(val == null ? -1 : (mapMetric === 'cases' ? val : val / 10)),
          fillOpacity: 0.95,
        })
        ;(sel as L.Path).bringToFront()
      }
    }
  }, [data, activeModelId, mapMetric, mapSource, obsWeek, selectedUbigeo, mapMode, selectedDeptCcdd])

  function handleBackToFull() {
    setMapMode('full')
    setSelectedDeptCcdd(null)
    onDistrictClick(null)

    const map = mapRef.current
    const depts = deptBordersLayerRef.current
    const dists = districtsLayerRef.current
    if (!map || !depts || !dists) return

    map.flyTo(PERU_CENTER, PERU_ZOOM, { animate: true, duration: 0.5 })
    // Reset COMPLETO del borde departamental (color incluido) → todo Perú marcado uniforme
    depts.setStyle({ fillColor: 'transparent', fillOpacity: 0, color: '#64748b', weight: 1.8, opacity: 0.9 })
    dists.setStyle((f) => getDistrictStyle(f, dataRef.current, modelRef.current, mapMetricRef.current, mapSourceRef.current, obsWeekRef.current, null, 'full', null))

    // Restaurar etiquetas de departamentos
    if (deptLabelsLayerRef.current && !map.hasLayer(deptLabelsLayerRef.current)) {
      deptLabelsLayerRef.current.addTo(map)
    }

    if (selectedDeptLayerRef.current) {
      map.removeLayer(selectedDeptLayerRef.current)
      selectedDeptLayerRef.current = null
    }
  }

  return (
    <div className="relative h-full w-full">
      {loading && (
        <div className="absolute inset-0 z-[2000] flex flex-col items-center justify-center bg-slate-50/90 gap-3">
          <div className="w-8 h-8 border-2 border-slate-200 border-t-blue-500 rounded-full animate-spin" />
          <p className="text-sm text-slate-500">Cargando {data.meta.n_distritos.toLocaleString()} distritos...</p>
        </div>
      )}

      {!modelHasMap && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1100] flex items-center gap-2 px-3 py-1.5 bg-amber-50 border border-amber-200 rounded-lg shadow-md">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
          <span className="text-[11px] font-medium text-amber-700">
            Modelo sin desagregación distrital — mapa mostrando M7b (ZIP)
          </span>
        </div>
      )}

      <button
        onClick={handleBackToFull}
        className="absolute top-3 left-3 z-[1000] flex items-center justify-center w-9 h-9 bg-white rounded-lg border border-slate-200 shadow-md text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
        title="Vista completa de Perú"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1" />
        </svg>
      </button>

      {mapMode === 'department' && (
        <button
          onClick={handleBackToFull}
          className="absolute top-3 left-14 z-[1000] flex items-center gap-1.5 px-3 py-2 bg-white rounded-lg border border-slate-200 shadow-md text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M15 19l-7-7 7-7" />
          </svg>
          Perú completo
        </button>
      )}

      {/* Source toggle: Reportado vs Predicho */}
      <div className="absolute top-3 right-3 z-[1000] flex items-center gap-1 p-1 bg-white border border-slate-200 rounded-lg shadow-md">
        <button
          onClick={() => setMapSource('obs')}
          className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-colors ${mapSource === 'obs' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-700'}`}
          title="Casos reportados en la semana de referencia"
        >
          Reportado
        </button>
        <button
          onClick={() => setMapSource('pred')}
          className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-colors ${mapSource === 'pred' ? 'bg-orange-600 text-white' : 'text-slate-500 hover:text-slate-700'}`}
          title="Predicción del modelo a 4 semanas (H4)"
        >
          Predicho (H4)
        </button>
      </div>

      {/* Colorbar & Metric Switch */}
      <div className="absolute bottom-5 left-3 z-[1000] bg-white border border-slate-200 rounded-xl p-3 shadow-md w-56">
        <div className="text-[10px] font-semibold text-slate-500 mb-1.5">
          {mapSource === 'obs' ? `Reportado · ${data.hist_weeks[obsWeek]}` : `Predicción H4 · ${data.meta.se_label}`}
        </div>
        <div className="flex items-center justify-between mb-2.5">
          <button
            onClick={() => setMapMetric('cases')}
            className={`text-[10px] font-semibold transition-colors ${mapMetric === 'cases' ? 'text-blue-700' : 'text-slate-400 hover:text-slate-500'}`}
          >
            Casos / SE
          </button>
          <button
            onClick={() => setMapMetric(m => m === 'cases' ? 'rate' : 'cases')}
            className="relative w-10 h-5 rounded-full transition-colors duration-200 flex-shrink-0"
            style={{ backgroundColor: mapMetric === 'cases' ? '#cbd5e1' : '#3b82f6' }}
          >
            <span
              className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-sm transition-transform duration-200"
              style={{ transform: mapMetric === 'cases' ? 'translateX(2px)' : 'translateX(22px)' }}
            />
          </button>
          <button
            onClick={() => setMapMetric('rate')}
            className={`text-[10px] font-semibold transition-colors ${mapMetric === 'rate' ? 'text-blue-700' : 'text-slate-400 hover:text-slate-500'}`}
          >
            Tasa ×100k
          </button>
        </div>

        <div className="w-full h-2.5 rounded-full mb-1.5 border border-slate-200" style={{ background: 'linear-gradient(to right,#f8fafc,#fee391,#fe9929,#cc4c02,#8c2d04)' }} />
        <div className="flex justify-between text-[10px] text-slate-400">
          {mapMetric === 'cases' ? (
            <><span>0</span><span>5</span><span>20</span><span>100</span><span>500+</span></>
          ) : (
            <><span>0</span><span>50</span><span>200</span><span>1k</span><span>5k+</span></>
          )}
        </div>
      </div>

      <div ref={mapContainerRef} className="w-full h-full" />
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function getDistrictStyle(
  feature: GeoJSON.Feature | undefined,
  data: Dashboard,
  modelId: string,
  metric: MapMetric,
  source: MapSource,
  obsWeek: number,
  selectedUbigeo: string | null,
  mode: MapMode,
  selectedDeptCcdd: string | null,
): L.PathOptions {
  if (!feature) return {}
  const props = feature.properties as Record<string, string>
  const ubigeo = props.UBIGEO || ''
  const ccdd = props.CCDD || ''
  const rec = data.districts[ubigeo]

  let cases: number | null = null
  if (rec) {
    cases = source === 'obs'
      ? (rec.hist.length ? (rec.hist[obsWeek] ?? null) : null)
      : (rec.pred[modelId]?.p ?? null)
  }
  const val = cases == null ? null : (metric === 'cases' ? cases : (cases / Math.max(rec!.pop, 1)) * 100000)
  const color = riskColor(val == null ? -1 : (metric === 'cases' ? val : val / 10))
  const hasData = val != null

  if (ubigeo === selectedUbigeo) {
    return { fillColor: color, fillOpacity: 0.95, color: '#312e81', weight: 3, opacity: 1 }
  }

  if (mode === 'department') {
    if (ccdd !== selectedDeptCcdd) {
      // Máscara gris suave (no blanco): se mantiene el contexto, atenuado
      return { fillColor: '#94a3b8', fillOpacity: 0.35, color: '#cbd5e1', weight: 0.3, opacity: 0.5 }
    }
    return { fillColor: color, fillOpacity: hasData ? 0.85 : 0.15, color: '#64748b', weight: 0.6, opacity: 0.6 }
  }

  return { fillColor: color, fillOpacity: hasData ? 0.85 : 0.15, color: '#cbd5e1', weight: 0.3, opacity: 0.5 }
}

function findLayerByUbigeo(geoLayer: L.GeoJSON, ubigeo: string): L.Layer | null {
  let found: L.Layer | null = null
  geoLayer.eachLayer((layer) => {
    const f = (layer as L.GeoJSON).feature as GeoJSON.Feature
    const featUbigeo = f?.properties?.UBIGEO
    if (featUbigeo === ubigeo) found = layer
  })
  return found
}

function zoomToDepartment(dists: L.GeoJSON, depts: L.GeoJSON, ccdd: string, map: L.Map) {
  const bounds = L.latLngBounds([])
  dists.eachLayer((layer) => {
    const f = (layer as L.GeoJSON).feature as GeoJSON.Feature
    const featCcdd = f?.properties?.CCDD
    if (featCcdd === ccdd) {
      const polyLayer = layer as unknown as { getBounds?: () => L.LatLngBounds }
      const b = polyLayer.getBounds?.()
      if (b) bounds.extend(b)
    }
  })
  if (bounds.isValid()) map.flyToBounds(bounds, { padding: [40, 40], animate: true, duration: 0.5 })

  depts.setStyle((f) => {
    if (!f) return {}
    const featCcdd = (f.properties as Record<string, string>).CCDD
    return {
      fillOpacity: 0,
      color: featCcdd === ccdd ? '#1e40af' : '#e2e8f0',
      weight: featCcdd === ccdd ? 2 : 0.5,
      opacity: featCcdd === ccdd ? 1 : 0.3,
    }
  })
}
