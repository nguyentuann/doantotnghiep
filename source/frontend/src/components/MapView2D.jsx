import { useEffect, useMemo } from 'react'
import { MapContainer, TileLayer, Polyline, CircleMarker, Tooltip, useMap } from 'react-leaflet'
import { useLocale } from '../i18n/LocaleContext'
import 'leaflet/dist/leaflet.css'

const SCS_CENTER = [15, 114]
const SCS_ZOOM = 5

function bearingDeg(lat1, lon1, lat2, lon2) {
  const toR = d => d * Math.PI / 180
  const dLon = toR(lon2 - lon1)
  const y = Math.sin(dLon) * Math.cos(toR(lat2))
  const x = Math.cos(toR(lat1)) * Math.sin(toR(lat2)) - Math.sin(toR(lat1)) * Math.cos(toR(lat2)) * Math.cos(dLon)
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360
}

function compassDir(deg) {
  const dirs = ['Bắc','Đông Bắc','Đông','Đông Nam','Nam','Tây Nam','Tây','Tây Bắc']
  return dirs[Math.round(deg / 45) % 8]
}

function withDirection(points) {
  return points.map((p, i) => {
    if (i === 0) return { ...p, direction: null }
    const prev = points[i - 1]
    return { ...p, direction: bearingDeg(prev.lat, prev.lon, p.lat, p.lon) }
  })
}

function intensityColor(vmax) {
  if (!vmax || vmax < 34) return '#94a3b8'
  if (vmax < 64)          return '#60a5fa'
  if (vmax < 83)          return '#34d399'
  if (vmax < 96)          return '#fbbf24'
  if (vmax < 113)         return '#f97316'
  if (vmax < 137)         return '#ef4444'
  return '#a855f7'
}

/** Fly camera to storm or reset to SCS */
function MapUpdater({ selectedStorm }) {
  const map = useMap()
  useEffect(() => {
    if (selectedStorm?.track?.length > 0) {
      const last = selectedStorm.track[selectedStorm.track.length - 1]
      map.flyTo([last.lat, last.lon], 6, { duration: 1.2 })
    } else {
      map.flyTo(SCS_CENTER, SCS_ZOOM, { duration: 1 })
    }
  }, [selectedStorm, map])
  return null
}

/** Actual track: colored polyline segments + circle markers with tooltips */
function ActualTrackLayer({ track }) {
  const { t } = useLocale()
  if (!track || track.length < 2) return null

  const pts = withDirection(track)
  const segments = []
  for (let i = 1; i < pts.length; i++) {
    segments.push({
      positions: [[pts[i-1].lat, pts[i-1].lon], [pts[i].lat, pts[i].lon]],
      color: intensityColor(pts[i].vmax),
    })
  }

  return (
    <>
      {segments.map((seg, i) => (
        <Polyline key={`seg-${i}`} positions={seg.positions} color={seg.color} weight={3} opacity={0.85} />
      ))}
      {pts.map((pt, i) => (
        <CircleMarker
          key={`pt-${i}`}
          center={[pt.lat, pt.lon]}
          radius={4}
          pathOptions={{ fillColor: intensityColor(pt.vmax), color: '#fff', weight: 1.5, fillOpacity: 0.9 }}
        >
          <Tooltip>
            <div style={{ fontSize: 12, lineHeight: 1.7 }}>
              <b style={{ color: '#e94560' }}>📍 {pt.time ?? pt.iso_time}</b><br />
              {pt.lat.toFixed(2)}°N &nbsp; {pt.lon.toFixed(2)}°E<br />
              {t.tooltip.wind} <b>{pt.vmax ?? '--'} kt</b> &nbsp;|&nbsp; {t.tooltip.pres} <b>{pt.pmin ?? '--'} hPa</b><br />
              {pt.direction != null && <>↗ {compassDir(pt.direction)} ({Math.round(pt.direction)}°)</>}
            </div>
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  )
}

/** Predicted track: dashed orange polyline + markers with tooltips */
function PredictedTrackLayer({ predictedTrack, cutoffPoint }) {
  if (!predictedTrack || predictedTrack.length < 1) return null

  const allPoints = cutoffPoint ? [cutoffPoint, ...predictedTrack] : predictedTrack
  const positions = allPoints.map(p => [p.lat, p.lon])
  const pts       = withDirection(allPoints)

  return (
    <>
      <Polyline positions={positions} color="#f59e0b" weight={3} dashArray="10 6" opacity={0.9} />
      {pts.slice(1).map((pt, i) => (
        <CircleMarker
          key={`pred-${i}`}
          center={[pt.lat, pt.lon]}
          radius={5}
          pathOptions={{ fillColor: '#f59e0b', color: '#fff', weight: 1.5, fillOpacity: 0.8 }}
        >
          <Tooltip>
            <div style={{ fontSize: 12, lineHeight: 1.7 }}>
              <b style={{ color: '#f59e0b' }}>🔮 +{(i + 1) * 24}h dự đoán</b><br />
              {pt.time ?? pt.iso_time ?? ''}<br />
              {pt.lat.toFixed(2)}°N &nbsp; {pt.lon.toFixed(2)}°E<br />
              {pt.direction != null && <>↗ {compassDir(pt.direction)} ({Math.round(pt.direction)}°)</>}
            </div>
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  )
}

/** SCS entry point: pulsing circle marker */
function ScsEntryMarker({ point }) {
  if (!point) return null
  return (
    <CircleMarker
      center={[point.lat, point.lon]}
      radius={10}
      pathOptions={{
        fillColor: 'transparent',
        color: '#ffffff',
        weight: 2.5,
        opacity: 0.8,
        fillOpacity: 0,
      }}
    >
      <Tooltip>
        <b>SCS Entry Point</b><br />
        {point.lat.toFixed(1)}°N, {point.lon.toFixed(1)}°E
      </Tooltip>
    </CircleMarker>
  )
}

// Cắt track tại điểm đổ bộ (lon < 108.5°E — bờ đông VN/Hải Nam)
function sliceAtLandfall(points) {
  const idx = points.findIndex((p, i) => i > 0 && p.lon < 108.5)
  return idx === -1 ? points : points.slice(0, idx + 1)  // +1 để hiện điểm đổ bộ
}

export default function MapView2D({ selectedStorm, showActual = true, showPredicted = true }) {
  const cutoff        = selectedStorm?.cutoff_index ?? selectedStorm?.track?.length ?? 0
  const cutoffPoint   = selectedStorm?.track?.[cutoff - 1] ?? selectedStorm?.track?.at(-1) ?? null
  const scsEntryPoint = selectedStorm?.track?.[cutoff] ?? null

  // Actual track: toàn bộ (kể cả SCS) để so sánh với dự đoán
  const actualTrack = selectedStorm?.track ?? []

  // Predicted track: từ cutoff, cắt tại bờ biển Việt Nam
  const rawPredicted = selectedStorm?.predicted_track ?? []
  const predictedTrack = useMemo(
    () => sliceAtLandfall(rawPredicted),
    [rawPredicted]
  )

  return (
    <MapContainer
      center={SCS_CENTER}
      zoom={SCS_ZOOM}
      style={{ height: '100%', width: '100%' }}
      zoomControl={true}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
      />

      {selectedStorm && <MapUpdater selectedStorm={selectedStorm} />}

      {selectedStorm && showActual && (
        <>
          <ActualTrackLayer track={actualTrack} />
          <ScsEntryMarker point={scsEntryPoint} />
        </>
      )}

      {selectedStorm && showPredicted && predictedTrack.length > 0 && (
        <PredictedTrackLayer predictedTrack={predictedTrack} cutoffPoint={cutoffPoint} />
      )}
    </MapContainer>
  )
}
