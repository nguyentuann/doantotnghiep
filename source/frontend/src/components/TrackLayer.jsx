import { Polyline, CircleMarker, Tooltip } from 'react-leaflet'

// Color by wind speed (kt): tropical storm → cat 1–5
function intensityColor(vmax) {
  if (!vmax) return '#94a3b8'
  if (vmax < 34) return '#94a3b8'   // tropical depression
  if (vmax < 64) return '#3b82f6'   // tropical storm
  if (vmax < 83) return '#22c55e'   // cat 1
  if (vmax < 96) return '#eab308'   // cat 2
  if (vmax < 113) return '#f97316'  // cat 3
  if (vmax < 137) return '#ef4444'  // cat 4
  return '#a855f7'                   // cat 5
}

export default function TrackLayer({ track }) {
  if (!track || track.length < 2) return null

  const positions = track.map((p) => [p.lat, p.lon])

  return (
    <>
      <Polyline positions={positions} color="#3b82f6" weight={2} opacity={0.7} />
      {track.map((point, i) => (
        <CircleMarker
          key={i}
          center={[point.lat, point.lon]}
          radius={4}
          pathOptions={{
            fillColor: intensityColor(point.vmax),
            color: '#fff',
            weight: 1,
            fillOpacity: 0.9,
          }}
        >
          <Tooltip>
            <strong>{point.time}</strong>
            <br />
            Vị trí: {point.lat.toFixed(1)}°N, {point.lon.toFixed(1)}°E
            <br />
            Gió: {point.vmax ?? '—'} kt | Áp: {point.pmin ?? '—'} hPa
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  )
}
