import { Polyline, CircleMarker, Tooltip } from 'react-leaflet'

// forecast: { points: [{lat, lon, hour, mae_km}], origin: {lat, lon} }
export default function ForecastLayer({ forecast }) {
  if (!forecast?.points?.length) return null

  const origin = forecast.origin
  const allPoints = [origin, ...forecast.points]
  const positions = allPoints.map((p) => [p.lat, p.lon])

  return (
    <>
      <Polyline
        positions={positions}
        color="#f59e0b"
        weight={2}
        dashArray="8 6"
        opacity={0.9}
      />
      {forecast.points.map((pt, i) => (
        <CircleMarker
          key={i}
          center={[pt.lat, pt.lon]}
          // uncertainty circle radius proportional to MAE
          radius={pt.mae_km ? Math.min(pt.mae_km / 20, 20) : 6}
          pathOptions={{
            fillColor: '#f59e0b',
            color: '#f59e0b',
            weight: 1,
            fillOpacity: 0.2,
          }}
        >
          <Tooltip>
            <strong>+{pt.hour}h forecast</strong>
            <br />
            {pt.lat.toFixed(1)}°N, {pt.lon.toFixed(1)}°E
            {pt.mae_km && <><br />MAE: ~{pt.mae_km} km</>}
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  )
}
