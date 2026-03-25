export default function StormInfoPanel({ storm }) {
  const last = storm.track?.[storm.track.length - 1]

  return (
    <div className="storm-info">
      <h2>{storm.name} ({storm.season})</h2>
      <div className="info-row">
        <span className="label">SID</span>
        <span className="value">{storm.sid}</span>
      </div>
      <div className="info-row">
        <span className="label">Basin</span>
        <span className="value">{storm.basin}</span>
      </div>
      {last && (
        <>
          <div className="info-row">
            <span className="label">Vị trí cuối</span>
            <span className="value">{last.lat.toFixed(1)}°N {last.lon.toFixed(1)}°E</span>
          </div>
          <div className="info-row">
            <span className="label">Gió cực đại</span>
            <span className="value">{last.vmax ?? '—'} kt</span>
          </div>
          <div className="info-row">
            <span className="label">Áp suất</span>
            <span className="value">{last.pmin ?? '—'} hPa</span>
          </div>
          <div className="info-row">
            <span className="label">Thời gian</span>
            <span className="value">{last.time}</span>
          </div>
        </>
      )}
      {storm.forecast && (
        <>
          <div className="info-row" style={{ marginTop: 12 }}>
            <span className="label">Dự đoán +24h</span>
            <span className="value">
              {storm.forecast.points[0]?.lat.toFixed(1)}°N{' '}
              {storm.forecast.points[0]?.lon.toFixed(1)}°E
            </span>
          </div>
          <div className="info-row">
            <span className="label">Dự đoán +48h</span>
            <span className="value">
              {storm.forecast.points[1]?.lat.toFixed(1)}°N{' '}
              {storm.forecast.points[1]?.lon.toFixed(1)}°E
            </span>
          </div>
        </>
      )}
    </div>
  )
}
