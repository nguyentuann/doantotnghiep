import { useLocale } from '../i18n/LocaleContext'

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  )
}

export default function StormInfoPanel({ storm }) {
  const { t } = useLocale()
  const si   = t.stormInfo
  const last = storm.track?.[storm.track.length - 1]
  const f24  = storm.forecast?.points[0]
  const f48  = storm.forecast?.points[1]

  return (
    <div className="storm-info">
      <h2>{storm.name} ({storm.season})</h2>

      <InfoRow label={si.sid}   value={storm.sid} />
      <InfoRow label={si.basin} value={storm.basin} />

      {last && (
        <>
          <InfoRow
            label={si.lastPosition}
            value={`${last.lat.toFixed(1)}°N  ${last.lon.toFixed(1)}°E`}
          />
          <InfoRow label={si.maxWind}  value={`${last.vmax ?? '—'} kt`} />
          <InfoRow label={si.pressure} value={`${last.pmin ?? '—'} hPa`} />
          <InfoRow label={si.time}     value={last.time} />
        </>
      )}

      {f24 && (
        <InfoRow
          label={si.forecast24h}
          value={`${f24.lat.toFixed(1)}°N  ${f24.lon.toFixed(1)}°E`}
        />
      )}
      {f48 && (
        <>
          <InfoRow
            label={si.forecast48h}
            value={`${f48.lat.toFixed(1)}°N  ${f48.lon.toFixed(1)}°E`}
          />
          <InfoRow
            label={si.mae}
            value={`±${f24?.mae_km ?? '—'} km / ±${f48?.mae_km ?? '—'} km`}
          />
        </>
      )}
    </div>
  )
}
