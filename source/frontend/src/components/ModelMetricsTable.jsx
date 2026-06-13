import { useLocale } from '../i18n/LocaleContext'

// Kết quả thực nghiệm — cập nhật 2026-04-15
const METRICS = [
  { tag: 'wp_full',      model: 'Ensemble',    mae24: 63.4, skill24: 12.7, mae48: 138.6, best: true  },
  { tag: 'wp_full',      model: 'BiLSTM',      mae24: 63.5, skill24: 12.5, mae48: 139.5, best: false },
  { tag: '12feat_clean', model: 'Ensemble',    mae24: 63.9, skill24: 12.0, mae48: 139.1, best: false },
  { tag: 'paper5feat',   model: 'Ensemble',    mae24: 63.9, skill24: 12.0, mae48: 140.0, best: false },
  { tag: '14feat',       model: 'Ensemble',    mae24: 65.1, skill24: 10.4, mae48: 141.8, best: false },
  { tag: 'CLIPER',       model: 'Baseline',    mae24: 72.6, skill24: 0.0,  mae48: 167.6, best: false },
]

export default function ModelMetricsTable() {
  const { t } = useLocale()
  const m = t.metrics

  return (
    <div className="metrics-container">
      <p className="chart-title">{m.title}</p>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>{m.model}</th>
            <th>{m.mae24}</th>
            <th>{m.skill24}</th>
            <th>{m.mae48}</th>
          </tr>
        </thead>
        <tbody>
          {METRICS.map((row, i) => (
            <tr key={i} className={row.best ? 'row-best' : row.tag === 'CLIPER' ? 'row-cliper' : ''}>
              <td>
                <span className="tag-badge">{row.tag}</span>
                <span className="model-name">{row.model}</span>
              </td>
              <td>{row.mae24} km</td>
              <td className={row.skill24 > 0 ? 'skill-positive' : 'skill-zero'}>
                {row.skill24.toFixed(1)}%
              </td>
              <td>{row.mae48} km</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="metrics-note">{m.note}</p>
    </div>
  )
}
