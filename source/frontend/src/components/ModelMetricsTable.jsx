import { useLocale } from '../i18n/LocaleContext'

// Kết quả thực nghiệm champion scs_v12_lb6 — SCS-only test 2021–2024 (279 sequences)
const METRICS = [
  { tag: 'scs_v12_lb6', model: 'Transformer', mae24: 101.3, skill24: 42.0, mae48: 249.1, best: true  },
  { tag: 'scs_v12_lb6', model: 'BiGRU+Attn',  mae24: 118.9, skill24: 31.9, mae48: 337.3, best: false },
  { tag: 'scs_v12_lb6', model: 'BiLSTM+Attn', mae24: 134.2, skill24: 23.1, mae48: 359.3, best: false },
  { tag: 'scs_v12_lb6', model: 'LSTM',         mae24: 134.0, skill24: 23.2, mae48: 362.2, best: false },
  { tag: 'CLIPER',       model: 'Baseline',     mae24: 174.6, skill24: 0.0,  mae48: 431.5, best: false },
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
