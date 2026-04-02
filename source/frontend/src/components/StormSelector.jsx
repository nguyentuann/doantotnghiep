import { useLocale } from '../i18n/LocaleContext'

export default function StormSelector({ storms, loading, selected, onSelect }) {
  const { t } = useLocale()

  if (loading) return <p style={{ color: '#aaa', fontSize: '0.8rem' }}>{t.loading}</p>
  if (!storms.length) return <p style={{ color: '#aaa', fontSize: '0.8rem' }}>{t.noData}</p>

  return (
    <select
      value={selected?.sid ?? ''}
      onChange={e => {
        const storm = storms.find(s => s.sid === e.target.value)
        onSelect(storm ?? null)
      }}
    >
      <option value="">{t.selectPlaceholder}</option>
      {storms.map(s => (
        <option key={s.sid} value={s.sid}>
          {s.name} ({s.season}) — {s.sid}
        </option>
      ))}
    </select>
  )
}
