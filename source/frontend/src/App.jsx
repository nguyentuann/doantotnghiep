import { useState, useEffect } from 'react'
import GlobeMap from './components/GlobeMap'
import StormInfoPanel from './components/StormInfoPanel'
import StormSelector from './components/StormSelector'
import { fetchStorms } from './api/storms'
import { MOCK_STORMS } from './api/mockData'
import { useLocale } from './i18n/LocaleContext'

const INTENSITY_KEYS = ['td', 'ts', 'c1', 'c2', 'c3', 'c4', 'c5']
const INTENSITY_COLORS = ['#94a3b8', '#60a5fa', '#34d399', '#fbbf24', '#f97316', '#ef4444', '#a855f7']

export default function App() {
  const [storms, setStorms]               = useState([])
  const [selectedStorm, setSelectedStorm] = useState(null)
  const [loading, setLoading]             = useState(true)
  const [usingMock, setUsingMock]         = useState(false)
  const { locale, toggle, t }             = useLocale()

  useEffect(() => {
    fetchStorms()
      .then(setStorms)
      .catch(() => {
        setStorms(MOCK_STORMS)
        setUsingMock(true)
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="header-top">
            <div>
              <h1>{t.appTitle}</h1>
              <p>{t.appSubtitle}</p>
            </div>
            <button className="lang-toggle" onClick={toggle} title="Switch language">
              {locale === 'vi' ? 'EN' : 'VI'}
            </button>
          </div>
          {usingMock && (
            <p className="mock-warning">{t.mockWarning}</p>
          )}
        </div>

        <div className="sidebar-section">
          <label>{t.selectStorm}</label>
          <StormSelector
            storms={storms}
            loading={loading}
            selected={selectedStorm}
            onSelect={setSelectedStorm}
          />
        </div>

        {selectedStorm && (
          <StormInfoPanel storm={selectedStorm} />
        )}

        <div className="legend">
          <h3>{t.legendTitle}</h3>
          {INTENSITY_KEYS.map((key, i) => (
            <div key={key} className="legend-item">
              <div className="legend-color" style={{ background: INTENSITY_COLORS[i] }} />
              <span>{t[key]}</span>
            </div>
          ))}
          <div className="legend-divider" />
          <div className="legend-item">
            <div className="legend-dash" />
            <span>{t.legendForecast}</span>
          </div>
        </div>
      </aside>

      <main className="map-container">
        <GlobeMap selectedStorm={selectedStorm} />
      </main>
    </div>
  )
}
