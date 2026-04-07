import { useState, useEffect } from 'react'
import GlobeMap from './components/GlobeMap'
import MapView2D from './components/MapView2D'
import StormInfoPanel from './components/StormInfoPanel'
import StormSelector from './components/StormSelector'
import { fetchStorms, fetchStormDetail } from './api/storms'
import { MOCK_STORMS } from './api/mockData'
import { useLocale } from './i18n/LocaleContext'

const INTENSITY_KEYS = ['td', 'ts', 'c1', 'c2', 'c3', 'c4', 'c5']
const INTENSITY_COLORS = ['#94a3b8', '#60a5fa', '#34d399', '#fbbf24', '#f97316', '#ef4444', '#a855f7']

export default function App() {
  const [storms, setStorms]               = useState([])
  const [selectedStorm, setSelectedStorm] = useState(null)
  const [loading, setLoading]             = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [usingMock, setUsingMock]         = useState(false)
  const [showActual, setShowActual]       = useState(true)
  const [showPredicted, setShowPredicted] = useState(true)
  const [viewMode, setViewMode]           = useState('3d')  // '3d' | '2d'
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

  async function handleSelect(stormItem) {
    if (!stormItem) { setSelectedStorm(null); return }
    if (usingMock) {
      setSelectedStorm(MOCK_STORMS.find(s => s.sid === stormItem.sid) ?? stormItem)
      return
    }
    setDetailLoading(true)
    fetchStormDetail(stormItem.sid)
      .then(setSelectedStorm)
      .catch(() => setSelectedStorm(stormItem))
      .finally(() => setDetailLoading(false))
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="header-top">
            <div>
              <h1>{t.appTitle}</h1>
              <p>{t.appSubtitle}</p>
            </div>
            <div className="header-buttons">
              <button
                className="view-toggle"
                onClick={() => setViewMode(v => v === '3d' ? '2d' : '3d')}
                title={t.viewToggleTitle}
              >
                {viewMode === '3d' ? '2D' : '3D'}
              </button>
              <button className="lang-toggle" onClick={toggle} title="Switch language">
                {locale === 'vi' ? 'EN' : 'VI'}
              </button>
            </div>
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
            onSelect={handleSelect}
          />
          {detailLoading && <p style={{ color: '#aaa', fontSize: '0.8rem' }}>{t.loading}</p>}
        </div>

        {selectedStorm && (
          <StormInfoPanel storm={selectedStorm} />
        )}

        <div className="legend">
          {/* --- Toggle 2 đường --- */}
          <button
            className={`layer-toggle ${showActual ? 'active' : 'inactive'}`}
            onClick={() => setShowActual(v => !v)}
          >
            <div className="legend-line-solid" style={{ opacity: showActual ? 1 : 0.3 }} />
            <span>{t.legendActual}</span>
            <span className="toggle-badge">{showActual ? t.toggleOn : t.toggleOff}</span>
          </button>

          <button
            className={`layer-toggle ${showPredicted ? 'active' : 'inactive'}`}
            onClick={() => setShowPredicted(v => !v)}
          >
            <div className="legend-line-dash" style={{ opacity: showPredicted ? 1 : 0.3 }} />
            <span>{t.legendPredicted}</span>
            <span className="toggle-badge">{showPredicted ? t.toggleOn : t.toggleOff}</span>
          </button>

          <div className="legend-item" style={{ marginTop: 4 }}>
            <div className="legend-dot-white" />
            <span>{t.legendScsEntry}</span>
          </div>

          <div className="legend-divider" />

          {/* --- Cường độ --- */}
          <h3>{t.legendTitle}</h3>
          {INTENSITY_KEYS.map((key, i) => (
            <div key={key} className="legend-item">
              <div className="legend-color" style={{ background: INTENSITY_COLORS[i] }} />
              <span>{t[key]}</span>
            </div>
          ))}
        </div>
      </aside>

      <main className="map-container">
        {viewMode === '3d' ? (
          <GlobeMap
            selectedStorm={selectedStorm}
            showActual={showActual}
            showPredicted={showPredicted}
          />
        ) : (
          <MapView2D
            selectedStorm={selectedStorm}
            showActual={showActual}
            showPredicted={showPredicted}
          />
        )}
      </main>
    </div>
  )
}
