import { useState, useEffect, useRef } from 'react'
import GlobeMap from './components/GlobeMap'
import MapView2D from './components/MapView2D'
import StormInfoPanel from './components/StormInfoPanel'
import IntensityChart from './components/IntensityChart'
import ModelMetricsTable from './components/ModelMetricsTable'
import StormSelector from './components/StormSelector'
import TrackAnimator from './components/TrackAnimator'
import ModelDashboard from './components/ModelDashboard'
import { fetchStorms, fetchStormDetail } from './api/storms'
import { MOCK_STORMS } from './api/mockData'
import { useLocale } from './i18n/LocaleContext'
import { useTheme } from './theme/ThemeContext'

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
  const [mainView, setMainView]           = useState('demo') // 'demo' | 'dashboard'
  const { locale, toggle, t }             = useLocale()
  const { theme, toggle: toggleTheme }    = useTheme()

  // Animation state
  const [animStep,  setAnimStep]  = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [animSpeed, setAnimSpeed] = useState(600)
  const intervalRef = useRef(null)

  // Reset animation khi đổi bão
  useEffect(() => {
    setAnimStep(null)
    setIsPlaying(false)
  }, [selectedStorm])

  // Animation loop — predicted đã interpolate 6h nên x4 số điểm
  useEffect(() => {
    clearInterval(intervalRef.current)
    if (!isPlaying || !selectedStorm) return
    const total = (selectedStorm.track?.length ?? 0) + (selectedStorm.predicted_track?.length ?? 0) * 4
    intervalRef.current = setInterval(() => {
      setAnimStep(prev => {
        const next = (prev ?? -1) + 1
        if (next >= total) { setIsPlaying(false); return total - 1 }
        return next
      })
    }, animSpeed)
    return () => clearInterval(intervalRef.current)
  }, [isPlaying, animSpeed, selectedStorm])

  function handlePlay() {
    if (!selectedStorm) return
    const total = (selectedStorm.track?.length ?? 0) + (selectedStorm.predicted_track?.length ?? 0) * 4
    if (!isPlaying && animStep !== null && animStep >= total - 1) {
      setAnimStep(0)
    }
    setIsPlaying(p => !p)
  }

  function handleReset() {
    setIsPlaying(false)
    setAnimStep(null)
  }

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

  // ─── View Dashboard: layout riêng full-width ───
  if (mainView === 'dashboard') {
    return (
      <div className="app-container app-dashboard-mode">
        <div className="topbar">
          <div className="topbar-title">
            <h1>{t.appTitle}</h1>
          </div>
          <div className="topbar-tabs">
            <button className="tab-btn" onClick={() => setMainView('demo')}>
              🌐 Demo dự báo
            </button>
            <button className="tab-btn tab-active" onClick={() => setMainView('dashboard')}>
              📊 Model Dashboard
            </button>
            <button className="theme-toggle" onClick={toggleTheme}
                    title={theme === 'dark' ? 'Light' : 'Dark'}>
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
        <main className="dashboard-main">
          <ModelDashboard />
        </main>
      </div>
    )
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
                className="tab-btn-mini"
                onClick={() => setMainView('dashboard')}
                title="Xem Model Dashboard"
              >
                📊
              </button>
              <button
                className="view-toggle"
                onClick={() => setViewMode(v => v === '3d' ? '2d' : '3d')}
                title={t.viewToggleTitle}
              >
                {viewMode === '3d' ? '2D' : '3D'}
              </button>
              <button
                className="theme-toggle"
                onClick={toggleTheme}
                title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {theme === 'dark' ? '☀️' : '🌙'}
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
          <>
            <StormInfoPanel storm={selectedStorm} />
            <IntensityChart storm={selectedStorm} />
          </>
        )}

        <ModelMetricsTable />

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
            animStep={animStep}
          />
        ) : (
          <MapView2D
            selectedStorm={selectedStorm}
            showActual={showActual}
            showPredicted={showPredicted}
            animStep={animStep}
          />
        )}

        {selectedStorm && (
          <TrackAnimator
            storm={selectedStorm}
            animStep={animStep}
            isPlaying={isPlaying}
            speed={animSpeed}
            onStep={s => { setIsPlaying(false); setAnimStep(s) }}
            onPlay={handlePlay}
            onReset={handleReset}
            onSpeed={setAnimSpeed}
          />
        )}
      </main>
    </div>
  )
}
