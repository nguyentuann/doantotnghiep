import { useState, useEffect } from 'react'
import StormMap from './components/StormMap'
import StormInfoPanel from './components/StormInfoPanel'
import StormSelector from './components/StormSelector'
import { fetchStorms } from './api/storms'

export default function App() {
  const [storms, setStorms] = useState([])
  const [selectedStorm, setSelectedStorm] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStorms()
      .then(setStorms)
      .catch(() => setStorms([]))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Typhoon Tracker</h1>
          <p>Dự đoán đường đi bão Biển Đông</p>
        </div>

        <div className="sidebar-section">
          <label>Chọn cơn bão</label>
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
          <h3>Chú thích</h3>
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#3b82f6' }} />
            <span>Track lịch sử</span>
          </div>
          <div className="legend-item">
            <div
              className="legend-color"
              style={{ background: '#f59e0b', borderTop: '2px dashed #f59e0b', height: 0 }}
            />
            <span>Dự đoán AI</span>
          </div>
        </div>
      </aside>

      <main className="map-container">
        <StormMap selectedStorm={selectedStorm} />
      </main>
    </div>
  )
}
