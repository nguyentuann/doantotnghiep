import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import Globe from 'react-globe.gl'
import { useLocale } from '../i18n/LocaleContext'

// Tâm mặc định — Biển Đông
const SCS_VIEW = { lat: 15, lng: 114, altitude: 2.2 }

// Màu theo cường độ gió (Saffir–Simpson)
function intensityColor(vmax) {
  if (!vmax || vmax < 34) return '#94a3b8'  // TD  — xám
  if (vmax < 64)          return '#60a5fa'  // TS  — xanh dương nhạt
  if (vmax < 83)          return '#34d399'  // C1  — xanh lá
  if (vmax < 96)          return '#fbbf24'  // C2  — vàng
  if (vmax < 113)         return '#f97316'  // C3  — cam
  if (vmax < 137)         return '#ef4444'  // C4  — đỏ
  return '#a855f7'                           // C5  — tím
}

export default function GlobeMap({ selectedStorm }) {
  const globeRef     = useRef()
  const containerRef = useRef()
  const [size, setSize] = useState({ width: 800, height: 600 })
  const { t } = useLocale()

  // Theo dõi kích thước container để Globe lấp đầy
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSize({ width, height })
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  // Auto-rotate khi không chọn bão; bay đến bão khi chọn
  useEffect(() => {
    if (!globeRef.current) return
    const controls = globeRef.current.controls()
    if (selectedStorm) {
      controls.autoRotate = false
      const last = selectedStorm.track[selectedStorm.track.length - 1]
      globeRef.current.pointOfView(
        { lat: last.lat, lng: last.lon, altitude: 1.5 },
        1200
      )
    } else {
      controls.autoRotate = true
      controls.autoRotateSpeed = 0.4
      globeRef.current.pointOfView(SCS_VIEW, 1000)
    }
  }, [selectedStorm])

  // --- Paths (track lịch sử + dự báo) ---
  const pathsData = useMemo(() => {
    if (!selectedStorm) return []
    const paths = []

    // Track lịch sử — màu theo cường độ từng điểm
    paths.push({
      id: 'track',
      points: selectedStorm.track,
      colors: selectedStorm.track.map(p => intensityColor(p.vmax)),
    })

    // Đường dự báo AI — dashed cam
    if (selectedStorm.forecast) {
      paths.push({
        id: 'forecast',
        points: [
          selectedStorm.forecast.origin,
          ...selectedStorm.forecast.points,
        ],
      })
    }

    return paths
  }, [selectedStorm])

  // --- Markers tại mỗi điểm track ---
  const pointsData = useMemo(() => {
    if (!selectedStorm) return []
    return selectedStorm.track
  }, [selectedStorm])

  // --- Vòng tròn sai số (uncertainty) tại điểm dự báo ---
  const ringsData = useMemo(() => {
    if (!selectedStorm?.forecast) return []
    return selectedStorm.forecast.points
  }, [selectedStorm])

  // --- Nhãn "+24h" / "+48h" ---
  const labelsData = useMemo(() => {
    if (!selectedStorm?.forecast) return []
    return selectedStorm.forecast.points.map(p => ({
      lat:  p.lat + 0.8,
      lng:  p.lon,
      text: `+${p.hour}h`,
    }))
  }, [selectedStorm])

  // --- Tooltip khi hover điểm track (cập nhật khi đổi ngôn ngữ) ---
  const pointLabel = useCallback(d => `
    <div style="
      background:#1a1a2e;
      padding:8px 12px;
      border-radius:8px;
      border:1px solid #0f3460;
      color:#eee;
      font-size:12px;
      line-height:1.6;
    ">
      <b style="color:#e94560">${d.time}</b><br/>
      ${d.lat.toFixed(1)}°N &nbsp; ${d.lon.toFixed(1)}°E<br/>
      ${t.tooltip.wind} <b>${d.vmax ?? '--'} kt</b> &nbsp;|&nbsp; ${t.tooltip.pres} <b>${d.pmin ?? '--'} hPa</b>
    </div>
  `, [t])

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', background: '#000' }}>
      <Globe
        ref={globeRef}
        width={size.width}
        height={size.height}

        // Texture địa cầu — ảnh ban đêm
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"

        // === Track lịch sử + đường dự báo ===
        pathsData={pathsData}
        pathPoints={d => d.points.map(p => [p.lat, p.lon, 0])}
        pathColor={d =>
          d.id === 'track' ? d.colors : '#f59e0b'
        }
        pathStroke={d => d.id === 'track' ? 2.5 : 2}
        pathDashLength={d => d.id === 'forecast' ? 0.5 : 0}
        pathDashGap={d => d.id === 'forecast' ? 0.25 : 0}
        pathDashAnimateTime={d => d.id === 'forecast' ? 3000 : 0}

        // === Marker tại từng điểm track ===
        pointsData={pointsData}
        pointLat="lat"
        pointLng="lon"
        pointColor={d => intensityColor(d.vmax)}
        pointAltitude={0.006}
        pointRadius={0.28}
        pointLabel={pointLabel}

        // === Vòng sai số tại điểm dự báo ===
        ringsData={ringsData}
        ringLat="lat"
        ringLng="lon"
        ringMaxRadius={d => d.mae_km / 111}   // km → độ (~111 km/độ)
        ringColor={() => t => `rgba(245,158,11,${(1 - t) * 0.6})`}
        ringPropagationSpeed={0.8}
        ringRepeatPeriod={2500}

        // === Nhãn +24h / +48h ===
        labelsData={labelsData}
        labelLat="lat"
        labelLng="lng"
        labelText="text"
        labelSize={0.6}
        labelColor={() => '#fbbf24'}
        labelDotRadius={0.4}
        labelAltitude={0.015}
        labelResolution={3}

        animateIn
      />
    </div>
  )
}
