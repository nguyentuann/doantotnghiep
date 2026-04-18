import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import Globe from 'react-globe.gl'
import { useLocale } from '../i18n/LocaleContext'

// Tâm mặc định — Biển Đông
const SCS_VIEW = { lat: 15, lng: 114, altitude: 2.2 }

// Tính hướng di chuyển (bearing) giữa 2 điểm
function bearingDeg(lat1, lon1, lat2, lon2) {
  const toR = d => d * Math.PI / 180
  const dLon = toR(lon2 - lon1)
  const y = Math.sin(dLon) * Math.cos(toR(lat2))
  const x = Math.cos(toR(lat1)) * Math.sin(toR(lat2)) - Math.sin(toR(lat1)) * Math.cos(toR(lat2)) * Math.cos(dLon)
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360
}

function compassDir(deg) {
  const dirs = ['Bắc','Đông Bắc','Đông','Đông Nam','Nam','Tây Nam','Tây','Tây Bắc']
  return dirs[Math.round(deg / 45) % 8]
}

// Thêm field `direction` vào mỗi điểm
function withDirection(points) {
  return points.map((p, i) => {
    if (i === 0) return { ...p, direction: null }
    const prev = points[i - 1]
    const deg  = bearingDeg(prev.lat, prev.lon, p.lat, p.lon)
    return { ...p, direction: deg }
  })
}

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

export default function GlobeMap({ selectedStorm, showActual = true, showPredicted = true }) {
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

  // --- Phát hiện điểm đổ bộ (vào đất liền VN/TQ) ---
  // Dừng hiển thị khi lon < 108.5°E (bờ đông Việt Nam / Hải Nam)
  // Trả về index điểm ĐẦU TIÊN đi qua bờ biển (slice tới index đó để hiện điểm đổ bộ)
  function landfallIndex(points) {
    for (let i = 1; i < points.length; i++) {
      if (points[i].lon < 108.5) return i
    }
    return points.length
  }

  // --- Paths: 2 đường chính ---
  const pathsData = useMemo(() => {
    if (!selectedStorm) return []
    const cutoff = selectedStorm.cutoff_index ?? selectedStorm.track.length
    const paths  = []

    // 1. Track thực tế — toàn bộ (kể cả trong SCS để so sánh với dự đoán)
    if (showActual && selectedStorm.track.length >= 2) {
      paths.push({
        id:     'actual',
        points: selectedStorm.track,
        colors: selectedStorm.track.map(p => intensityColor(p.vmax)),
      })
    }

    // 2. Track dự đoán từ cutoff — cắt tại bờ biển Việt Nam
    const pred = selectedStorm.predicted_track ?? []
    if (showPredicted && pred.length >= 1) {
      const cutoffPt = selectedStorm.track[cutoff - 1] ?? selectedStorm.track.at(-1)
      const allPred  = [cutoffPt, ...pred]
      const stopIdx  = landfallIndex(allPred)
      const predPts  = allPred.slice(0, stopIdx + 1)  // +1 để hiện điểm đổ bộ
      if (predPts.length >= 2) {
        paths.push({ id: 'predicted', points: predPts })
      }
    }

    return paths
  }, [selectedStorm, showActual, showPredicted])

  // --- Markers: actual + predicted gộp chung, có direction ---
  const pointsData = useMemo(() => {
    if (!selectedStorm) return []
    const cutoff = selectedStorm.cutoff_index ?? selectedStorm.track.length
    const result = []

    if (showActual) {
      const pts = withDirection(selectedStorm.track)
      pts.forEach(p => result.push({ ...p, isPredicted: false }))
    }

    if (showPredicted) {
      const pred    = selectedStorm.predicted_track ?? []
      const cutoffPt = selectedStorm.track[cutoff - 1] ?? selectedStorm.track.at(-1)
      const allPred  = withDirection([cutoffPt, ...pred])
      allPred.slice(1).forEach(p => result.push({ ...p, isPredicted: true }))
    }

    return result
  }, [selectedStorm, showActual, showPredicted])

  // --- Marker điểm vào SCS ---
  const cutoffData = useMemo(() => {
    if (!selectedStorm) return []
    const cutoff = selectedStorm.cutoff_index ?? 0
    const pt = selectedStorm.track[cutoff]
    return pt ? [pt] : []
  }, [selectedStorm])

  // --- Nhãn điểm đầu dự đoán ---
  const labelsData = useMemo(() => {
    const pred = selectedStorm?.predicted_track ?? []
    if (!pred.length) return []
    return [pred[0]].map(p => ({
      lat:  p.lat + 0.8,
      lng:  p.lon,
      text: t.tooltip?.forecastStart ?? 'Dự báo',
    }))
  }, [selectedStorm, t])

  // --- Tooltip khi hover điểm track ---
  const pointLabel = useCallback(d => {
    const timeStr = d.time ?? d.iso_time ?? ''
    const dirStr  = d.direction != null
      ? `${compassDir(d.direction)} (${Math.round(d.direction)}°)`
      : '—'
    const typeColor = d.isPredicted ? '#f59e0b' : '#e94560'
    const typeLabel = d.isPredicted ? '🔮 Dự đoán AI' : '📍 Thực tế'
    return `
      <div style="
        background:#1a1a2e;padding:8px 12px;border-radius:8px;
        border:1px solid #0f3460;color:#eee;font-size:12px;line-height:1.8;
        min-width:160px;
      ">
        <b style="color:${typeColor}">${typeLabel}</b><br/>
        <b>${timeStr}</b><br/>
        ${d.lat.toFixed(2)}°N &nbsp; ${d.lon.toFixed(2)}°E<br/>
        ${t.tooltip.wind} <b>${d.vmax ?? '--'} kt</b> &nbsp;|&nbsp; ${t.tooltip.pres} <b>${d.pmin ?? '--'} hPa</b><br/>
        ↗ ${dirStr}
      </div>
    `
  }, [t])

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
        pathColor={d => d.id === 'actual' ? d.colors : '#f59e0b'}
        pathStroke={d => d.id === 'actual' ? 2.5 : 2.5}
        pathDashLength={d => d.id === 'predicted' ? 0.5 : 0}
        pathDashGap={d => d.id === 'predicted' ? 0.25 : 0}
        pathDashAnimateTime={d => d.id === 'predicted' ? 3000 : 0}

        // === Marker tại từng điểm track ===
        pointsData={pointsData}
        pointLat="lat"
        pointLng="lon"
        pointColor={d => d.isPredicted ? '#f59e0b' : intensityColor(d.vmax)}
        pointAltitude={0.006}
        pointRadius={0.28}
        pointLabel={pointLabel}

        // === Vòng sóng tại điểm vào SCS ===
        ringsData={cutoffData}
        ringLat="lat"
        ringLng="lon"
        ringMaxRadius={2.5}
        ringColor={() => () => 'rgba(255,255,255,0.6)'}
        ringPropagationSpeed={0.6}
        ringRepeatPeriod={2000}

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
