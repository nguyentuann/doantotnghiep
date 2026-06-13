import { useTheme } from '../theme/ThemeContext'

const SPEEDS = [
  { label: '0.5×', ms: 1200 },
  { label: '1×',   ms: 600  },
  { label: '2×',   ms: 300  },
  { label: '4×',   ms: 150  },
]

export default function TrackAnimator({
  storm,
  animStep,
  isPlaying,
  speed,
  onStep,
  onPlay,
  onReset,
  onSpeed,
}) {
  const { theme } = useTheme()
  if (!storm) return null

  const trackLen = storm.track?.length ?? 0
  const predLen  = (storm.predicted_track?.length ?? 0) * 4  // 24h → 6h
  const total    = trackLen + predLen
  if (total === 0) return null

  const step    = animStep ?? 0
  const isPred  = step >= trackLen
  // predicted đã x4 (6h), lấy điểm AI thực = mỗi 4 bước
  const predIdx = isPred ? Math.floor((step - trackLen) / 4) : 0
  const current = isPred
    ? storm.predicted_track?.[predIdx]
    : storm.track?.[step]
  const timeStr = current?.iso_time ?? current?.time ?? ''

  const bg      = theme === 'dark' ? 'rgba(22,33,62,0.92)' : 'rgba(248,250,252,0.95)'
  const border  = theme === 'dark' ? '#0f3460' : '#cbd5e1'
  const text    = theme === 'dark' ? '#eee' : '#1e293b'
  const muted   = theme === 'dark' ? '#888' : '#64748b'
  const accent  = isPred ? '#f59e0b' : '#60a5fa'

  return (
    <div style={{
      position: 'absolute',
      bottom: 24,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 1000,
      background: bg,
      border: `1px solid ${border}`,
      borderRadius: 12,
      padding: '10px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      minWidth: 360,
      maxWidth: 480,
      boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
      backdropFilter: 'blur(8px)',
    }}>
      {/* Time + label */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: muted }}>
          Bước {step + 1} / {total}
        </span>
        <span style={{ fontSize: 12, color: accent, fontWeight: 600 }}>
          {isPred ? '🔮 Dự đoán AI' : '📍 Thực tế'} — {timeStr}
        </span>
        <span style={{ fontSize: 11, color: muted }}>
          {current ? `${current.lat?.toFixed(1)}°N  ${current.lon?.toFixed(1)}°E` : ''}
        </span>
      </div>

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={total - 1}
        value={step}
        onChange={e => onStep(Number(e.target.value))}
        style={{ width: '100%', accentColor: accent, cursor: 'pointer' }}
      />

      {/* Track markers dưới slider */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: muted, marginTop: -6 }}>
        <span>Start</span>
        {predLen > 0 && (
          <span style={{ color: '#f59e0b' }}>
            ▲ AI ({Math.round(trackLen / total * 100)}%)
          </span>
        )}
        <span>End</span>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {/* Reset */}
        <button onClick={onReset} style={btnStyle(theme)} title="Reset">⏮</button>

        {/* Play/Pause */}
        <button
          onClick={onPlay}
          style={{ ...btnStyle(theme), background: accent, color: '#fff', border: 'none', padding: '5px 18px', flex: 1, fontSize: 14 }}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>

        {/* Speed */}
        <div style={{ display: 'flex', gap: 3 }}>
          {SPEEDS.map(s => (
            <button
              key={s.ms}
              onClick={() => onSpeed(s.ms)}
              style={{
                ...btnStyle(theme),
                background: speed === s.ms ? accent : 'transparent',
                color: speed === s.ms ? '#fff' : muted,
                border: `1px solid ${speed === s.ms ? accent : border}`,
                padding: '3px 7px',
                fontSize: 10,
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function btnStyle(theme) {
  return {
    padding: '4px 10px',
    borderRadius: 6,
    border: `1px solid ${theme === 'dark' ? '#0f3460' : '#cbd5e1'}`,
    background: 'transparent',
    color: theme === 'dark' ? '#eee' : '#1e293b',
    cursor: 'pointer',
    fontSize: 13,
  }
}
