import { useState, useEffect } from 'react'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale,
  BarElement, PointElement, LineElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'
import {
  fetchDashboardModels, fetchPerStorm,
  fetchFeatureImportance, fetchAblation,
} from '../api/dashboard'
import { useTheme } from '../theme/ThemeContext'

// Màu chart thích ứng theme — đọc 1 lần ở component cha, truyền xuống các chart
function chartColors(theme) {
  const light = theme === 'light'
  return {
    text: light ? '#334155' : '#cccccc',     // nhãn legend/title
    tick: light ? '#475569' : '#aaaaaa',      // nhãn trục
    grid: light ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.05)',
  }
}

ChartJS.register(
  CategoryScale, LinearScale,
  BarElement, PointElement, LineElement,
  Title, Tooltip, Legend,
)

const ARCH_COLORS = {
  lstm: '#94a3b8', bilstm: '#60a5fa', bigru: '#fbbf24', transformer: '#34d399',
}

export default function ModelDashboard() {
  const [data, setData]       = useState(null)     // {cliper, models}
  const [selectedTag, setTag] = useState(null)
  const [perStorm, setPerStorm]   = useState(null)
  const [featImp, setFeatImp]     = useState(null)
  const [ablation, setAblation]   = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const { theme } = useTheme()
  const C = chartColors(theme)

  // Load danh sách model + ablation 1 lần
  useEffect(() => {
    Promise.all([fetchDashboardModels(), fetchAblation()])
      .then(([models, abl]) => {
        setData(models)
        setAblation(abl)
        const champ = models.models.find(m => m.is_champion) ?? models.models[0]
        setTag(champ?.tag ?? null)
      })
      .catch(e => setError(e.message || 'Không kết nối được backend'))
      .finally(() => setLoading(false))
  }, [])

  // Load per-storm + feature-importance khi đổi tag
  useEffect(() => {
    if (!selectedTag) return
    const m = data?.models.find(x => x.tag === selectedTag)
    setPerStorm(null); setFeatImp(null)
    if (m?.has_per_storm) fetchPerStorm(selectedTag).then(setPerStorm).catch(() => {})
    if (m?.has_feature_importance) fetchFeatureImportance(selectedTag).then(setFeatImp).catch(() => {})
  }, [selectedTag, data])

  if (loading) return <div className="dash-state">Đang tải dữ liệu model…</div>
  if (error)   return <div className="dash-state dash-error">⚠ {error}<br/><small>Hãy chắc chắn backend đang chạy (uvicorn main:app --port 8000)</small></div>
  if (!data)   return <div className="dash-state">Không có dữ liệu.</div>

  const model = data.models.find(m => m.tag === selectedTag)
  const cliper = data.cliper

  return (
    <div className="dashboard">
      {/* ─── Header + Model selector ─── */}
      <div className="dash-header">
        <div>
          <h2>Model Dashboard</h2>
          <p className="dash-sub">So sánh và phân tích các mô hình dự đoán bão SCS</p>
        </div>
        <div className="dash-selector">
          <label>Chọn model (tag):</label>
          <select value={selectedTag ?? ''} onChange={e => setTag(e.target.value)}>
            {data.models.map(m => (
              <option key={m.tag} value={m.tag}>
                {m.display} — Skill {m.best_skill_24h?.toFixed(1)}%
              </option>
            ))}
          </select>
        </div>
      </div>

      {model && (
        <>
          {/* ─── PHẦN 1: Overview cards ─── */}
          <Section title="Tổng quan model">
            <div className="dash-cards">
              <Card label="MAE 24h" value={`${model.best_mae_24h?.toFixed(1)} km`} color="#10b981" />
              <Card label="Skill 24h" value={`${model.best_skill_24h?.toFixed(1)}%`} color="#2563eb" highlight />
              <Card label="Số đặc trưng" value={model.n_features} color="#d97706" />
              <Card label="Lookback" value={`${model.lookback} bước`} color="#7c3aed" />
              <Card label="Kiến trúc tốt nhất" value={model.architectures.find(a=>a.arch===model.best_arch)?.display} color="#ea580c" />
              <Card label="CLIPER 24h" value={`${cliper.mae_24h} km`} color="#64748b" />
            </div>
            <p className="dash-desc">{model.description}</p>
          </Section>

          {/* ─── PHẦN 2: So sánh kiến trúc ─── */}
          <Section title="So sánh kiến trúc">
            {model.architectures.length > 1 ? (
              <div className="dash-grid-2">
                <div className="dash-chart">
                  <ArchBarChart archs={model.architectures} cliper={cliper} C={C} />
                </div>
                <ArchTable archs={model.architectures} bestArch={model.best_arch} />
              </div>
            ) : (
              <div>
                <p className="dash-note">Tag này chỉ huấn luyện kiến trúc Transformer.</p>
                <ArchTable archs={model.architectures} bestArch={model.best_arch} />
              </div>
            )}
          </Section>

          {/* ─── PHẦN 3: Ablation progression ─── */}
          {ablation?.progression?.length > 0 && (
            <Section title="Tiến trình cải tiến (ablation)">
              <div className="dash-chart" style={{ height: 280 }}>
                <AblationChart steps={ablation.progression} C={C} />
              </div>
            </Section>
          )}

          {/* ─── PHẦN 4a: Feature importance ─── */}
          {featImp && (
            <Section title="Tầm quan trọng đặc trưng (permutation)">
              <p className="dash-note">
                Baseline MAE 24h = {featImp.baseline_mae_24h} km · {featImp.n_repeats} lần lặp ·
                (đã loại lat/lon do artifact CLIPER)
              </p>
              <div className="dash-chart" style={{ height: 340 }}>
                <FeatureImportanceChart feats={featImp.per_feature_clean.slice(0, 15)} C={C} />
              </div>
            </Section>
          )}

          {/* ─── PHẦN 4b: Per-storm ─── */}
          {perStorm && (
            <Section title={`Chi tiết theo cơn bão (${perStorm.n_storms} bão)`}>
              <PerStormTable storms={perStorm.per_storm} />
            </Section>
          )}

          {!featImp && !perStorm && (
            <p className="dash-note">
              Tag <b>{model.tag}</b> chưa có dữ liệu per-storm / feature-importance chi tiết.
              Chọn <b>scs_v12_lb6</b> (champion) để xem đầy đủ.
            </p>
          )}
        </>
      )}
    </div>
  )
}

// ─── Sub-components ──────────────────────────────────────────────────────

function Section({ title, children }) {
  return (
    <div className="dash-section">
      <h3 className="dash-section-title">{title}</h3>
      {children}
    </div>
  )
}

function Card({ label, value, color, highlight }) {
  return (
    <div className={`dash-card ${highlight ? 'dash-card-hl' : ''}`}
         style={{ borderTopColor: color }}>
      <div className="dash-card-value" style={{ color }}>{value}</div>
      <div className="dash-card-label">{label}</div>
    </div>
  )
}

function ArchTable({ archs, bestArch }) {
  return (
    <table className="dash-table">
      <thead>
        <tr><th>Kiến trúc</th><th>Params</th><th>MAE 24h</th><th>Skill 24h</th><th>Skill 48h</th></tr>
      </thead>
      <tbody>
        {archs.map(a => (
          <tr key={a.arch} className={a.arch === bestArch ? 'dash-row-best' : ''}>
            <td>{a.display}{a.arch === bestArch ? ' ★' : ''}</td>
            <td>{(a.params/1000).toFixed(0)}k</td>
            <td>{a.mae_24h?.toFixed(1)} km</td>
            <td className="skill-pos">{a.skill_24h?.toFixed(1)}%</td>
            <td>{a.skill_48h?.toFixed(1)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ArchBarChart({ archs, C }) {
  const data = {
    labels: archs.map(a => a.display),
    datasets: [
      { label: 'MAE 24h (km)', data: archs.map(a => a.mae_24h),
        backgroundColor: archs.map(a => ARCH_COLORS[a.arch] || '#888') },
    ],
  }
  const options = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: C.text, font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: C.tick, font: { size: 10 } }, grid: { display: false } },
      y: { ticks: { color: C.tick, font: { size: 9 } }, grid: { color: C.grid },
           title: { display: true, text: 'MAE 24h (km)', color: C.tick, font: { size: 9 } } },
    },
  }
  return <Bar data={data} options={options} />
}

function AblationChart({ steps, C }) {
  const data = {
    labels: steps.map(s => s.tag.replace('scs_', '')),
    datasets: [
      { label: 'Skill 24h (%)', data: steps.map(s => s.skill_24h),
        borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.12)',
        tension: 0.3, pointRadius: 4, borderWidth: 2, fill: true },
      { label: 'Skill 48h (%)', data: steps.map(s => s.skill_48h),
        borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)',
        tension: 0.3, pointRadius: 4, borderWidth: 2, fill: false },
    ],
  }
  const options = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: C.text, font: { size: 10 } } },
      tooltip: {
        callbacks: {
          afterBody: (items) => {
            const i = items[0].dataIndex
            return steps[i].label
          },
        },
      },
    },
    scales: {
      x: { ticks: { color: C.tick, font: { size: 10 } }, grid: { color: C.grid } },
      y: { ticks: { color: C.tick, font: { size: 9 } }, grid: { color: C.grid },
           title: { display: true, text: 'Skill Score (%)', color: C.tick, font: { size: 9 } } },
    },
  }
  return <Line data={data} options={options} />
}

function FeatureImportanceChart({ feats, C }) {
  const data = {
    labels: feats.map(f => f.feature),
    datasets: [
      { label: 'MAE tăng khi shuffle (km)', data: feats.map(f => f.mae_increase),
        backgroundColor: feats.map(f =>
          f.feature.startsWith('asteer') ? '#10b981'
          : (f.feature === 'dlat' || f.feature === 'dlon' || f.feature === 'speed_kmh') ? '#3b82f6'
          : '#f59e0b'),
      },
    ],
  }
  const options = {
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: C.text, font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: C.tick, font: { size: 9 } }, grid: { color: C.grid },
           title: { display: true, text: 'MAE +(km)', color: C.tick, font: { size: 9 } } },
      y: { ticks: { color: C.tick, font: { size: 9 } }, grid: { display: false } },
    },
  }
  return <Bar data={data} options={options} />
}

function PerStormTable({ storms }) {
  const [sortKey, setSortKey] = useState('mae_24h')
  const [asc, setAsc] = useState(true)
  const sorted = [...storms].sort((a, b) => {
    const v = (a[sortKey] - b[sortKey]) * (asc ? 1 : -1)
    return v
  })
  function header(key, label) {
    return (
      <th onClick={() => { sortKey === key ? setAsc(!asc) : (setSortKey(key), setAsc(true)) }}
          style={{ cursor: 'pointer' }}>
        {label}{sortKey === key ? (asc ? ' ▲' : ' ▼') : ''}
      </th>
    )
  }
  return (
    <div className="dash-table-scroll">
      <table className="dash-table dash-table-storm">
        <thead>
          <tr>
            {header('sid', 'SID')}
            {header('season', 'Năm')}
            {header('n_sequences', 'N')}
            {header('mae_24h', 'MAE 24h')}
            {header('mae_48h', 'MAE 48h')}
            {header('skill_24h', 'Skill 24h')}
            {header('skill_48h', 'Skill 48h')}
          </tr>
        </thead>
        <tbody>
          {sorted.map(s => (
            <tr key={s.sid}>
              <td>{s.sid}</td>
              <td>{s.season}</td>
              <td>{s.n_sequences}</td>
              <td>{s.mae_24h?.toFixed(1)}</td>
              <td>{s.mae_48h?.toFixed(1)}</td>
              <td className={s.skill_24h > 0 ? 'skill-pos' : 'skill-neg'}>{s.skill_24h?.toFixed(1)}%</td>
              <td className={s.skill_48h > 0 ? 'skill-pos' : 'skill-neg'}>{s.skill_48h?.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
