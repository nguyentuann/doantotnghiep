import {
  Chart as ChartJS,
  CategoryScale, LinearScale,
  PointElement, LineElement,
  Title, Tooltip, Legend,
  Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { useLocale } from '../i18n/LocaleContext'

ChartJS.register(
  CategoryScale, LinearScale,
  PointElement, LineElement,
  Title, Tooltip, Legend,
  Filler,
)

export default function IntensityChart({ storm }) {
  const { t } = useLocale()
  const track = storm?.track
  if (!track || track.length === 0) return null

  // Lấy mỗi 2 bước (12h) để tránh nhãn quá dày
  const step  = Math.max(1, Math.floor(track.length / 12))
  const pts   = track.filter((_, i) => i % step === 0)
  const labels = pts.map(p => {
    const d = new Date(p.time || p.iso_time)
    if (isNaN(d)) return ''
    return `${(d.getMonth()+1)}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}h`
  })

  const vmaxData = pts.map(p => p.vmax ?? null)
  const pminData = pts.map(p => p.pmin ?? null)

  const data = {
    labels,
    datasets: [
      {
        label: t.tooltip.wind + ' (kt)',
        data: vmaxData,
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.08)',
        yAxisID: 'yWind',
        tension: 0.3,
        pointRadius: 2,
        borderWidth: 2,
        fill: true,
      },
      {
        label: t.tooltip.pres + ' (hPa)',
        data: pminData,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96,165,250,0.08)',
        yAxisID: 'yPres',
        tension: 0.3,
        pointRadius: 2,
        borderWidth: 2,
        fill: false,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#ccc', font: { size: 10 }, boxWidth: 12 },
      },
      tooltip: {
        bodyColor: '#eee',
        titleColor: '#aaa',
        backgroundColor: '#16213e',
        borderColor: '#0f3460',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: { color: '#888', font: { size: 9 }, maxRotation: 30 },
        grid:  { color: 'rgba(255,255,255,0.05)' },
      },
      yWind: {
        position: 'left',
        ticks: { color: '#f97316', font: { size: 9 } },
        grid:  { color: 'rgba(255,255,255,0.05)' },
        title: { display: true, text: 'kt', color: '#f97316', font: { size: 9 } },
      },
      yPres: {
        position: 'right',
        reverse: true,
        ticks: { color: '#60a5fa', font: { size: 9 } },
        grid:  { drawOnChartArea: false },
        title: { display: true, text: 'hPa', color: '#60a5fa', font: { size: 9 } },
      },
    },
  }

  return (
    <div className="chart-container">
      <p className="chart-title">{t.chartIntensity}</p>
      <div style={{ height: 160 }}>
        <Line data={data} options={options} />
      </div>
    </div>
  )
}
