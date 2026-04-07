import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// Chuẩn hóa track point: iso_time → time
function normalizeTrack(track = []) {
  return track.map(p => ({ ...p, time: p.iso_time ?? p.time }))
}

// Chuẩn hóa forecast: {origin_lat, origin_lon, points} → {origin: {lat, lon}, points}
function normalizeForecast(forecast) {
  if (!forecast) return null
  return {
    origin:     { lat: forecast.origin_lat, lon: forecast.origin_lon },
    points:     forecast.points ?? [],
    model_used: forecast.model_used,
  }
}

// GET /api/storms → list of { sid, name, season, basin }
export async function fetchStorms() {
  const res = await api.get('/storms')
  return res.data
}

// GET /api/storms/:sid → full storm với track + predicted_track
export async function fetchStormDetail(sid) {
  const res = await api.get(`/storms/${sid}`)
  const s   = res.data
  return {
    ...s,
    track:           normalizeTrack(s.track),
    predicted_track: normalizeTrack(s.predicted_track ?? []),
  }
}

// POST /api/predict → { track: [...last N points] } → { forecast: {...} }
export async function predictTrack(track) {
  const res = await api.post('/predict', { track })
  return normalizeForecast(res.data.forecast)
}
