import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// GET /api/storms → list of { sid, name, season, basin }
export async function fetchStorms() {
  const res = await api.get('/storms')
  return res.data
}

// GET /api/storms/:sid → full storm with track + forecast
export async function fetchStormDetail(sid) {
  const res = await api.get(`/storms/${sid}`)
  return res.data
}

// POST /api/predict → { track: [...last N points] } → { forecast: {...} }
export async function predictTrack(track) {
  const res = await api.post('/predict', { track })
  return res.data
}
