import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// GET /api/dashboard/models → { cliper, models: [...] }
export async function fetchDashboardModels() {
  const res = await api.get('/dashboard/models')
  return res.data
}

// GET /api/dashboard/per-storm/:tag → { tag, best_model, n_storms, per_storm: [...] }
export async function fetchPerStorm(tag) {
  const res = await api.get(`/dashboard/per-storm/${tag}`)
  return res.data
}

// GET /api/dashboard/feature-importance/:tag → { ..., per_feature_clean, per_group }
export async function fetchFeatureImportance(tag) {
  const res = await api.get(`/dashboard/feature-importance/${tag}`)
  return res.data
}

// GET /api/dashboard/ablation → { progression: [...] }
export async function fetchAblation() {
  const res = await api.get('/dashboard/ablation')
  return res.data
}
