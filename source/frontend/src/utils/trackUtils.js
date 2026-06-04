/**
 * Chuẩn hoá predicted_track thành mảng 6h-interval để hiển thị.
 *
 * Sprint1 backend trả về 6h-interval points → pass through, đánh dấu tất cả isAIPoint=true.
 * Legacy backend trả về 24h-interval points → interpolate 4×6h như cũ (isAIPoint chỉ tại 24h).
 *
 * Phát hiện interval dựa trên khoảng thời gian giữa 2 điểm đầu tiên.
 */
export function interpolatePredicted6h(predicted, cutoffPoint) {
  if (!predicted?.length || !cutoffPoint) return []

  // Phát hiện interval: nếu 2 điểm đầu cách nhau ≤ 8h → đã là 6h
  const t0ms = new Date(cutoffPoint.iso_time ?? cutoffPoint.time ?? 0).getTime()
  const t1ms = new Date(predicted[0].iso_time ?? predicted[0].time ?? 0).getTime()
  const firstIntervalH = (t1ms - t0ms) / 3600000

  if (firstIntervalH <= 8) {
    // Sprint1: đã là 6h → pass through với isAIPoint=true
    return predicted.map(p => ({ ...p, isAIPoint: true }))
  }

  // Legacy 24h → interpolate 4×6h
  const result    = []
  const allPoints = [cutoffPoint, ...predicted]

  for (let i = 1; i < allPoints.length; i++) {
    const from = allPoints[i - 1]
    const to   = allPoints[i]

    const t0 = new Date(from.iso_time ?? from.time ?? 0).getTime()
    const t1 = new Date(to.iso_time   ?? to.time   ?? 0).getTime()
    const dt = t1 - t0

    for (let step = 1; step <= 4; step++) {
      const frac = step / 4
      const tMs  = dt > 0 ? t0 + frac * dt : 0

      result.push({
        lat:       from.lat + frac * (to.lat - from.lat),
        lon:       from.lon + frac * (to.lon - from.lon),
        iso_time:  tMs > 0 ? new Date(tMs).toISOString().replace('T', ' ').slice(0, 19) : '',
        vmax:      to.vmax,
        pmin:      to.pmin,
        isAIPoint: step === 4,
      })
    }
  }

  return result
}
