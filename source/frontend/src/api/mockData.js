/**
 * Mock data cho preview khi backend chưa chạy
 * Dựa trên track thực tế của các cơn bão test set 2021–2024
 */

// Typhoon RAI (2021) — đổ bộ Philippines, vào Biển Đông
const RAI_TRACK = [
  { lat: 10.2, lon: 126.8, vmax: 35,  pmin: 998, time: '2021-12-13 00:00' },
  { lat: 10.4, lon: 125.6, vmax: 45,  pmin: 990, time: '2021-12-13 06:00' },
  { lat: 10.6, lon: 124.3, vmax: 65,  pmin: 975, time: '2021-12-13 12:00' },
  { lat: 10.9, lon: 123.1, vmax: 95,  pmin: 950, time: '2021-12-13 18:00' },
  { lat: 11.2, lon: 121.8, vmax: 115, pmin: 930, time: '2021-12-14 00:00' },
  { lat: 11.5, lon: 120.5, vmax: 120, pmin: 925, time: '2021-12-14 06:00' },
  { lat: 11.8, lon: 119.2, vmax: 115, pmin: 930, time: '2021-12-14 12:00' },
  { lat: 12.1, lon: 117.9, vmax: 105, pmin: 940, time: '2021-12-14 18:00' },
  { lat: 12.4, lon: 116.5, vmax: 95,  pmin: 950, time: '2021-12-15 00:00' },
  { lat: 12.6, lon: 115.1, vmax: 85,  pmin: 960, time: '2021-12-15 06:00' },
  { lat: 12.8, lon: 113.8, vmax: 75,  pmin: 968, time: '2021-12-15 12:00' },
  { lat: 13.0, lon: 112.4, vmax: 65,  pmin: 975, time: '2021-12-15 18:00' },
  { lat: 13.2, lon: 111.0, vmax: 60,  pmin: 980, time: '2021-12-16 00:00' },
  { lat: 13.3, lon: 109.6, vmax: 55,  pmin: 984, time: '2021-12-16 06:00' },
  { lat: 13.4, lon: 108.3, vmax: 50,  pmin: 988, time: '2021-12-16 12:00' },
  { lat: 13.5, lon: 107.0, vmax: 45,  pmin: 992, time: '2021-12-16 18:00' },
]

// Typhoon NORU (2022) — đổ bộ miền Trung Việt Nam
const NORU_TRACK = [
  { lat: 13.5, lon: 131.2, vmax: 35,  pmin: 998, time: '2022-09-23 00:00' },
  { lat: 13.8, lon: 129.5, vmax: 55,  pmin: 985, time: '2022-09-23 12:00' },
  { lat: 14.0, lon: 127.8, vmax: 75,  pmin: 970, time: '2022-09-24 00:00' },
  { lat: 14.2, lon: 126.1, vmax: 95,  pmin: 950, time: '2022-09-24 12:00' },
  { lat: 14.3, lon: 124.4, vmax: 115, pmin: 930, time: '2022-09-25 00:00' },
  { lat: 14.4, lon: 122.7, vmax: 130, pmin: 915, time: '2022-09-25 12:00' },
  { lat: 14.5, lon: 121.0, vmax: 130, pmin: 915, time: '2022-09-26 00:00' },
  { lat: 14.5, lon: 119.3, vmax: 120, pmin: 925, time: '2022-09-26 12:00' },
  { lat: 14.4, lon: 117.6, vmax: 110, pmin: 935, time: '2022-09-27 00:00' },
  { lat: 14.3, lon: 115.9, vmax: 100, pmin: 945, time: '2022-09-27 12:00' },
  { lat: 14.1, lon: 114.2, vmax: 90,  pmin: 955, time: '2022-09-28 00:00' },
  { lat: 13.9, lon: 112.5, vmax: 80,  pmin: 963, time: '2022-09-28 12:00' },
  { lat: 13.7, lon: 110.8, vmax: 75,  pmin: 968, time: '2022-09-29 00:00' },
  { lat: 13.5, lon: 109.1, vmax: 70,  pmin: 972, time: '2022-09-29 12:00' },
  { lat: 13.3, lon: 107.8, vmax: 65,  pmin: 976, time: '2022-09-30 00:00' },
]

// Typhoon HAIKUI (2023) — đổ bộ Đài Loan, vào Biển Đông
const HAIKUI_TRACK = [
  { lat: 19.5, lon: 126.3, vmax: 45,  pmin: 990, time: '2023-09-01 00:00' },
  { lat: 20.1, lon: 124.8, vmax: 65,  pmin: 975, time: '2023-09-01 12:00' },
  { lat: 20.6, lon: 123.2, vmax: 85,  pmin: 958, time: '2023-09-02 00:00' },
  { lat: 21.0, lon: 121.6, vmax: 100, pmin: 945, time: '2023-09-02 12:00' },
  { lat: 21.3, lon: 120.0, vmax: 110, pmin: 935, time: '2023-09-03 00:00' },
  { lat: 21.5, lon: 118.5, vmax: 105, pmin: 940, time: '2023-09-03 12:00' },
  { lat: 21.4, lon: 117.0, vmax: 95,  pmin: 950, time: '2023-09-04 00:00' },
  { lat: 21.2, lon: 115.5, vmax: 85,  pmin: 960, time: '2023-09-04 12:00' },
  { lat: 20.9, lon: 114.0, vmax: 75,  pmin: 968, time: '2023-09-05 00:00' },
  { lat: 20.5, lon: 112.5, vmax: 65,  pmin: 975, time: '2023-09-05 12:00' },
  { lat: 20.1, lon: 111.2, vmax: 55,  pmin: 983, time: '2023-09-06 00:00' },
  { lat: 19.7, lon: 110.1, vmax: 50,  pmin: 988, time: '2023-09-06 12:00' },
]

// Forecast mock — tính từ vị trí cuối của mỗi track
function makeForecast(track, mae24 = 105, mae48 = 183) {
  const last = track[track.length - 1]
  const prev = track[track.length - 2]
  const dlat = last.lat - prev.lat
  const dlon = last.lon - prev.lon

  return {
    origin: { lat: last.lat, lon: last.lon },
    points: [
      {
        lat: parseFloat((last.lat + 4 * dlat + (Math.random() - 0.5) * 0.3).toFixed(2)),
        lon: parseFloat((last.lon + 4 * dlon + (Math.random() - 0.5) * 0.3).toFixed(2)),
        hour: 24,
        mae_km: mae24,
      },
      {
        lat: parseFloat((last.lat + 8 * dlat + (Math.random() - 0.5) * 0.5).toFixed(2)),
        lon: parseFloat((last.lon + 8 * dlon + (Math.random() - 0.5) * 0.5).toFixed(2)),
        hour: 48,
        mae_km: mae48,
      },
    ],
  }
}

export const MOCK_STORMS = [
  {
    sid:    '2021341N10135',
    name:   'RAI',
    season: 2021,
    basin:  'WP',
    track:  RAI_TRACK,
    forecast: makeForecast(RAI_TRACK),
  },
  {
    sid:    '2022268N14131',
    name:   'NORU',
    season: 2022,
    basin:  'WP',
    track:  NORU_TRACK,
    forecast: makeForecast(NORU_TRACK),
  },
  {
    sid:    '2023244N19126',
    name:   'HAIKUI',
    season: 2023,
    basin:  'WP',
    track:  HAIKUI_TRACK,
    forecast: makeForecast(HAIKUI_TRACK),
  },
]
