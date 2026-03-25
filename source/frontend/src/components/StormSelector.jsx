export default function StormSelector({ storms, loading, selected, onSelect }) {
  if (loading) return <p style={{ color: '#aaa', fontSize: '0.8rem' }}>Đang tải...</p>
  if (!storms.length) return <p style={{ color: '#aaa', fontSize: '0.8rem' }}>Không có dữ liệu</p>

  return (
    <select
      value={selected?.sid ?? ''}
      onChange={(e) => {
        const storm = storms.find((s) => s.sid === e.target.value)
        onSelect(storm ?? null)
      }}
    >
      <option value="">-- Chọn cơn bão --</option>
      {storms.map((s) => (
        <option key={s.sid} value={s.sid}>
          {s.name} ({s.season}) — {s.sid}
        </option>
      ))}
    </select>
  )
}
