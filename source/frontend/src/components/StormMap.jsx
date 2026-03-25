import { MapContainer, TileLayer, useMap } from 'react-leaflet'
import TrackLayer from './TrackLayer'
import ForecastLayer from './ForecastLayer'

// SCS center
const SCS_CENTER = [15, 114]
const SCS_ZOOM = 5

function MapUpdater({ storm }) {
  const map = useMap()
  if (storm?.track?.length > 0) {
    const last = storm.track[storm.track.length - 1]
    map.flyTo([last.lat, last.lon], 6, { duration: 1 })
  }
  return null
}

export default function StormMap({ selectedStorm }) {
  return (
    <MapContainer
      center={SCS_CENTER}
      zoom={SCS_ZOOM}
      style={{ height: '100%', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      />

      {selectedStorm && (
        <>
          <MapUpdater storm={selectedStorm} />
          <TrackLayer track={selectedStorm.track} />
          {selectedStorm.forecast && (
            <ForecastLayer forecast={selectedStorm.forecast} />
          )}
        </>
      )}
    </MapContainer>
  )
}
