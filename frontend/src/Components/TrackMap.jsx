import { useEffect } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const windColor = (kt) => (kt >= 64 ? "#ef4444" : kt >= 34 ? "#f59e0b" : "#10b981");

function FitTrack({ points }) {
  const map = useMap();
  useEffect(() => {
    if (!points.length) return;
    map.fitBounds(points.map((p) => [p.latitude, p.longitude]), { padding: [40, 40], maxZoom: 7 });
  }, [points, map]);
  return null;
}

/** track: [{latitude, longitude, wind_knots, iso_time, is_forecast?}] — forecast point drawn dashed in sky-blue. */
export default function TrackMap({ track = [] }) {
  const valid = track.filter((p) => Number.isFinite(p.latitude) && Number.isFinite(p.longitude));
  const observed = valid.filter((p) => !p.is_forecast);
  const forecast = valid.find((p) => p.is_forecast);
  const last = observed[observed.length - 1];

  if (!valid.length) {
    return (
      <div className="h-[420px] flex items-center justify-center text-[13px] text-ink-faint">
        No track positions available yet.
      </div>
    );
  }

  return (
    <MapContainer
      center={[valid[0].latitude, valid[0].longitude]}
      zoom={5}
      scrollWheelZoom={false}
      style={{ height: 420, width: "100%", background: "#08090a" }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; OpenStreetMap contributors &copy; CARTO'
      />
      <FitTrack points={valid} />
      {observed.length > 1 && (
        <Polyline positions={observed.map((p) => [p.latitude, p.longitude])} pathOptions={{ color: "#B4B7BD", weight: 2 }} />
      )}
      {last && forecast && (
        <Polyline
          positions={[[last.latitude, last.longitude], [forecast.latitude, forecast.longitude]]}
          pathOptions={{ color: "#38bdf8", weight: 2.5, dashArray: "6 6" }}
        />
      )}
      {observed.map((p, i) => (
        <CircleMarker
          key={i}
          center={[p.latitude, p.longitude]}
          radius={i === observed.length - 1 ? 8 : 4}
          pathOptions={{ color: "#08090a", weight: 1, fillColor: windColor(p.wind_knots), fillOpacity: 0.95 }}
        >
          <Tooltip>{`${p.iso_time || "Current"} · ${Math.round(p.wind_knots)} kt`}</Tooltip>
        </CircleMarker>
      ))}
      {forecast && (
        <CircleMarker
          center={[forecast.latitude, forecast.longitude]}
          radius={8}
          pathOptions={{ color: "#38bdf8", weight: 2, fillColor: "#38bdf8", fillOpacity: 0.25 }}
        >
          <Tooltip permanent direction="top">{`AI +6 h · ${Math.round(forecast.wind_knots)} kt`}</Tooltip>
        </CircleMarker>
      )}
    </MapContainer>
  );
}
