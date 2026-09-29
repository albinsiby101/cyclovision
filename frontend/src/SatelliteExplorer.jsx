import { useEffect, useState } from "react";
import { MapContainer, TileLayer, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Crosshair, Layers, LocateFixed, MapPin, Satellite, ShieldAlert, TrendingUp, TriangleAlert
} from "lucide-react";
import { analyseGpsBox, fetchClimatology } from "./api";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
];

function levelStyle(level) {
  const map = {
    LOW: { color: "#10b981", ring: "rgba(16,185,129,0.18)" },
    MODERATE: { color: "#f59e0b", ring: "rgba(245,158,11,0.18)" },
    HIGH: { color: "#ef4444", ring: "rgba(239,68,68,0.18)" },
    EXTREME: { color: "#dc2626", ring: "rgba(220,38,38,0.20)" }
  };
  return map[level] || { color: "#94a3b8", ring: "rgba(148,163,184,0.15)" };
}

function Cell({ label, value, sub, accent }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-2 border-b border-hairline last:border-0">
      <div className="text-[12px] text-ink-soft">{label}</div>
      <div className="text-right">
        <div className="text-[13px] font-semibold" style={{ color: accent || undefined }}>{value}</div>
        {sub && <div className="text-[11px] text-ink-soft">{sub}</div>}
      </div>
    </div>
  );
}

const rect = { color: "#f59e0b", weight: 2, dashArray: "6 4", fillOpacity: 0.08 };

let rectStart = null;
let rectLayer = null;

function DrawRect({ drawMode, onDone }) {
  const map = useMapEvents({
    mousedown: (e) => {
      if (!drawMode || e.originalEvent.button !== 0) return;
      if (map.dragging) map.dragging.disable();
      if (rectLayer) rectLayer.remove();
      rectStart = e.latlng;
      rectLayer = L.rectangle(L.latLngBounds(e.latlng, e.latlng), rect).addTo(map);
    },
    mousemove: (e) => {
      if (!drawMode || !rectStart || !rectLayer) return;
      rectLayer.setBounds(L.latLngBounds(rectStart, e.latlng));
    },
    mouseup: (e) => {
      if (!drawMode || !rectStart) return;
      const b = L.latLngBounds(rectStart, e.latlng);
      if (rectLayer) { rectLayer.remove(); rectLayer = null; }
      rectStart = null;
      if (map.dragging) map.dragging.enable();
      if (b.isValid() && b.getSouth() !== b.getNorth() && b.getWest() !== b.getEast()) onDone(b);
    }
  });
  return null;
}

function MapControls({ drawMode, setDrawMode, onLocate, month, onMonth,
                       climVisible, onToggleClim, climatology, climatologyLoaded }) {
  return (
    <div className="absolute top-3 left-3 z-[1000] space-y-2">
      <div className="flex items-center gap-1.5">
        <button
          onClick={() => setDrawMode((v) => !v)}
          className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-semibold shadow-card border transition-colors ${
            drawMode
              ? "bg-[#f59e0b] text-black border-[#f59e0b]"
              : "bg-surface text-ink border-hairline hover:border-[#f59e0b]"
          }`}
          title="Drag on the map to select a GPS area"
        >
          <Crosshair className="h-3.5 w-3.5" /> {drawMode ? "Drawing… click map" : "Select area"}
        </button>
        <button
          onClick={onLocate}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-semibold bg-surface text-ink border border-hairline shadow-card hover:border-apple"
          title="Center on your GPS location"
        >
          <LocateFixed className="h-3.5 w-3.5 text-apple" /> GPS
        </button>
      </div>

      <div className="flex items-center gap-2 rounded-lg bg-surface border border-hairline shadow-card px-3 py-2">
        <Satellite className="h-3.5 w-3.5 text-[#f59e0b]" />
        <input type="range" min="1" max="12" value={month} onChange={(e) => onMonth(Number(e.target.value))}
          className="w-24 accent-[#f59e0b]" />
        <span className="text-[12px] font-semibold text-ink w-9">{MONTHS[month - 1]}</span>
        <button
          onClick={onToggleClim}
          className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-semibold border transition-colors ${
            climVisible ? "bg-[#f59e0b] text-black border-[#f59e0b]" : "bg-canvas text-ink-soft border-hairline"
          }`}
        >
          <Layers className="h-3 w-3" /> {climVisible ? "Climate ON" : "Climate OFF"}
        </button>
      </div>

      {climatology && climVisible && (
        <div className="rounded-lg bg-surface border border-hairline shadow-card px-3 py-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-semibold text-ink">Cyclone-prone zones</span>
            <span className="text-[10px] text-ink-soft/70">{climatologyLoaded ? `✓ ${MONTHS[month - 1]} climatology` : "loading…"}</span>
          </div>
          <div className="flex items-center gap-2">
            {(climatology.legend || []).map((l) => (
              <span key={l.t} className="flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded-sm inline-block" style={{ background: l.color }} />
                <span className="text-[10px] text-ink-soft">{l.t}</span>
              </span>
            ))}
            <span className="ml-auto text-[10px] text-ink-soft/80">hist. storm-track frequency</span>
          </div>
        </div>
      )}
    </div>
  );
}

function MapCore({ drawMode, setDrawMode, onSelect, month, onMonth,
                   climVisible, onToggleClim, climatology, climatologyLoaded }) {
  const map = useMap();

  const onLocate = () => {
    if (!navigator.geolocation) { alert("Geolocation is not available in this browser."); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        map.flyTo([lat, lng], 6, { duration: 1.2 });
        L.marker([lat, lng]).addTo(map).bindPopup("<b>Your GPS location</b>");
      },
      (err) => alert("GPS failed: " + err.message),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  return (
    <>
      <DrawRect drawMode={drawMode} onDone={onSelect} />
      <MapControls
        drawMode={drawMode} setDrawMode={setDrawMode} onLocate={onLocate}
        month={month} onMonth={onMonth}
        climVisible={climVisible} onToggleClim={onToggleClim}
        climatology={climatology} climatologyLoaded={climatologyLoaded}
      />
    </>
  );
}

function ClimOverlay({ climatology, visible }) {
  const map = useMap();
  useEffect(() => {
    if (!visible || !climatology) return;
    const overlay = L.imageOverlay(
      climatology.image,
      L.latLngBounds(L.latLng(-90, -180), L.latLng(90, 180)),
      { opacity: 0.35, interactive: false }
    );
    overlay.addTo(map);
    return () => { overlay.remove(); };
  }, [map, climatology, visible]);
  return null;
}

function MapCanvas({ drawMode, setDrawMode, onSelect, month, onMonth }) {
  const [climatology, setClimatology] = useState(null);
  const [climatologyLoaded, setClimatologyLoaded] = useState(false);
  const [climVisible, setClimVisible] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setClimatologyLoaded(false);
    fetchClimatology(month)
      .then((d) => { if (!cancelled) { setClimatology(d); setClimatologyLoaded(true); } })
      .catch(() => { if (!cancelled) setClimatology(null); });
    return () => { cancelled = true; };
  }, [month]);

  return (
    <MapContainer
      center={[14.2, 78.5]}
      zoom={4}
      minZoom={2}
      maxZoom={9}
      className="h-full w-full bg-canvas"
      zoomControl={true}
      attributionControl={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <ClimOverlay climatology={climVisible ? climatology : null}
                   visible={climVisible && Boolean(climatology)} />
      <MapCore
        drawMode={drawMode} setDrawMode={setDrawMode} onSelect={onSelect}
        month={month} onMonth={onMonth}
        climVisible={climVisible}
        onToggleClim={() => setClimVisible((v) => !v)}
        climatology={climatology} climatologyLoaded={climatologyLoaded}
      />
    </MapContainer>
  );
}

export default function SatelliteExplorer() {
  const [drawMode, setDrawMode] = useState(false);
  const [bounds, setBounds] = useState(null);
  const [month, setMonth] = useState(() => new Date().getMonth() + 1);
  const [hour, setHour] = useState(() => new Date().getHours());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const onSelect = (b) => {
    setBounds({
      west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth()
    });
    setResult(null);
  };

  const analyse = async () => {
    if (!bounds) return;
    setLoading(true); setError(null);
    try {
      setResult(await analyseGpsBox({ ...bounds, month, hour }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setDrawMode(false);
    }
  };

  const detected = result?.detection?.cyclone_detected ?? null;
  const plausible = result?.input_verified ?? result?.input_plausible ?? null;
  const prone = result?.prone || null;
  const pLevel = prone?.level || null;
  const pColor = levelStyle(pLevel).color;
  const cat = result?.intensity_category || "Inconclusive";
  const risk = result?.risk_level || "INCONCLUSIVE";
  const sat = result?.satellite || {};
  const preview = sat.frame_preview_b64 || null;
  const grad = result?.gradcam?.overlay_image_base64 || null;
  const season = prone?.season_note || "";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight text-ink">GPS satellite explorer</h1>
        <p className="text-[14px] text-ink-soft mt-1">
          Drag a box on the map (or use <span className="text-apple font-semibold">GPS</span> to jump to your
          location), then analyse the latest near-real-time thermal IR pass for that area.
        </p>
      </div>

      <div className="flex flex-col xl:flex-row gap-4">
        <div className="relative flex-1 h-[540px] rounded-2xl overflow-hidden border border-hairline shadow-card">
          <MapCanvas drawMode={drawMode} setDrawMode={setDrawMode} onSelect={onSelect} month={month}
                     onMonth={setMonth} />
          {bounds && (
            <div className="absolute bottom-3 left-3 z-[1000] rounded-lg bg-surface border border-hairline shadow-card px-3 py-2 text-[12px] text-ink">
              <div className="flex items-center gap-2">
                <MapPin className="h-3.5 w-3.5 text-[#f59e0b]" />
                <span className="font-semibold">
                  {bounds.south.toFixed(2)}° – {bounds.north.toFixed(2)}°N,&nbsp;
                  {bounds.west.toFixed(2)}° – {bounds.east.toFixed(2)}°E
                </span>
                <button
                  onClick={analyse}
                  disabled={loading}
                  className="ml-2 rounded-md px-3 py-1 text-[12px] font-semibold bg-[#f59e0b] text-black hover:brightness-110 disabled:opacity-60"
                >
                  {loading ? "Fetching live pass…" : "Analyse this area"}
                </button>
              </div>
            </div>
          )}
        </div>

        <aside className="w-full xl:w-[380px] space-y-4">
          <section className="rounded-2xl bg-surface border border-hairline shadow-card p-5">
            <div className="flex items-center justify-between border-b border-hairline pb-3 mb-3">
              <div>
                <h2 className="font-display text-[15px] font-semibold">Cyclone-proneness</h2>
                <p className="text-[12px] text-ink-soft">Climatology × live imagery verdict</p>
              </div>
              <TrendingUp className="h-5 w-5 text-[#f59e0b]" />
            </div>
            {!prone ? (
              <div className="text-[13px] text-ink-soft">
                {error ? (
                  <span className="text-[#ef4444] font-medium">{error}</span>
                ) : loading ? (
                  "Pulling the latest Himawari/GOES pass for this box…"
                ) : (
                  "Select an area on the map, then press “Analyse this area”."
                )}
              </div>
            ) : (
              <>
                <div className="flex items-center gap-3 mb-3">
                  <span
                    className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[12px] font-bold uppercase tracking-wide"
                    style={{ color: pColor, background: levelStyle(pLevel).ring }}
                  >
                    <ShieldAlert className="h-3.5 w-3.5" /> {pLevel}
                  </span>
                  <span className="text-[11px] text-ink-soft">
                    {Math.round((prone.climatology_index || 0) * 100)}% historical activity
                  </span>
                </div>
                <div className="h-2 rounded-full bg-canvas border border-hairline overflow-hidden mb-3">
                  <div className="h-full rounded-full transition-all" style={{
                    width: `${Math.round((prone.climatology_index || 0) * 100)}%`,
                    background: pColor
                  }} />
                </div>
                <p className="text-[12px] text-ink-soft leading-relaxed">{season}.</p>
                <div className="mt-1 text-[11px] text-ink-soft/70">{prone.basis}</div>
              </>
            )}
          </section>

          {result && (
            <section className="rounded-2xl bg-surface border border-hairline shadow-card p-5">
              <div className="flex items-center justify-between border-b border-hairline pb-3 mb-3">
                <div>
                  <h2 className="font-display text-[15px] font-semibold">Live image verdict</h2>
                  <p className="text-[12px] text-ink-soft">{sat.layer || "satellite layer"} · {sat.tiles_fetched || 0} tiles stitched</p>
                </div>
                <Satellite className="h-5 w-5 text-[#f59e0b]" />
              </div>
              {plausible === false ? (
                <div className="flex items-start gap-3 rounded-xl bg-red-500/10 border border-red-500/30 p-4">
                  <TriangleAlert className="h-4 w-4 text-[#ef4444] mt-0.5 shrink-0" />
                  <div className="text-[12px] text-ink-soft leading-relaxed">
                    <span className="font-semibold text-[#ef4444]">Input rejected by plausibility gate.</span>
                    {" "}{result.verification_note || result.input_note || ""}
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-3">
                    <span className={`rounded-lg px-3 py-1.5 text-[12px] font-bold ${
                      detected ? "bg-[#ef4444]/10 text-[#ef4444]" : "bg-[#10b981]/10 text-[#10b981]"
                    }`}>
                      {detected ? "CYCLONE SIGNATURE" : "No cyclone signature"}
                    </span>
                    <span className="text-[12px] text-ink-soft">category: <b className="text-ink">{cat}</b></span>
                  </div>
                  <div className="mt-3">
                    <Cell label="Detection confidence"
                          value={result.detection_confidence != null ? `${Math.round(result.detection_confidence * 100)}%` : "—"} />
                    <Cell label="Intensity category" value={cat} />
                    <Cell label="Est. wind" value={result.wind_speed_knots != null ? `${result.wind_speed_knots} kt` : "—"} />
                    <Cell label="RI risk" value={risk} accent={risk === "HIGH" ? "#ef4444" : undefined} />
                  </div>
                  {(preview || grad) && (
                    <div className="mt-4 grid grid-cols-2 gap-3">
                      {preview && (
                        <div>
                          <div className="text-[11px] font-semibold text-ink-soft uppercase tracking-wide mb-1.5">Live IR frame</div>
                          <img src={preview} alt="Live IR frame" className="rounded-xl border border-hairline w-full h-[110px] object-cover" />
                        </div>
                      )}
                      {result.gradcam?.heatmap_available && grad && (
                        <div>
                          <div className="text-[11px] font-semibold text-ink-soft uppercase tracking-wide mb-1.5">CNN attention</div>
                          <img src={grad} alt="Grad-CAM" className="rounded-xl border border-hairline w-full h-[110px] object-cover" />
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </section>
          )}

          <section className="rounded-2xl bg-surface border border-hairline shadow-card p-5">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-soft mb-2">How it works</div>
            <ul className="text-[12px] text-ink-soft space-y-2 leading-relaxed">
              <li>• <b className="text-ink">GPS box → GIBS WMTS</b> ("best" endpoint) pulls the <b>latest</b> thermal IR pass (Himawari-9 / GOES-18/16, Band 13) covering your box, stitching ~2 km tiles.</li>
              <li>• <b className="text-ink">CNN stack</b> classifies the stitched frame (detection + intensity) after the plausibility gate.</li>
              <li>• <b className="text-ink">Climatology</b> reads monthly IBTrACS storm-track frequency for the selected month.</li>
              <li>• Result blends both signals from real near-real-time data — not uploaded photos.</li>
            </ul>
            <div className="mt-3 text-[11px] text-ink-soft/70">
              Imagery: NASA GIBS (public, no key). Climatology: this repo's IBTrACS storm cache.
            </div>
            <div className="mt-3 flex items-center gap-2">
              <label className="text-[12px] text-ink-soft">Analysis hour (UTC)</label>
              <input type="number" min="0" max="23" value={hour} onChange={(e) => setHour(Number(e.target.value))}
                className="w-16 rounded-md bg-canvas border border-hairline px-2 py-1 text-[12px] text-ink" />
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}