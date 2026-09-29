import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  Upload,
  Satellite,
  Compass,
  TrendingUp,
  ShieldAlert,
  Layers,
  Radio,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  X,
  Loader2,
  Flame,
  Wind,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  MapPin,
  Zap,
  Gauge,
  } from "lucide-react";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line
} from "recharts";
import TrackMap from "./components/TrackMap";
import { fetchHealth, fetchStorms, fetchAnalysis, forecastObservation, analyseSatellite, getFrameUrl } from "./api";
import SatelliteExplorer from "./SatelliteExplorer";

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "analyze", label: "Analyze", icon: Upload },
  { id: "gps", label: "GPS Explorer", icon: MapPin },
  { id: "satellite", label: "Satellite", icon: Satellite },
  { id: "track", label: "Track", icon: Compass },
  { id: "analytics", label: "Analytics", icon: TrendingUp },
  { id: "response", label: "Response", icon: ShieldAlert },
  { id: "architecture", label: "Architecture", icon: Layers }
];

const STORM_OPTIONS = [
  { key: "demo_cyclone_amphan", label: "AMPHAN", region: "Bay of Bengal" },
  { key: "demo_biparjoy", label: "BIPARJOY", region: "Arabian Sea" },
  { key: "demo_non_cyclone", label: "Baseline Convection", region: "Open ocean" }
];

const CLASS_RANGES = {
  "Depression": "17-27 kt",
  "Deep Depression": "28-33 kt",
  "Cyclonic Storm": "34-47 kt",
  "Severe Cyclonic Storm": "48-63 kt",
  "Very Severe Cyclonic Storm": "64-89 kt",
  "Extremely Severe Cyclonic Storm": "90-119 kt",
  "Super Cyclonic Storm": ">= 120 kt"
};

const SEVERITY_STYLE = {
  LOW: { dot: "bg-[#10b981]", label: "text-[#10b981]" },
  MODERATE: { dot: "bg-[#f59e0b]", label: "text-[#f59e0b]" },
  HIGH: { dot: "bg-[#f59e0b]", label: "text-[#f59e0b]" },
  EXTREME: { dot: "bg-[#ef4444]", label: "text-[#ef4444]" }
};

function RiskGauge({ value, riskLevel }) {
  const hasValue = value != null && !Number.isNaN(value);
  const pct = hasValue ? Math.min(100, Math.max(0, Math.round(value * 100))) : 0;
  const color = !hasValue ? "#3f4752" : pct >= 65 ? "#ef4444" : pct >= 35 ? "#f59e0b" : "#10b981";
  const radius = 60;
  const circumference = Math.PI * radius;
  const dash = !hasValue ? 0 : (pct / 100) * circumference;
  return (
    <div className="relative w-40 h-[120px] shrink-0">
      <svg width="160" height="96" viewBox="0 0 160 96" className="block">
        <path
          d="M20 88 a60 60 0 0 1 120 0"
          fill="none"
          stroke="#1e2228"
          strokeWidth="11"
          strokeLinecap="round"
        />
        <path
          d="M20 88 a60 60 0 0 1 120 0"
          fill="none"
          stroke={color}
          strokeWidth="11"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          style={{ transition: "stroke-dasharray 0.7s ease" }}
        />
      </svg>
      <div className="absolute inset-x-0 top-9 text-center">
        <div className="text-3xl font-display font-bold" style={{ color }}>
          {hasValue ? `${pct}%` : "—"}
        </div>
        <div className="text-[11px] font-semibold tracking-wide text-ink-soft uppercase mt-0.5">
          {!hasValue ? "No data" : riskLevel || (pct >= 65 ? "HIGH" : pct >= 35 ? "Moderate" : "Low")}
        </div>
      </div>
    </div>
  );
}

function TrendIcon({ trend }) {
  if (trend === "Intensifying")
    return <ArrowUpRight className="h-4 w-4 text-[#ef4444]" />;
  if (trend === "Weakening")
    return <ArrowDownRight className="h-4 w-4 text-[#10b981]" />;
  return <Minus className="h-4 w-4 text-[#474b53]" />;
}

function StatTile({ icon: Icon, label, value, sub, accent }) {
  return (
    <div className="bg-surface rounded-2xl border border-hairline shadow-card p-5 flex flex-col gap-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-faint">{label}</span>
        <Icon className="h-4 w-4" style={{ color: accent }} />
      </div>
      <div className="font-display text-2xl font-semibold leading-none text-ink">{value}</div>
      <div className="text-[11px] text-ink-soft leading-snug">{sub}</div>
    </div>
  );
}

function AlertToast({ analysis, regionLabel, onViewDetails }) {
  const [visible, setVisible] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [dragX, setDragX] = useState(0);
  const dragStart = useRef(null);
  const suppressClick = useRef(false);

  useEffect(() => {
    setVisible(true);
    setExpanded(false);
    setDragX(0);
  }, [analysis]);

  if (!analysis || !visible || !analysis.alert_message) return null;

  const risk = analysis.risk_level || "ADVISORY";
  const accent =
    riskLevelColor(risk);

  const dismiss = () => {
    setExpanded(false);
    setVisible(false);
  };

  const onPointerDown = (e) => {
    if (!e.isPrimary || e.button !== 0) return;
    suppressClick.current = false;
    dragStart.current = e.clientX;
  };

  const onPointerMove = (e) => {
    if (dragStart.current == null || !e.isPrimary) return;
    const delta = e.clientX - dragStart.current;
    if (Math.abs(delta) > 10) {
      suppressClick.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    setDragX(delta);
  };

  const onPointerUp = (e) => {
    if (dragStart.current == null || !e.isPrimary) return;
    if (Math.abs(e.clientX - dragStart.current) > 90) dismiss();
    setDragX(0);
    dragStart.current = null;
  };

  const cancelDrag = () => {
    dragStart.current = null;
    setDragX(0);
  };

  const summary = `Sustained winds ${analysis.wind_speed_knots != null ? `${analysis.wind_speed_knots}` : "—"} kt · ${analysis.intensity_category || "classification pending"} · formation ${analysis.rapid_intensification_risk != null ? `${Math.round(analysis.rapid_intensification_risk * 100)}` : "—"}% (${analysis.risk_level || "—"}) · Action radius ${analysis.suggested_action_radius_km != null ? `${analysis.suggested_action_radius_km}` : "—"} km · Tendency ${analysis.temporal_tendency || "—"}.`;

  return (
    <div className="fixed top-4 right-4 z-[60] w-[min(92vw,340px)] select-none" style={{ opacity: Math.max(0, 1 - Math.abs(dragX) / 260) }}>
      <div
        className={`bg-surface2/95 backdrop-blur-xl border border-hairline shadow-lift transition-transform duration-150 ${dragX ? "" : "transition-transform"}`}
        style={{
          transform: `translateX(${dragX}px)`,
          borderTop: `2px solid ${accent}`,
          touchAction: "pan-y"
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        role="status"
        aria-live="polite"
      >
        <div className="flex items-start gap-2.5 px-4 py-3">
          <span className="mt-1 h-2 w-2 rounded-full shrink-0 animate-pulse" style={{ background: accent }} />
          <button onClick={() => setExpanded((v) => !v)} className="flex-1 text-left min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono font-medium uppercase tracking-widest" style={{ color: accent }}>
                {risk === "HIGH" || risk === "EXTREME" ? `ALERT · ${risk} FORMATION RISK` : `ADVISORY · ${risk} FORMATION RISK`}
              </span>
            </div>
            <p className="text-[12px] leading-snug text-ink-soft mt-1 line-clamp-2">
              {analysis.alert_message}
            </p>
          </button>
          <button onClick={dismiss} aria-label="Dismiss alert" className="shrink-0 text-ink-faint hover:text-ink transition-colors">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        {expanded && (
          <div className="border-t border-hairline px-4 py-3 space-y-2.5">
            <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-ink-faint">
              <span>{analysis.storm_name || "Uploaded observation"}</span>
              <span>{regionLabel || "Region unspecified"}</span>
            </div>
            <p className="text-[12px] leading-relaxed text-ink-soft">{summary}</p>
            {analysis.effects?.length > 0 && (
              <ul className="space-y-1.5">
                {analysis.effects.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-[12px] text-ink-soft">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full shrink-0" style={{ background: accent }} />
                    <span><span className="text-ink font-medium">{f.label}</span> — {f.detail}</span>
                  </li>
                ))}
              </ul>
            )}
            <p className="text-[11px] text-ink-faint leading-relaxed pt-1 border-t border-hairline">
              Experimental decision support — verify with official IMD bulletins before action.
            </p>
          </div>
        )}

        <div className="h-[2px] bg-hairline">
          <div className="h-full" style={{ width: `${Math.min(100, (Math.abs(dragX) / 90) * 100)}%`, background: accent }} />
        </div>
      </div>
      <p className="mt-1.5 text-center text-[9px] font-mono uppercase tracking-widest text-ink-ghost">
        Swipe to dismiss · Tap for details
      </p>
    </div>
  );
}

function riskLevelColor(level) {
  if (level === "EXTREME") return "#ef4444";
  if (level === "HIGH") return "#ef4444";
  if (level === "MODERATE") return "#f59e0b";
  return "#10b981";
}

function Field({ label, value, onChange, placeholder, suffix }) {
  return (
    <label className="block">
      <span className="text-[11px] uppercase tracking-wide text-ink-faint font-semibold">{label}</span>
      <div className="mt-1.5 relative">
        <input
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          inputMode="decimal"
          className="w-full rounded-xl border border-hairline bg-surface px-4 py-2.5 text-[13px] text-ink placeholder:text-ink-faint outline-none focus:border-apple"
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-ink-faint pointer-events-none">{suffix}</span>
        )}
      </div>
    </label>
  );
}

const FIELD_DEFAULTS = {
  name: "Amphan",
  latitude: "13.7",
  longitude: "86.3",
  wind_knots: "130",
  pressure_hpa: "920",
  storm_speed_knots: "12",
  storm_direction_deg: "290",
  month: "5",
  hour: "12",
  obs_num: "0"
};

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedStorm, setSelectedStorm] = useState("demo_cyclone_amphan");
  const [channelMode, setChannelMode] = useState("ir_enhanced");
  const [frameIdx, setFrameIdx] = useState(5);
  const [health, setHealth] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [storms, setStorms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [obsForm, setObsForm] = useState(FIELD_DEFAULTS);
  const [analyzeMode, setAnalyzeMode] = useState("tabular");
  const [irFile, setIrFile] = useState(null);
  const [wvFile, setWvFile] = useState(null);
  const [irPreviewUrl, setIrPreviewUrl] = useState(null);
  const [wvPreviewUrl, setWvPreviewUrl] = useState(null);

  const isCustomForecast = analysis?.source_mode === "MODEL_FORECAST";
  const isUploadAnalysis = analysis?.source_mode === "UPLOAD_ANALYSIS";
  const isImageDriven = isUploadAnalysis;

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    runAnalysis(selectedStorm);
  }, [selectedStorm]);

  const loadInitialData = async () => {
    setLoading(true);
    const [h, s] = await Promise.all([fetchHealth(), fetchStorms()]);
    setHealth(h);
    setStorms(s);
    await runAnalysis(selectedStorm);
    setLoading(false);
  };

  const runAnalysis = async (stormKey) => {
    setLoading(true);
    setUploadError(null);
    const res = await fetchAnalysis(stormKey);
    setAnalysis(res);
    setLoading(false);
  };

  const setField = (key) => (e) => setObsForm((f) => ({ ...f, [key]: e.target.value }));

  const prefillFromStorm = (key) => {
    const slug = key.replace("demo_cyclone_", "").replace("demo_", "");
    const s = storms.find((x) => x.name.toLowerCase().replace(/\s+/g, "_") === slug);
    if (!s) return;
    const last = s.track[s.track.length - 1];
    setObsForm((f) => ({
      ...f,
      name: s.name,
      latitude: String(last.latitude),
      longitude: String(last.longitude),
      wind_knots: String(Math.round(last.wind_knots)),
      pressure_hpa: last.pressure_hpa != null ? String(Math.round(last.pressure_hpa)) : f.pressure_hpa
    }));
  };

  const handleForecast = async () => {
    const num = (v) => {
      const n = parseFloat(v);
      return Number.isFinite(n) ? n : null;
    };
    const v = {
      name: obsForm.name.trim() || "Untitled observation",
      latitude: num(obsForm.latitude),
      longitude: num(obsForm.longitude),
      wind_knots: num(obsForm.wind_knots),
      pressure_hpa: num(obsForm.pressure_hpa),
      storm_speed_knots: num(obsForm.storm_speed_knots) ?? 12,
      storm_direction_deg: num(obsForm.storm_direction_deg) ?? 290,
      month: parseInt(obsForm.month, 10) || 1,
      hour: parseInt(obsForm.hour, 10) || 0,
      obs_num: parseInt(obsForm.obs_num, 10) || 0
    };
    const checks = [
      [v.latitude, -90, 90, "Latitude must be between -90 and 90."],
      [v.longitude, -180, 180, "Longitude must be between -180 and 180."],
      [v.wind_knots, 0, 250, "Wind speed must be between 0 and 250 kt."],
      [v.pressure_hpa, 850, 1055, "Pressure must be between 850 and 1055 hPa."],
      [v.storm_speed_knots, 0, 80, "System speed must be between 0 and 80 kt."],
      [v.storm_direction_deg, 0, 360, "Heading must be between 0 and 360 deg."]
    ];
    const errors = [];
    for (const [val, lo, hi, msg] of checks) {
      if (val == null || Number.isNaN(val) || val < lo || val > hi) errors.push(msg);
    }
    if (obsForm.month && (parseInt(obsForm.month, 10) < 1 || parseInt(obsForm.month, 10) > 12)) errors.push("Month must be 1-12.");
    if (obsForm.hour && (parseInt(obsForm.hour, 10) < 0 || parseInt(obsForm.hour, 10) > 23)) errors.push("Hour must be 0-23.");
    if (errors.length) {
      setUploadError(errors.join(" "));
      return;
    }
    setRunning(true);
    setUploadError(null);
    try {
      const res = await forecastObservation(v);
      setAnalysis(res);
    } catch (err) {
      setUploadError(err.message || "Forecast failed. Check the backend is running.");
    } finally {
      setRunning(false);
    }
  };

  const getObservedSrc = () => {
    if (isImageDriven && analysis?.gradcam?.overlay_image_base64) {
      return analysis.gradcam.overlay_image_base64;
    }
    if (isImageDriven && irPreviewUrl) return irPreviewUrl;
    return getFrameUrl(selectedStorm, "ir_enhanced", frameIdx);
  };

  const setIrImage = (e) => {
    const file = e.target.files?.[0];
    setIrFile(file || null);
    if (irPreviewUrl) URL.revokeObjectURL(irPreviewUrl);
    setIrPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const setWvImage = (e) => {
    const file = e.target.files?.[0];
    setWvFile(file || null);
    if (wvPreviewUrl) URL.revokeObjectURL(wvPreviewUrl);
    setWvPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const handleImageAnalyse = async () => {
    if (!irFile) {
      setUploadError("Upload an IR channel satellite image to run the CNN analysis.");
      return;
    }
    const num = (v) => {
      const n = parseFloat(v);
      return Number.isFinite(n) ? n : null;
    };
    setRunning(true);
    setUploadError(null);
    try {
      const res = await analyseSatellite({
        irFile,
        wvFile: wvFile || undefined,
        name: obsForm.name.trim() || "Uploaded observation",
        latitude: num(obsForm.latitude) ?? 13.7,
        longitude: num(obsForm.longitude) ?? 86.3,
        stormSpeedKnots: num(obsForm.storm_speed_knots) ?? 12,
        stormDirectionDeg: num(obsForm.storm_direction_deg) ?? 290
      });
      setAnalysis(res);
    } catch (err) {
      setUploadError(err.message || "Image analysis failed. Ensure the CNN checkpoints are loaded and try again.");
    } finally {
      setRunning(false);
    }
  };

  const getSatelliteTabSrc = (mode, idx) =>
    getFrameUrl(selectedStorm, mode === "wv_enhanced" ? "wv_enhanced" : "ir_enhanced", idx);

  const probEntries = analysis?.intensity_class_probabilities
    ? Object.entries(analysis.intensity_class_probabilities)
        .map(([cat, p]) => ({ cat, range: CLASS_RANGES[cat] || "", prob: Math.round(p * 100) }))
        .sort((a, b) => b.prob - a.prob)
    : [];

  const chartData =
    analysis?.outlook?.length
      ? [
          { hour: "NOW", wind: analysis.wind_speed_knots || 0 },
          ...analysis.outlook.map((p) => ({ hour: `+${p.horizon_hours}h`, wind: p.projected_wind_knots }))
        ]
      : [];

  // Match by storm_id from the analysis (works for every cached storm); fall back to a single custom point.
  const trackStorm = storms.find((x) => x.storm_id && x.storm_id === analysis?.storm_id);
  const trackPoints = trackStorm
    ? trackStorm.track
    : analysis?.latitude != null
    ? [
        { latitude: analysis.latitude, longitude: analysis.longitude, wind_knots: analysis.wind_speed_knots || 0, iso_time: "Current" },
        ...(analysis.next_position
          ? [{ latitude: analysis.next_position.latitude, longitude: analysis.next_position.longitude,
               wind_knots: analysis.outlook?.[0]?.projected_wind_knots ?? analysis.wind_speed_knots ?? 0, is_forecast: true }]
          : [])
      ]
    : [];

  const activeStormMeta = STORM_OPTIONS.find((s) => s.key === selectedStorm);
  const probTop = probEntries[0] || null;

  const resetToDemo = () => {
    setIrFile(null);
    setWvFile(null);
    if (irPreviewUrl) URL.revokeObjectURL(irPreviewUrl);
    if (wvPreviewUrl) URL.revokeObjectURL(wvPreviewUrl);
    setIrPreviewUrl(null);
    setWvPreviewUrl(null);
    prefillFromStorm(selectedStorm);
    runAnalysis(selectedStorm);
  };

  const regionLabel = isUploadAnalysis
    ? "Satellite image analysis"
    : isCustomForecast
    ? "Custom observation"
    : `${activeStormMeta?.region || ""} · ${activeStormMeta?.label || ""}`;

  const intensityShortLabel = (cat) => {
    if (!cat) return "…";
    const words = cat.split(" ");
    return words.length > 2 ? words.slice(0, 2).join(" ") : cat;
  };

  return (
    <div className="mission-shell min-h-screen bg-page text-ink">
      <AlertToast
        analysis={analysis}
        regionLabel={regionLabel}
        onViewDetails={() => setActiveTab("response")}
      />
      <aside className="mission-sidebar" aria-label="Main navigation">
          <div className="px-4 pb-3 border-b border-hairline">
            <button onClick={() => setActiveTab("overview")} className="flex items-center gap-2.5 select-none">
              <span className="h-8 w-8 rounded-[10px] bg-apple text-[#08090a] flex items-center justify-center">
                <Radio className="h-4 w-4" />
              </span>
              <span className="flex flex-col items-start">
                <span className="font-display text-[16px] font-semibold tracking-tight text-ink leading-none">
                  Cyclo<span className="text-apple">Vision</span>
                </span>
                <span className="text-[9px] font-mono text-ink-faint tracking-widest uppercase mt-1">
                  Mission Control · SIH26070
                </span>
              </span>
            </button>
            <div className="mt-3 flex items-center justify-between px-2.5 py-1 bg-canvas border border-hairline rounded">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-apple-green animate-pulse" />
                <span className="text-[10px] font-mono text-ink-soft uppercase tracking-wider">
                  {health?.status === "demo_offline" ? "Offline demo" : health?.models_loaded ? "Models loaded" : "Connecting"}
                </span>
              </div>
              <span className="text-[10px] font-mono text-apple">{health?.device?.toUpperCase() || "CPU"}</span>
            </div>
          </div>

          <nav className="flex flex-col px-3 py-4 gap-1 flex-1 overflow-y-auto">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded text-[13px] font-medium whitespace-nowrap transition-colors text-left ${
                    active
                      ? "bg-canvas-dim text-ink border-l-2 border-apple"
                      : "text-ink-soft hover:bg-canvas hover:text-ink border-l-2 border-transparent"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="hidden md:flex items-center gap-3 shrink-0">
            <span className="flex items-center gap-1.5 text-[11px] font-semibold text-[#10b981]">
              <span className="h-1.5 w-1.5 rounded-full bg-[#10b981]" />
              {health?.device?.toUpperCase() || "CPU"}
            </span>
            <button
              onClick={() => ((isCustomForecast || isUploadAnalysis) ? resetToDemo() : runAnalysis(selectedStorm))}
              className="flex items-center gap-1.5 text-[12px] font-medium text-apple hover:bg-canvas px-3 py-1.5 rounded-full transition-colors"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              {(isCustomForecast || isUploadAnalysis) ? "Back to Demo" : "Refresh"}
            </button>
          </div>
      </aside>
      <div className="mission-main">
      <header className="sticky top-0 z-40 bg-page/95 backdrop-blur-xl">

        <div className="border-t border-hairline bg-surface/60">
          <div className="px-6 h-11 flex items-center gap-3 overflow-x-auto font-mono">
            <span className="text-[10px] font-medium uppercase tracking-widest text-ink-faint shrink-0">
              TARGET SYSTEM
            </span>
            {!isCustomForecast && !isUploadAnalysis ? (
              <div className="flex items-center gap-1.5">
                {STORM_OPTIONS.map((s) => {
                  const active = selectedStorm === s.key;
                  return (
                    <button
                      key={s.key}
                      onClick={() => {
                        prefillFromStorm(s.key);
                        setSelectedStorm(s.key);
                      }}
                      className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[12px] font-medium border whitespace-nowrap transition-colors ${
                        active
                          ? "bg-apple text-white border-apple"
                          : "bg-surface text-ink-soft border-hairline hover:text-ink"
                      }`}
                    >
                      {s.label}
                      <span className={`text-[10px] ${active ? "text-white/80" : "text-ink-faint"}`}>{s.region}</span>
                    </button>
                  );
                })}
              </div>
            ) : isUploadAnalysis ? (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[12px] font-semibold text-apple-green bg-apple-green-soft border border-apple-green/30 whitespace-nowrap">
                <Satellite className="h-3.5 w-3.5" />
                IMAGE ANALYSIS — {analysis?.storm_name || "Uploaded imagery"}
              </span>
            ) : (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[12px] font-semibold text-apple-green bg-apple-green-soft border border-apple-green/30 whitespace-nowrap">
                <CheckCircle2 className="h-3.5 w-3.5" />
                OBSERVATION FORECAST — {analysis?.storm_name || "Custom observation"}
              </span>
            )}
            <span className="ml-auto shrink-0 text-[10px] font-medium uppercase tracking-widest text-ink-faint">
              {(isCustomForecast || isUploadAnalysis) ? "" : `${activeStormMeta?.label || ""} · ${activeStormMeta?.region || ""}`}
            </span>
          </div>
        </div>
      </header>

      {/* ======= Disclaimer ======= */}
      <div className="bg-surface border-b border-hairline">
        <div className="max-w-6xl mx-auto px-5 py-2 flex items-center gap-2 text-[12px] text-ink-soft">
          <AlertTriangle className="h-4 w-4 text-[#f59e0b] shrink-0" />
          <p className="leading-snug">
            <strong className="text-ink">Decision-support prototype.</strong> Model output and projections are experimental — always verify with official IMD bulletins before action.
          </p>
        </div>
      </div>

      <main className="max-w-6xl mx-auto px-5 py-8">
        {/* ======= OVERVIEW ======= */}
        {activeTab === "overview" && (
          <div className="space-y-8">
            {/* Hero */}
            <section className="rounded-3xl bg-surface border border-hairline shadow-card overflow-hidden">
              <div className="px-8 pt-10 pb-8 sm:px-12 bg-gradient-to-b from-canvas-dim to-surface">
                <div className="flex items-center gap-2 text-[12px] font-semibold text-[#10b981] mb-4">
                  <span className="h-2 w-2 rounded-full bg-[#10b981]" />
                  AI MODELS LOADED — {health?.mode === "operational" ? "OPERATIONAL" : "READY"} · {health?.device?.toUpperCase() || "CPU"}
                </div>
                <h1 className="font-display text-4xl sm:text-5xl font-bold tracking-tight leading-[1.05] text-ink">
                  Tropical cyclone intelligence,
                  <br className="hidden sm:block" />
                  <span className="text-apple"> from a single observation.</span>
                </h1>
                <p className="mt-4 text-[15px] text-ink-soft leading-relaxed max-w-2xl">
                  Upload a single INSAT-3D satellite image or enter one best-track observation. CycloVision's CNN vision
                  stack extracts the current state from the imagery, then forecasts the storm's intensity across the IMD
                  categories, estimates cyclonic formation probability, projects the 24·48·72 hour track and intensity
                  outlook, and suggests coastal response actions.
                </p>
                <div className="mt-7 flex flex-wrap items-center gap-3">
                  <button
                    onClick={() => { setAnalyzeMode("image"); setActiveTab("analyze"); }}
                    className="flex items-center gap-2 bg-apple hover:bg-apple-hover text-white text-[14px] font-medium px-5 py-2.5 rounded-full shadow-lift transition-colors"
                  >
                    <Upload className="h-4 w-4" />
                    Analyze satellite imagery
                  </button>
                  <button
                    onClick={() => { setAnalyzeMode("tabular"); setActiveTab("analyze"); }}
                    className="flex items-center gap-2 text-apple text-[14px] font-medium px-4 py-2.5 hover:bg-canvas rounded-full transition-colors"
                  >
                    <Gauge className="h-4 w-4" />
                    Enter an observation
                  </button>
                  <button
                    onClick={() => {
                      setFrameIdx(5);
                      setActiveTab("satellite");
                    }}
                    className="flex items-center gap-2 text-apple text-[14px] font-medium px-4 py-2.5 hover:bg-canvas rounded-full transition-colors"
                  >
                    <Satellite className="h-4 w-4" />
                    View live demo imagery
                  </button>
                </div>
              </div>
            </section>

            {/* Stat tiles */}
            <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatTile
                icon={Gauge}
                label="Detection"
                value={
                  analysis
                    ? analysis.cyclone_detected == null && analysis.input_verified === false
                      ? "Rejected"
                      : analysis.cyclone_detected == null
                      ? "Unavailable"
                      : analysis.cyclone_detected
                      ? "Detected"
                      : "Not detected"
                    : "Loading…"
                }
                sub={analysis ? (analysis.input_verified === false ? "Input did not resemble satellite imagery" : analysis.detection_confidence != null ? `${Math.round(analysis.detection_confidence * 100)}% model confidence · ${isUploadAnalysis ? "CNN detector" : "RandomForest formation classifier"}` : "No backend data") : "Awaiting backend analysis…"}
                accent="#38bdf8"
              />
              <StatTile
                icon={Wind}
                label="Intensity"
                value={analysis?.input_verified === false ? "Rejected" : intensityShortLabel(analysis?.intensity_category)}
                sub={analysis?.input_verified === false ? "No classification issued for invalid input" : analysis?.wind_speed_knots != null ? `${analysis.wind_speed_knots} kt ${analysis.intensity_confidence != null ? `· ${Math.round(analysis.intensity_confidence * 100)}% conf` : ""}` : "No backend data"}
                accent="#38bdf8"
              />
              <StatTile
                icon={Flame}
                label="Formation risk"
                value={analysis?.input_verified === false ? "Rejected" : analysis?.rapid_intensification_risk != null ? `${Math.round(analysis.rapid_intensification_risk * 100)}%` : "—"}
                sub={`${analysis?.risk_level || "Unavailable"} · P(≥ 34 kt cyclonic strength)`}
                accent="#ef4444"
              />
              <StatTile
                icon={MapPin}
                label="Action radius"
                value={analysis?.suggested_action_radius_km != null ? `${analysis.suggested_action_radius_km} km` : "—"}
                sub="NDMA / SDMA precautionary coastal ring"
                accent="#f59e0b"
              />
            </section>

            {/* Split: satellite + chart */}
            <section className="grid lg:grid-cols-2 gap-5">
              <div className="bg-surface rounded-2xl border border-hairline shadow-card p-6 flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-display text-[15px] font-semibold">INSAT-3D observation</h3>
                    <p className="text-[12px] text-ink-soft">Enhanced IR frame at the forecast reference time</p>
                  </div>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                    {isUploadAnalysis ? "CNN T-0" : "Demo T-0"}
                  </span>
                </div>
                <div className="relative aspect-square w-full max-w-sm mx-auto rounded-xl overflow-hidden border border-hairline bg-[#0a0b0e] flex items-center justify-center">
                  <img src={getObservedSrc()} alt="Satellite observation" className="w-full h-full object-contain" />
                  {isUploadAnalysis && (
                    <span className="absolute top-3 left-3 bg-black/60 text-white text-[10px] font-semibold px-2 py-1 rounded-md backdrop-blur">
                      IR + ATTENTION
                    </span>
                  )}
                  {!isUploadAnalysis && (
                    <span className="absolute top-3 left-3 bg-black/60 text-white text-[10px] font-semibold px-2 py-1 rounded-md backdrop-blur">
                      IR 10.8 μm
                    </span>
                  )}
                </div>
                <p className="mt-4 text-[12px] text-ink-soft leading-relaxed text-center">
                  {analysis?.system_type || "RandomForest ensemble (intensity · formation · track)"}
                </p>
              </div>

              <div className="bg-surface rounded-2xl border border-hairline shadow-card p-6 flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-display text-[15px] font-semibold">Projected intensity</h3>
                    <p className="text-[12px] text-ink-soft">Model + expert-system outlook, next 72 hours</p>
                  </div>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                    Wind Speed
                  </span>
                </div>
                <div className="h-56">
                  {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 10, right: 8, left: -14, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e2228" />
                        <XAxis dataKey="hour" stroke="#474b53" fontSize={11} tickLine={false} axisLine={false} />
                        <YAxis stroke="#474b53" fontSize={11} tickLine={false} axisLine={false} domain={["auto", "auto"]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: "#14171C", border: "1px solid #1E2228", borderRadius: "10px", fontSize: "12px" }}
                          labelStyle={{ color: "#B4B7BD" }}
                          formatter={(v) => [`${v} kt`, "Wind speed"]}
                        />
                        <Line type="monotone" dataKey="wind" name="Wind" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 3, fill: "#38bdf8" }} activeDot={{ r: 5 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-[13px] text-ink-faint">
                      Loading projections…
                    </div>
                  )}
                </div>
                <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-canvas border border-hairline px-4 py-3">
                  <Zap className="h-4 w-4 text-[#f59e0b] mt-0.5 shrink-0" />
                  <div className="text-[12px] text-ink-soft leading-relaxed">
                    <span className="font-semibold text-ink">Current:</span> {analysis?.intensity_category || "…"} ({analysis?.wind_speed_knots ? `${analysis.wind_speed_knots} kt` : "…"}) ·{" "}
                    <span className="font-semibold text-ink">{analysis?.temporal_tendency || "…"}</span>. {analysis?.alert_message || ""}
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ======= ANALYZE ======= */}
        {activeTab === "analyze" && (
          <div className="space-y-8">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-ink">Analyze a storm</h1>
              <p className="text-[14px] text-ink-soft mt-1">
                Run the CNN vision stack on uploaded satellite imagery (image → tabular observation), or enter a best-track
                observation directly. CycloVision returns intensity, formation probability, impacts and a 72-hour outlook.
              </p>
            </div>

            {/* Mode toggle */}
            <div className="flex bg-canvas p-1 rounded-full border border-hairline w-fit">
              {[
                { id: "image", label: "From satellite imagery", icon: Satellite },
                { id: "tabular", label: "From observations", icon: Gauge }
              ].map((m) => {
                const Icon = m.icon;
                return (
                  <button
                    key={m.id}
                    onClick={() => setAnalyzeMode(m.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-medium transition-colors ${
                      analyzeMode === m.id ? "bg-surface text-ink shadow-card border border-hairline" : "text-ink-soft"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {m.label}
                  </button>
                );
              })}
            </div>

            {/* Image upload mode */}
            {analyzeMode === "image" && (
              <section className="bg-surface rounded-2xl border border-hairline shadow-card p-6">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline pb-4 mb-5">
                  <div>
                    <h2 className="font-display text-[15px] font-semibold">CNN vision analysis</h2>
                    <p className="text-[12px] text-ink-soft mt-0.5">
                      ResNet-18 detection + CNN-ConvLSTM intensity extract the tabular observation directly from the imagery;
                      the RandomForest chain then projects track, formation and outlook.
                    </p>
                  </div>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                    Insat-3D · IR 10.8 μm
                  </span>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <label className="block">
                    <span className="text-[11px] uppercase tracking-wide text-ink-faint font-semibold">IR channel (required)</span>
                    <div className="mt-1.5">
                      <label className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-hairline bg-canvas px-4 py-7 cursor-pointer hover:border-apple transition-colors">
                        {irPreviewUrl ? (
                          <img src={irPreviewUrl} alt="IR preview" className="w-28 h-28 object-contain rounded-lg bg-[#0a0b0e]" />
                        ) : (
                          <Upload className="h-6 w-6 text-ink-faint" />
                        )}
                        <span className="text-[12px] text-ink-soft">{irPreviewUrl ? "Change IR image" : "Choose an IR satellite frame…"}</span>
                        <input type="file" accept="image/*" className="hidden" onChange={setIrImage} />
                      </label>
                    </div>
                  </label>
                  <label className="block">
                    <span className="text-[11px] uppercase tracking-wide text-ink-faint font-semibold">WV channel (optional)</span>
                    <div className="mt-1.5">
                      <label className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-hairline bg-canvas px-4 py-7 cursor-pointer hover:border-apple transition-colors">
                        {wvPreviewUrl ? (
                          <img src={wvPreviewUrl} alt="WV preview" className="w-28 h-28 object-contain rounded-lg bg-[#0a0b0e]" />
                        ) : (
                          <Upload className="h-6 w-6 text-ink-faint" />
                        )}
                        <span className="text-[12px] text-ink-soft">{wvPreviewUrl ? "Change WV image" : "Optional — IR proxy used if omitted"}</span>
                        <input type="file" accept="image/*" className="hidden" onChange={setWvImage} />
                      </label>
                    </div>
                  </label>
                </div>

                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
                  <Field label="Storm name (optional)" value={obsForm.name} onChange={setField("name")} placeholder="e.g. Cyclone Amphan" />
                  <Field label="Latitude" value={obsForm.latitude} onChange={setField("latitude")} placeholder="13.7" suffix="°N" />
                  <Field label="Longitude" value={obsForm.longitude} onChange={setField("longitude")} placeholder="86.3" suffix="°E" />
                  <Field label="System speed" value={obsForm.storm_speed_knots} onChange={setField("storm_speed_knots")} placeholder="12" suffix="kt" />
                </div>

                <div className="flex flex-wrap items-center gap-2.5 mt-6">
                  <button
                    onClick={handleImageAnalyse}
                    disabled={running || !irFile}
                    className="flex-1 sm:flex-none flex items-center justify-center gap-2 bg-apple hover:bg-apple-hover disabled:opacity-40 disabled:cursor-not-allowed text-white text-[14px] font-medium px-6 py-3 rounded-full transition-colors"
                  >
                    {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Satellite className="h-4 w-4" />}
                    {running ? "Running CNN → forecast chain…" : "Analyze imagery"}
                  </button>
                  <button
                    onClick={resetToDemo}
                    disabled={running}
                    className="flex items-center gap-2 text-[13px] font-medium text-apple hover:bg-canvas px-4 py-3 rounded-full transition-colors"
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                    Load demo storm
                  </button>
                </div>

                {uploadError && (
                  <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-apple-red-soft border border-apple-red/30 px-4 py-3 text-[13px] text-[#ef4444]">
                    <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                    {uploadError}
                  </div>
                )}
              </section>
            )}

            {/* Observation form */}
            {analyzeMode === "tabular" && (
            <section className="bg-surface rounded-2xl border border-hairline shadow-card p-6">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline pb-4 mb-5">
                <div>
                  <h2 className="font-display text-[15px] font-semibold">Observation parameters</h2>
                  <p className="text-[12px] text-ink-soft mt-0.5">
                    Mapped to the exact engineered features the RandomForests were trained on (IBTrACS North Indian Ocean basin).
                  </p>
                </div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                  Position · motion · history
                </span>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <Field label="Storm name (optional)" value={obsForm.name} onChange={setField("name")} placeholder="e.g. Cyclone Amphan" />
                <Field label="Latitude" value={obsForm.latitude} onChange={setField("latitude")} placeholder="13.7" suffix="°N" />
                <Field label="Longitude" value={obsForm.longitude} onChange={setField("longitude")} placeholder="86.3" suffix="°E" />
                <Field label="Sustained wind" value={obsForm.wind_knots} onChange={setField("wind_knots")} placeholder="130" suffix="kt" />
                <Field label="Surface pressure" value={obsForm.pressure_hpa} onChange={setField("pressure_hpa")} placeholder="920" suffix="hPa" />
                <Field label="System speed" value={obsForm.storm_speed_knots} onChange={setField("storm_speed_knots")} placeholder="12" suffix="kt" />
                <Field label="System heading" value={obsForm.storm_direction_deg} onChange={setField("storm_direction_deg")} placeholder="290" suffix="°" />
                <Field label="Month" value={obsForm.month} onChange={setField("month")} placeholder="5" />
                <Field label="Hour (UTC)" value={obsForm.hour} onChange={setField("hour")} placeholder="12" />
                <Field label="Observation #" value={obsForm.obs_num} onChange={setField("obs_num")} placeholder="0" suffix="within storm" />
              </div>

              <div className="flex flex-wrap items-center gap-2.5 mt-6">
                <button
                  onClick={handleForecast}
                  disabled={running}
                  className="flex-1 sm:flex-none flex items-center justify-center gap-2 bg-apple hover:bg-apple-hover disabled:opacity-40 disabled:cursor-not-allowed text-white text-[14px] font-medium px-6 py-3 rounded-full transition-colors"
                >
                  {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                  {running ? "Running intensity · formation · track…" : "Run forecast chain"}
                </button>
                <button
                  onClick={resetToDemo}
                  disabled={running}
                  className="flex items-center gap-2 text-[13px] font-medium text-apple hover:bg-canvas px-4 py-3 rounded-full transition-colors"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                  Load demo observation
                </button>
              </div>

              {uploadError && (
                <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-apple-red-soft border border-apple-red/30 px-4 py-3 text-[13px] text-[#ef4444]">
                  <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                  {uploadError}
                </div>
              )}
            </section>
            )}

            {/* Report */}
            {loading ? (
              <section className="rounded-2xl border border-hairline bg-surface shadow-card p-10 flex flex-col items-center gap-3 text-ink-soft">
                <Loader2 className="h-6 w-6 animate-spin text-apple" />
                <p className="text-[13px]">Loading intelligence…</p>
              </section>
            ) : analysis ? (
              <section className="space-y-5">
                {/* Report header */}
                <div className="rounded-2xl bg-surface border border-hairline shadow-card p-6">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                        {isCustomForecast ? "Observation forecast report" : isUploadAnalysis ? "Image analysis report" : "Live demonstration report"}
                        {(isCustomForecast || isUploadAnalysis) && (
                          <span className="flex items-center gap-1 text-[#10b981]">
                            <CheckCircle2 className="h-3 w-3" /> {analysis.latitude}°N {analysis.longitude}°E
                          </span>
                        )}
                      </div>
                      <h2 className="font-display text-2xl font-bold text-ink mt-1">
                        {analysis.storm_name || "Custom observation"}
                      </h2>
                      <p className="text-[13px] text-ink-soft mt-0.5">
                        {analysis.system_type || analysis.intensity_category || "classification pending"}{analysis.intensity_confidence != null ? ` · ${Math.round(analysis.intensity_confidence * 100)}% confidence` : ""}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-[12px] font-semibold px-3 py-1.5 rounded-full ${
                          analysis.cyclone_detected == null && analysis.input_verified === false
                            ? "bg-apple-red-soft text-[#ef4444]"
                            : analysis.cyclone_detected
                            ? "bg-apple-amber-soft text-apple-amber"
                            : "bg-canvas-dim text-ink-soft"
                        }`}
                      >
                        {analysis.cyclone_detected == null && analysis.input_verified === false
                          ? "INPUT REJECTED — NOT SATELLITE IMAGERY"
                          : analysis.cyclone_detected == null
                          ? "NO RESULT"
                          : analysis.cyclone_detected
                          ? `CYCLONIC SYSTEM · ${Math.round((analysis.detection_confidence || 0) * 100)}%`
                          : "NON-CYCLONIC PATTERN"}
                      </span>
                      <span className="text-[12px] font-medium text-ink-soft px-3 py-1.5 rounded-full bg-canvas border border-hairline">
                        {analysis.temporal_tendency}
                      </span>
                    </div>
                  </div>

                  <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-apple-amber-soft border border-apple-amber/30 px-4 py-3 text-[12px] text-apple-amber leading-relaxed">
                      {isUploadAnalysis && analysis.input_verified === false ? (
                        <>
                          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                          <span>
                            <strong className="font-semibold">Input rejected.</strong> {analysis.verification_note}{" "}
                            No cyclone classification or forecast was produced for this image.
                          </span>
                        </>
                      ) : isUploadAnalysis ? (
                        <>
                          <Satellite className="h-4 w-4 shrink-0 mt-0.5" />
                          The current state (category, wind, pressure) was derived by the CNN stack directly from the
                          uploaded satellite imagery — the RandomForest chain then projected track, formation and outlook.
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                          The observation was entered as tabular data — the intensity posterior reflects this single set of
                          parameters, not imagery.
                        </>
                      )}
                    </div>
                </div>

                <div className="grid lg:grid-cols-5 gap-5">
                  {/* Type + RI */}
                  <div className="lg:col-span-2 space-y-5">
                    <div className="rounded-2xl bg-surface border border-hairline shadow-card p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-display text-[15px] font-semibold">Storm classification</h3>
                        <Wind className="h-4 w-4 text-apple" />
                      </div>
                      <div className="font-display text-[26px] font-bold leading-tight text-ink">
                        {analysis.input_verified === false ? "Inconclusive" : analysis.intensity_category}
                      </div>
                      <div className="mt-1 text-[13px] text-ink-soft">
                        {analysis.input_verified === false ? "Input rejected by satellite-imagery plausibility gate" : analysis.system_type}
                      </div>
                      <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 text-[13px]">
                        <div>
                          <dt className="text-[11px] uppercase tracking-wide text-ink-faint">Sustained winds</dt>
                          <dd className="font-semibold text-ink mt-0.5">{analysis.input_verified === false || analysis.wind_speed_knots == null ? "—" : `${analysis.wind_speed_knots} kt`}</dd>
                        </div>
                        <div>
                          <dt className="text-[11px] uppercase tracking-wide text-ink-faint">Confidence</dt>
                          <dd className="font-semibold text-ink mt-0.5">{analysis.input_verified === false || analysis.intensity_confidence == null ? "—" : `${Math.round(analysis.intensity_confidence * 100)}%`}</dd>
                        </div>
                        <div>
                          <dt className="text-[11px] uppercase tracking-wide text-ink-faint">Coastal action ring</dt>
                          <dd className="font-semibold text-ink mt-0.5">{analysis.suggested_action_radius_km != null ? `${analysis.suggested_action_radius_km} km` : "—"}</dd>
                        </div>
                        <div>
                          <dt className="text-[11px] uppercase tracking-wide text-ink-faint">Tendency</dt>
                          <dd className="font-semibold text-ink mt-0.5">{analysis.input_verified === false ? "—" : analysis.temporal_tendency}</dd>
                        </div>
                      </dl>
                    </div>

                    <div className="rounded-2xl bg-surface border border-hairline shadow-card p-6 flex items-center justify-between gap-4">
                      <div>
                        <h3 className="font-display text-[15px] font-semibold">Formation & intensification</h3>
                        <p className="text-[12px] text-ink-soft mt-0.5 leading-snug">
                          {analysis.input_verified === false
                            ? "No RI assessment was issued — the input was rejected by the plausibility gate."
                            : isUploadAnalysis
                            ? "Rapid-intensification probability from the CNN RI early-warning model (ve 30 kt / 24 h window)."
                            : "Probability the system is a genuine cyclone (reaches &ge; 34 kt sustained), from the RandomForest formation classifier."}
                        </p>
                      </div>
                      <RiskGauge value={analysis.rapid_intensification_risk} riskLevel={analysis.risk_level} />
                    </div>
                  </div>

                  {/* Effects + outlook */}
                  <div className="lg:col-span-3 space-y-5">
                    <div className="rounded-2xl bg-surface border border-hairline shadow-card p-6">
                      <h3 className="font-display text-[15px] font-semibold mb-4">Expected impacts</h3>
                      {analysis.input_verified === false ? (
                        <p className="text-[13px] text-ink-soft leading-relaxed">
                          No impact assessment produced — the uploaded image was rejected by the
                          satellite-imagery plausibility gate.
                        </p>
                      ) : (
                      <ul className="space-y-2.5">
                        {analysis.effects?.map((f, i) => {
                          const style = SEVERITY_STYLE[f.severity] || SEVERITY_STYLE.LOW;
                          return (
                            <li key={i} className="flex items-start gap-3">
                              <span className={`mt-1.5 h-2 w-2 rounded-full shrink-0 ${style.dot}`} />
                              <div>
                                <span className="text-[13px] font-semibold text-ink">{f.label}</span>
                                <span className="text-[13px] text-ink-soft"> — {f.detail}</span>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                      )}
                    </div>

                    <div className="rounded-2xl bg-surface border border-hairline shadow-card p-6">
                      <h3 className="font-display text-[15px] font-semibold mb-4">72-hour outlook</h3>
                      {analysis.input_verified === false ? (
                        <p className="text-[13px] text-ink-soft leading-relaxed">
                          No 24·48·72 h track projection issued — forecasting was skipped for the rejected input.
                        </p>
                      ) : (
                      <div className="grid sm:grid-cols-3 gap-3">
                        {analysis.outlook?.map((p) => (
                          <div key={p.horizon_hours} className="rounded-xl bg-canvas border border-hairline p-4">
                            <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
                              <span>+{p.horizon_hours} h</span>
                              <TrendIcon trend={p.trend} />
                            </div>
                            <div className="mt-2 font-display text-xl font-bold text-ink">
                              {Math.round(p.projected_wind_knots)} <span className="text-[13px] font-medium text-ink-soft">kt</span>
                            </div>
                            <div className="text-[12px] font-medium text-ink">{p.category}</div>
                            <div className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-faint">
                              <span className={`font-semibold ${p.trend === "Weakening" ? "text-[#10b981]" : p.trend === "Intensifying" ? "text-[#ef4444]" : "text-ink-soft"}`}>
                                {p.trend}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                      )}
                      <p className="mt-3 text-[11px] text-ink-faint leading-relaxed">
                        {analysis.outlook?.[0]?.confidence_note} Deterministic 6 h re-chaining of the track and intensity
                        models — experimental, not official IMD forecasts.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Advisory banner */}
                <div className="rounded-2xl border border-hairline shadow-card p-6 flex flex-col sm:flex-row items-start gap-4 bg-surface">
                  <span className="h-11 w-11 shrink-0 rounded-full bg-apple-amber-soft flex items-center justify-center">
                    <ShieldAlert className="h-5 w-5 text-apple-amber" />
                  </span>
                  <div>
                    <h3 className="font-display text-[15px] font-semibold">Recommended action (NDMA / SDMA)</h3>
                    <p className="text-[13px] text-ink-soft leading-relaxed mt-1">{analysis.alert_message}</p>
                    <p className="text-[11px] text-ink-faint mt-2">
                      Prepared by the CycloVision advisory engine. Always follow official IMD warnings and local disaster-management authorities.
                    </p>
                  </div>
                </div>
              </section>
            ) : null}
          </div>
        )}

        {/* ======= SATELLITE ======= */}
        {activeTab === "gps" && <SatelliteExplorer />}

          {activeTab === "satellite" && (
          <div className="space-y-6">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-ink">Satellite & Grad-CAM</h1>
              <p className="text-[14px] text-ink-soft mt-1">
                Inspect the 6-frame sequence across thermal IR, water vapour and the model's attention map.
              </p>
            </div>

            {isUploadAnalysis ? (
              <div className="grid lg:grid-cols-2 gap-5">
                <section className="bg-surface rounded-2xl border border-hairline shadow-card p-6">
                  <div className="flex items-center justify-between border-b border-hairline pb-4">
                    <div>
                      <h2 className="font-display text-[15px] font-semibold">CNN attention (Grad-CAM)</h2>
                      <p className="text-[12px] text-ink-soft mt-0.5">
                        {analysis?.gradcam?.target_class || "Cyclonic Pattern (Eye / Rainbands)"}
                      </p>
                    </div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                      Detector · layer4
                    </span>
                  </div>
                  {analysis?.gradcam?.overlay_image_base64 ? (
                    <div className="mt-5 relative aspect-square w-full max-w-md mx-auto rounded-2xl overflow-hidden border border-hairline bg-[#0a0b0e] flex items-center justify-center">
                      <img src={analysis.gradcam.overlay_image_base64} alt="Grad-CAM overlay" className="w-full h-full object-contain" />
                      <span className="absolute top-3 left-3 bg-black/60 backdrop-blur text-white text-[11px] font-semibold px-2.5 py-1 rounded-md">
                        IR + ATTENTION OVERLAY
                      </span>
                    </div>
                  ) : (
                    <div className="h-56 flex items-center justify-center text-ink-faint text-[13px]">
                      {analysis?.gradcam?.heatmap_available ? "Generating attention map…" : "Grad-CAM unavailable for this run."}
                    </div>
                  )}
                  <p className="mt-4 text-[12px] text-ink-soft leading-relaxed px-2">
                    {analysis?.gradcam?.explanation || "Regions that most influenced the CNN prediction."}
                  </p>
                </section>

                <section className="bg-surface rounded-2xl border border-hairline shadow-card p-6">
                  <div className="flex items-center justify-between border-b border-hairline pb-4">
                    <div>
                      <h2 className="font-display text-[15px] font-semibold">Uploaded channels</h2>
                      <p className="text-[12px] text-ink-soft mt-0.5">
                        The CNN ran on the single frame above (replicated into a 6-frame sequence).
                      </p>
                    </div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                      {analysis?.wv_source === "PROXY_FROM_IR" ? "WV = IR PROXY" : "WV = UPLOADED"}
                    </span>
                  </div>
                  <div className="mt-5 space-y-5">
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-ink-faint mb-2">IR 10.8 μm</p>
                      <div className="aspect-video w-full rounded-2xl overflow-hidden border border-hairline bg-[#0a0b0e] flex items-center justify-center">
                        {irPreviewUrl ? (
                          <img src={irPreviewUrl} alt="Uploaded IR" className="w-full h-full object-contain" />
                        ) : (
                          <img src={getFrameUrl(selectedStorm, "ir_enhanced", frameIdx)} alt="IR fallback" className="w-full h-full object-contain" />
                        )}
                      </div>
                    </div>
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-ink-faint mb-2">WV 6.8 μm · {analysis?.wv_source === "PROXY_FROM_IR" ? "IR-derived proxy" : "uploaded channel"}</p>
                      <div className="aspect-video w-full rounded-2xl overflow-hidden border border-hairline bg-[#0a0b0e] flex items-center justify-center">
                        {wvPreviewUrl ? (
                          <img src={wvPreviewUrl} alt="Uploaded WV" className="w-full h-full object-contain" />
                        ) : (
                          <img src={getFrameUrl(selectedStorm, "wv_enhanced", frameIdx)} alt="WV fallback" className="w-full h-full object-contain" />
                        )}
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            ) : (
            <section className="bg-surface rounded-2xl border border-hairline shadow-card p-6">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline pb-4">
                <div>
                  <h2 className="font-display text-[15px] font-semibold">Multi-spectral scrubber</h2>
                  <p className="text-[12px] text-ink-soft mt-0.5">
                    IR 10.8 μm · WV 6.8 μm · demo sequence for the selected storm
                  </p>
                </div>
                <div className="flex bg-canvas p-1 rounded-full border border-hairline">
                  {[ 
                    { id: "ir_enhanced", label: "Thermal IR" },
                    { id: "wv_enhanced", label: "Water Vapour" }
                  ].map((ch) => (
                    <button
                      key={ch.id}
                      onClick={() => setChannelMode(ch.id)}
                      className={`px-3.5 py-1.5 rounded-full text-[13px] font-medium transition-colors ${
                        channelMode === ch.id ? "bg-surface text-ink shadow-card border border-hairline" : "text-ink-soft"
                      }`}
                    >
                      {ch.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-6 flex flex-col items-center">
                <div className="relative w-full max-w-lg aspect-square rounded-2xl overflow-hidden border border-hairline bg-[#0a0b0e] flex items-center justify-center">
                  <img
                    src={getSatelliteTabSrc(channelMode, frameIdx)}
                    alt="Satellite frame"
                    className="w-full h-full object-contain"
                  />
                  <span className="absolute top-3 left-3 bg-black/60 backdrop-blur text-white text-[11px] font-semibold px-2.5 py-1 rounded-md">
                    FRAME {frameIdx + 1} / 6 · {channelMode.toUpperCase()}
                  </span>
                </div>

                <div className="w-full max-w-lg mt-6 space-y-2.5">
                  <input
                    type="range"
                    min="0"
                    max="5"
                    value={frameIdx}
                    onChange={(e) => setFrameIdx(parseInt(e.target.value))}
                    className="apple-range w-full"
                  />
                  <div className="flex justify-between text-[11px] font-medium text-ink-faint">
                    <span>T-15 h</span>
                    <span>T-12 h</span>
                    <span>T-9 h</span>
                    <span>T-6 h</span>
                    <span>T-3 h</span>
                    <span className="text-apple font-semibold">T-0 latest</span>
                  </div>
                </div>
              </div>
            </section>
            )}
          </div>
        )}

        {/* ======= TRACK ======= */}
        {activeTab === "track" && (
          <div className="space-y-6">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-ink">Best-track & forecast</h1>
              <p className="text-[14px] text-ink-soft mt-1">
                IBTrACS observed positions with the AI-projected position 6 hours ahead, North Indian Ocean.
              </p>
            </div>
            {isCustomForecast && (
              <div className="flex items-center gap-2.5 rounded-2xl bg-apple-amber-soft border border-apple-amber/30 px-5 py-4 text-[13px] text-apple-amber">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                Custom observation active — no IBTrACS track association. Historical tracks remain available below.
              </div>
            )}
            {isUploadAnalysis && (
              <div className="flex items-center gap-2.5 rounded-2xl bg-apple-amber-soft border border-apple-amber/30 px-5 py-4 text-[13px] text-apple-amber">
                <Satellite className="h-4 w-4 shrink-0" />
                Image analysis active — the RF projection is anchored on the uploaded observation's position. Historical tracks remain available below.
              </div>
            )}
            <section className="rounded-2xl bg-surface border border-hairline shadow-card overflow-hidden">
              <div className="px-6 pt-5 pb-3 flex items-center justify-between">
                <h2 className="font-display text-[15px] font-semibold">Storm path</h2>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">Dashed = AI +6 h</span>
              </div>
              <TrackMap track={trackPoints} />
            </section>
            <section className="rounded-2xl bg-surface border border-hairline shadow-card overflow-hidden">
              <div className="px-6 pt-5 pb-1 flex items-center justify-between">
                <h2 className="font-display text-[15px] font-semibold">Track table</h2>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">Observed · Amber = AI projected</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[13px]">
                  <thead>
                    <tr className="text-[11px] uppercase tracking-wide text-ink-faint border-b border-hairline">
                      <th className="py-3.5 pl-6 pr-4 font-semibold">Timestamp (UTC)</th>
                      <th className="px-4 py-3.5 font-semibold">Lat</th>
                      <th className="px-4 py-3.5 font-semibold">Lon</th>
                      <th className="px-4 py-3.5 font-semibold">Wind</th>
                      <th className="px-4 py-3.5 font-semibold">Category</th>
                      <th className="px-6 py-3.5 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(trackPoints.length ? trackPoints : null)
                      ?.map((pt, i) => (
                        <tr key={i} className={`border-b border-hairline ${pt.is_forecast ? "bg-apple-amber-soft" : ""}`}>
                          <td className="py-3 pl-6 pr-4 font-mono text-[12px] text-ink-soft">{pt.iso_time}</td>
                          <td className="px-4 py-3">{pt.latitude}° N</td>
                          <td className="px-4 py-3">{pt.longitude}° E</td>
                          <td className="px-4 py-3 font-semibold text-ink">{Math.round(pt.wind_knots)} kt</td>
                          <td className="px-4 py-3">{pt.category}</td>
                          <td className="px-6 py-3">
                            {pt.is_forecast ? (
                              <span className="text-[11px] font-semibold text-apple-amber">AI FORECAST +6 h</span>
                            ) : (
                              <span className="text-[11px] font-semibold text-ink-faint">OBSERVED (IBTrACS)</span>
                            )}
                          </td>
                        </tr>
                      )) || (
                        <tr>
                          <td colSpan="6" className="py-8 text-center text-ink-faint text-[13px]">
                            Loading track records…
                          </td>
                        </tr>
                      )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* ======= ANALYTICS ======= */}
        {activeTab === "analytics" && (
          <div className="space-y-6">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-ink">Ensemble posterior</h1>
              <p className="text-[14px] text-ink-soft mt-1">
                The intensity RandomForest's per-tree votes across the 7 IMD North Indian Ocean categories.
              </p>
            </div>
            <div className="grid lg:grid-cols-3 gap-5">
              <section className="lg:col-span-2 rounded-2xl bg-surface border border-hairline shadow-card p-6">
                {probEntries.length > 0 && (
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="font-display text-[15px] font-semibold">Category probability</h2>
                    <span className="text-[12px] font-semibold text-apple">{probTop?.cat} · {probTop?.prob}%</span>
                  </div>
                )}
                {analysis?.input_verified === false ? (
                  <div className="h-40 flex items-center justify-center text-ink-faint text-[13px] text-center px-6">
                    No posterior issued — input rejected by plausibility gate.
                  </div>
                ) : probEntries.length > 0 ? (
                  <div className="space-y-4">
                    {probEntries.map((item, i) => (
                      <div key={i} className="space-y-1.5">
                        <div className="flex justify-between text-[13px]">
                          <span className="text-ink">
                            {item.cat}
                            {item.range && <span className="text-ink-faint"> · {item.range}</span>}
                          </span>
                          <span className="font-semibold text-apple">{item.prob}%</span>
                        </div>
                        <div className="w-full h-2.5 rounded-full bg-canvas overflow-hidden">
                          <div
                            className="h-full rounded-full bg-apple transition-all"
                            style={{ width: `${item.prob}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-40 flex items-center justify-center text-ink-faint text-[13px]">
                    Loading posterior…
                  </div>
                )}
              </section>

              <section className="rounded-2xl bg-surface border border-hairline shadow-card p-6 flex flex-col items-center justify-center gap-3">
                <h2 className="font-display text-[15px] font-semibold self-start">Formation probability</h2>
                <RiskGauge value={analysis?.input_verified === false ? null : analysis?.rapid_intensification_risk} riskLevel={analysis?.risk_level} />
                <p className="text-[11px] text-ink-faint text-center leading-relaxed">
                  RandomForest classifier — P(reaches &ge; 34 kt cyclonic strength) on the North Indian Ocean basin.
                </p>
              </section>
            </div>
          </div>
        )}

        {/* ======= RESPONSE ======= */}
        {activeTab === "response" && (
          <div className="space-y-6">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-ink">Response & decision support</h1>
              <p className="text-[14px] text-ink-soft mt-1">
                Automated early-warning recommendations synthesized from Models 1–3.
              </p>
            </div>
            <section className="rounded-2xl bg-surface border border-hairline shadow-card p-6">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline pb-4 mb-6">
                <div>
                  <h2 className="font-display text-[15px] font-semibold">NDMA / SDMA action matrix</h2>
                  <p className="text-[12px] text-ink-soft mt-0.5">For {analysis?.intensity_category || "the active system"} — {analysis?.risk_level || "…"} RI risk</p>
                </div>
                <ShieldAlert className="h-5 w-5 text-[#f59e0b]" />
              </div>

              <div className="grid md:grid-cols-2 gap-5">
                <div className="rounded-2xl bg-apple-amber-soft border border-apple-amber/30 p-5">
                  <div className="text-[12px] font-semibold uppercase tracking-wide text-apple-amber mb-2.5">Rapid intensification protocol</div>
                  <ul className="text-[13px] text-ink-soft space-y-2">
                    <li>• Advance evacuation alerts by 18 h along high-risk coastal belts.</li>
                    <li>• Pre-position NDRF / SDRF search-and-rescue teams in elevated staging zones.</li>
                    <li>• Activate 15-minute INSAT-3DR rapid-scan telemetry.</li>
                  </ul>
                </div>
                <div className="rounded-2xl bg-canvas border border-hairline p-5">
                  <div className="text-[12px] font-semibold uppercase tracking-wide text-apple mb-2.5">Port & maritime advisory</div>
                  <ul className="text-[13px] text-ink-soft space-y-2">
                    <li>• Issue Great Danger Signal (No. 8 / 9 / 10) at regional major ports.</li>
                    <li>• Suspend coastal trawling and deep-sea fishing operations.</li>
                    <li>• Secure cranes, floating docks and container stacks in harbours.</li>
                  </ul>
                </div>
              </div>

              <div className="mt-5 flex items-start gap-3 rounded-2xl bg-surface border border-hairline p-5">
                <Zap className="h-4 w-4 text-[#f59e0b] mt-0.5 shrink-0" />
                <div className="text-[13px] text-ink-soft leading-relaxed">
                  <span className="font-semibold text-ink">Current advisory:</span> {analysis?.alert_message || "Awaiting analysis."}
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ======= ARCHITECTURE ======= */}
        {activeTab === "architecture" && (
          <div className="space-y-6">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-ink">System architecture</h1>
              <p className="text-[14px] text-ink-soft mt-1">Designed for Smart India Hackathon 2026 · SIH26070.</p>
            </div>
            <section className="rounded-2xl bg-surface border border-hairline shadow-card p-6 space-y-5">
              <div className="rounded-2xl bg-canvas border border-hairline px-5 py-4 text-[13px] font-mono text-ink-soft overflow-x-auto">
                INSAT imagery → CNN (ResNet-18 detection / CNN-ConvLSTM intensity / RI) → tabular observation → IBTrACS RandomForest (intensity · formation · track) → FastAPI → React dashboard
              </div>
              <div className="grid md:grid-cols-3 gap-5">
                {[
                  {
                    title: "Vision · CNN (uploaded imagery)",
                    body: "ResNet-18 detector classifies cyclonic vs non-cyclonic patterns; CNN+ConvLSTM intensity model classifies the IMD category; Rapid-Intensification model adds a 24 h RI risk. Grad-CAM explains the attention."
                  },
                  {
                    title: "Model 1 · Intensity",
                    body: "RandomForest regressor (300 trees, depth 12) over 11 engineered inputs — position, pressure, motion and history deltas. Test MAE 3.9 kt, R² 0.90 vs IBTrACS sustained winds."
                  },
                  {
                    title: "Model 2 · Formation",
                    body: "RandomForest classifier (300 trees, depth 8, class-balanced) predicting P(reaches ≥ 34 kt cyclonic strength). Test accuracy 0.75 across North Indian Ocean storms."
                  }
                ].map((c) => (
                  <div key={c.title} className="rounded-2xl bg-surface border border-hairline p-5">
                    <div className="font-semibold text-ink text-[14px] mb-1.5">{c.title}</div>
                    <p className="text-[13px] text-ink-soft leading-relaxed">{c.body}</p>
                  </div>
                ))}
              </div>
              <div className="rounded-2xl bg-surface border border-hairline p-5">
                <div className="font-semibold text-ink text-[14px] mb-1.5">Model 3 · Track</div>
                <p className="text-[13px] text-ink-soft leading-relaxed">Multi-output RandomForest projecting the 6 h next position (lat + lon), re-chained to 24·48·72 h outlook. Test MAE ≈ 16.3 km lat / 18.1 km lon.</p>
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="border-t border-hairline bg-surface/70">
        <div className="max-w-6xl mx-auto px-5 py-5 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-ink-faint">
          <span>CycloVision · Team La Squadra · SIH26070 Disaster Management</span>
          <span>Experimental prototype — not an official IMD or government system.</span>
        </div>
      </footer>
      </div>
    </div>
  );
}