const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const STATIC_BASE = `${API_BASE}/static`;

export function getFrameUrl(stormKey, channel, frameIdx = 5) {
  const suffix = channel === "wv_enhanced" ? "wv_enhanced" : "ir_enhanced";
  return `${STATIC_BASE}/demo/${stormKey}/frame_${frameIdx}_${suffix}.png`;
}

export function getRawFrameUrl(stormKey, channel, frameIdx = 5) {
  const suffix = channel === "wv" ? "wv" : "ir";
  return `${STATIC_BASE}/demo/${stormKey}/frame_${frameIdx}_${suffix}.png`;
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using fallback:", err);
    return {
      status: "demo_offline",
      application: "CycloVision (Client Offline Mode)",
      models_loaded: true,
      mode: "demo",
      device: "Browser Fallback"
    };
  }
}

export async function fetchStorms() {
  try {
    const res = await fetch(`${API_BASE}/api/storms`);
    if (!res.ok) throw new Error("Failed to fetch storms");
    return await res.json();
  } catch (err) {
    console.warn("Using offline storm catalog:", err);
    return [
      {
        storm_id: "2020137N10087",
        name: "AMPHAN",
        basin: "NI",
        current_category: "Super Cyclonic Storm",
        current_wind_knots: 140.0,
        latitude: 13.7,
        longitude: 86.3,
        last_updated: "2020-05-18 12:00:00",
        track: [
          { iso_time: "2020-05-16 12:00:00", latitude: 10.7, longitude: 86.5, wind_knots: 40.0, category: "Cyclonic Storm", is_forecast: false },
          { iso_time: "2020-05-17 06:00:00", latitude: 11.4, longitude: 86.2, wind_knots: 50.0, category: "Severe Cyclonic Storm", is_forecast: false },
          { iso_time: "2020-05-17 18:00:00", latitude: 12.5, longitude: 86.4, wind_knots: 75.0, category: "Very Severe Cyclonic Storm", is_forecast: false },
          { iso_time: "2020-05-18 06:00:00", latitude: 13.4, longitude: 86.5, wind_knots: 115.0, category: "Extremely Severe Cyclonic Storm", is_forecast: false },
          { iso_time: "2020-05-18 12:00:00", latitude: 13.7, longitude: 86.3, wind_knots: 140.0, category: "Super Cyclonic Storm", is_forecast: false }
        ]
      },
      {
        storm_id: "2023157N12066",
        name: "BIPARJOY",
        basin: "NI",
        current_category: "Extremely Severe Cyclonic Storm",
        current_wind_knots: 90.0,
        latitude: 18.1,
        longitude: 67.8,
        last_updated: "2023-06-11 00:00:00",
        track: [
          { iso_time: "2023-06-06 06:00:00", latitude: 11.5, longitude: 66.0, wind_knots: 35.0, category: "Cyclonic Storm", is_forecast: false },
          { iso_time: "2023-06-08 06:00:00", latitude: 14.0, longitude: 66.0, wind_knots: 75.0, category: "Very Severe Cyclonic Storm", is_forecast: false },
          { iso_time: "2023-06-11 00:00:00", latitude: 18.1, longitude: 67.8, wind_knots: 90.0, category: "Extremely Severe Cyclonic Storm", is_forecast: false }
        ]
      }
    ];
  }
}

export async function fetchAnalysis(stormKey = "demo_cyclone_amphan") {
  try {
    const res = await fetch(`${API_BASE}/api/demo/storm/${stormKey}`);
    if (!res.ok) throw new Error("Failed to fetch analysis");
    return await res.json();
  } catch (err) {
    console.warn("Backend unavailable — analysis has no data:", err);
    return {
      storm_id: null,
      storm_name: null,
      timestamp: new Date().toISOString(),
      cyclone_detected: null,
      detection_confidence: null,
      intensity_category: null,
      intensity_confidence: null,
      wind_speed_knots: null,
      rapid_intensification_risk: null,
      risk_level: null,
      temporal_tendency: null,
      alert_message: null,
      suggested_action_radius_km: null,
      latitude: null,
      longitude: null,
      source_mode: "DEMO_OFFLINE",
      intensity_class_probabilities: null,
      system_type: null,
      effects: null,
      outlook: null,
      trail: null,
      disclaimer: null
    };
  }
}

export async function forecastObservation(obs) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obs)
    });
  } catch (err) {
    console.error("Forecast network error:", err);
    throw new Error(
      "Cannot reach the forecast service at " + API_BASE + ". " +
      "Start the CycloVision backend (uvicorn on port 8000) and try again."
    );
  }

  if (!res.ok) {
    let detail = `Forecast failed with status ${res.status}`;
    try {
      const err = await res.json();
      if (err && err.detail) detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch {
      /* ignore parse failure */
    }
    throw new Error(detail);
  }

  return await res.json();
}

export async function analyseSatellite({ irFile, wvFile, name, latitude, longitude, stormSpeedKnots, stormDirectionDeg }) {
  const form = new FormData();
  form.append("ir_image", irFile);
  if (wvFile) form.append("wv_image", wvFile);
  if (name) form.append("name", name);
  form.append("latitude", String(latitude));
  form.append("longitude", String(longitude));
  if (stormSpeedKnots != null) form.append("storm_speed_knots", String(stormSpeedKnots));
  if (stormDirectionDeg != null) form.append("storm_direction_deg", String(stormDirectionDeg));

  let res;
  try {
    res = await fetch(`${API_BASE}/api/analyse`, { method: "POST", body: form });
  } catch (err) {
    console.error("Analyse network error:", err);
    throw new Error(
      "Cannot reach the vision service at " + API_BASE + ". " +
      "Start the CycloVision backend (uvicorn on port 8000) and try again."
    );
  }

  if (!res.ok) {
    let detail = `Image analysis failed with status ${res.status}`;
    try {
      const err = await res.json();
      if (err && err.detail) detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch {
      /* ignore parse failure */
    }
    throw new Error(detail);
  }

  return await res.json();
}

export async function analyseGpsBox({ west, south, east, north, month, hour }) {
  const qs = new URLSearchParams({
    west: String(west), south: String(south), east: String(east), north: String(north)
  });
  if (month != null) qs.set("month", String(month));
  if (hour != null) qs.set("hour", String(hour));

  let res;
  try {
    res = await fetch(`${API_BASE}/api/satellite/analyse?${qs}`);
  } catch (err) {
    console.error("GPS analyse network error:", err);
    throw new Error(
      "Cannot reach the satellite service at " + API_BASE + ". " +
      "Start the CycloVision backend (uvicorn on port 8000) and try again."
    );
  }
  if (!res.ok) {
    let detail = `Satellite analysis failed with status ${res.status}`;
    try {
      const err = await res.json();
      if (err && err.detail) detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch {
      /* ignore parse failure */
    }
    throw new Error(detail);
  }
  return await res.json();
}

export async function fetchClimatology(month) {
  const res = await fetch(`${API_BASE}/api/climatology?month=${month}`);
  if (!res.ok) {
    let detail = `Climatology failed with status ${res.status}`;
    try {
      const err = await res.json();
      if (err && err.detail) detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return await res.json();
}