"""
Pydantic v2 Schemas for the IBTrACS-driven CycloVision forecast API.
Requests accept a single tabular storm observation; responses carry the
three RandomForest outputs plus the flat fields the dashboard renders.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    application: str
    version: str
    models_loaded: bool
    mode: str
    device: str
    model_files: Optional[dict] = None


class ObservationInput(BaseModel):
    name: Optional[str] = Field(default=None, max_length=64)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    wind_knots: float = Field(..., ge=0.0, le=250.0)
    pressure_hpa: float = Field(..., ge=850.0, le=1055.0)
    storm_speed_knots: float = Field(..., ge=0.0, le=80.0)
    storm_direction_deg: float = Field(..., ge=0.0, le=360.0)
    month: int = Field(..., ge=1, le=12)
    hour: int = Field(..., ge=0, le=23)
    obs_num: int = Field(default=0, ge=0, le=100000)


class IntensityPrediction(BaseModel):
    predicted_wind_knots: float
    category: str
    category_confidence: Optional[float] = None


class FormationPrediction(BaseModel):
    probability_becomes_cyclone: float = Field(..., ge=0.0, le=1.0)
    verdict: str


class NextPosition(BaseModel):
    latitude: float
    longitude: float
    distance_km: float
    heading_deg: float
    horizon_hours: int = 6


class EffectItem(BaseModel):
    label: str
    detail: str
    severity: str  # LOW, MODERATE, HIGH, EXTREME


class OutlookPoint(BaseModel):
    horizon_hours: int
    projected_wind_knots: float
    category: str
    trend: str  # Intensifying, Stable, Weakening
    confidence_note: str


class StormTrackPoint(BaseModel):
    iso_time: str
    latitude: float
    longitude: float
    wind_knots: float
    category: str
    is_forecast: bool = False


class StormSummary(BaseModel):
    storm_id: str
    name: str
    basin: str
    current_category: str
    current_wind_knots: float
    latitude: float
    longitude: float
    last_updated: str
    track: List[StormTrackPoint]


class ForecastResponse(BaseModel):
    # typed model blocks
    intensity: IntensityPrediction
    formation: FormationPrediction
    next_position: NextPosition
    outlook: List[OutlookPoint]
    # flat fields reused by the existing dashboard
    storm_id: Optional[str] = None
    storm_name: Optional[str] = None
    timestamp: str
    cyclone_detected: bool
    detection_confidence: float
    intensity_category: str
    intensity_confidence: Optional[float] = None
    wind_speed_knots: float
    rapid_intensification_risk: float
    risk_level: str
    latitude: float
    longitude: float
    temporal_tendency: str
    alert_message: str
    suggested_action_radius_km: int
    source_mode: str
    disclaimer: str
    system_type: Optional[str] = None
    effects: List[EffectItem]
    intensity_class_probabilities: Optional[dict] = None
    trail: List[StormTrackPoint] = Field(default_factory=list)