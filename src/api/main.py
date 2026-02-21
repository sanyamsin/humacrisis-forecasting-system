from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logger import log

app = FastAPI(
    title="HumaCrisis Forecasting API",
    description="Multi-dimensional humanitarian crisis forecasting for Sub-Saharan Africa",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    country: str
    date: str
    ipc_phase: float
    total_events: int
    total_fatalities: int
    total_displaced: int
    forecast_horizon: Optional[int] = 3

class PredictionResponse(BaseModel):
    country: str
    forecast_date: str
    crisis_severity_index: float
    risk_level: str
    confidence: float
    recommendations: list[str]

class CountryStatus(BaseModel):
    country: str
    latest_date: str
    current_ipc: float
    current_events: int
    current_displaced: int
    crisis_severity: float
    trend: str

# ── Helper functions ────────────────────────────────────────
def get_risk_level(severity: float) -> str:
    if severity >= 0.75: return "CRITICAL"
    elif severity >= 0.60: return "HIGH"
    elif severity >= 0.40: return "MEDIUM"
    else: return "LOW"

def get_recommendations(risk_level: str, country: str) -> list[str]:
    base = {
        "CRITICAL": [
            f"Immediate emergency response needed in {country}",
            "Activate emergency food distribution pipelines",
            "Deploy rapid response humanitarian teams",
            "Alert OCHA and cluster coordination mechanisms",
            "Prepare population displacement contingency plans"
        ],
        "HIGH": [
            f"Scale up humanitarian operations in {country}",
            "Pre-position emergency supplies",
            "Strengthen early warning monitoring",
            "Coordinate with local authorities and NGOs"
        ],
        "MEDIUM": [
            f"Monitor situation closely in {country}",
            "Review and update contingency plans",
            "Strengthen community resilience programs",
            "Maintain regular data collection"
        ],
        "LOW": [
            f"Maintain regular monitoring in {country}",
            "Focus on development and resilience building",
            "Update risk assessments quarterly"
        ]
    }
    return base.get(risk_level, [])

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "HumaCrisis Forecasting API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/predict", "/status/{country}", "/countries", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/countries")
def get_countries():
    return {
        "countries": [
            {"code": "SOM", "name": "Somalia"},
            {"code": "SSD", "name": "South Sudan"},
            {"code": "ETH", "name": "Ethiopia"},
            {"code": "CAR", "name": "Central African Republic"},
            {"code": "COD", "name": "DR Congo"},
            {"code": "MLI", "name": "Mali"},
            {"code": "NER", "name": "Niger"},
            {"code": "TCD", "name": "Chad"},
            {"code": "MRT", "name": "Mauritania"},
            {"code": "SEN", "name": "Senegal"},
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Simulate model prediction
        severity = np.clip(
            (request.ipc_phase / 5) * 0.4 +
            np.log1p(request.total_events) / 10 * 0.3 +
            np.log1p(request.total_displaced) / 20 * 0.3,
            0, 1
        )

        risk_level = get_risk_level(float(severity))

        forecast_date = pd.Timestamp(request.date) + pd.DateOffset(
            months=request.forecast_horizon
        )

        return PredictionResponse(
            country=request.country,
            forecast_date=forecast_date.strftime("%Y-%m-%d"),
            crisis_severity_index=round(float(severity), 4),
            risk_level=risk_level,
            confidence=round(float(np.random.uniform(0.75, 0.95)), 2),
            recommendations=get_recommendations(risk_level, request.country)
        )
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{country}", response_model=CountryStatus)
def get_country_status(country: str):
    try:
        df = pd.read_csv("data/processed/merged_dataset.csv", parse_dates=["date"])
        country_df = df[df["country"] == country.upper()].sort_values("date")

        if country_df.empty:
            raise HTTPException(status_code=404, detail=f"Country {country} not found")

        latest = country_df.iloc[-1]
        prev = country_df.iloc[-2] if len(country_df) > 1 else latest

        severity = float(np.clip(
            (latest["ipc_phase"] / 5) * 0.4 +
            np.log1p(latest["total_events"]) / 10 * 0.3 +
            np.log1p(latest["total_displaced"]) / 20 * 0.3,
            0, 1
        ))

        prev_severity = float(np.clip(
            (prev["ipc_phase"] / 5) * 0.4 +
            np.log1p(prev["total_events"]) / 10 * 0.3 +
            np.log1p(prev["total_displaced"]) / 20 * 0.3,
            0, 1
        ))

        trend = "↑ Deteriorating" if severity > prev_severity + 0.02 else (
            "↓ Improving" if severity < prev_severity - 0.02 else "→ Stable"
        )

        return CountryStatus(
            country=country.upper(),
            latest_date=latest["date"].strftime("%Y-%m-%d"),
            current_ipc=float(latest["ipc_phase"]),
            current_events=int(latest["total_events"]),
            current_displaced=int(latest["total_displaced"]),
            crisis_severity=round(severity, 4),
            trend=trend
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))