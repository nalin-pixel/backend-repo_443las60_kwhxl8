import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import math
import random

app = FastAPI(title="LaunchPad Live API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    fuel_type: Literal["RP-1", "Liquid Hydrogen", "Solid", "Methalox"]
    thrust_kN: float = Field(gt=0, description="Total liftoff thrust in kiloNewtons")
    payload_kg: float = Field(ge=0, description="Payload mass in kilograms")
    weather: Literal["Clear", "Windy", "Rain", "Storm"]


class SimulationResult(BaseModel):
    success: bool
    probability: float
    reasons: List[str]
    burn_time_s: float
    max_altitude_km: float
    delta_v_ms: float


@app.get("/")
def root():
    return {"message": "LaunchPad Live API running"}


@app.get("/api/options")
def get_options():
    return {
        "fuelTypes": ["RP-1", "Liquid Hydrogen", "Solid", "Methalox"],
        "weatherOptions": ["Clear", "Windy", "Rain", "Storm"],
        "thrustRange": {"min": 500, "max": 10000},
        "payloadRange": {"min": 100, "max": 50000},
    }


@app.post("/api/simulate", response_model=SimulationResult)
def simulate_launch(req: SimulationRequest):
    # Heuristic model for success probability
    # Thrust-to-weight ratio at liftoff: T/W ~ thrust_N / (mass_kg * g)
    g = 9.81
    structure_mass = max(10000.0, 0.15 * req.payload_kg + 15000)  # simplistic
    fuel_mass_factor = {
        "RP-1": 1.0,
        "Liquid Hydrogen": 0.85,
        "Solid": 1.2,
        "Methalox": 0.9,
    }[req.fuel_type]
    vehicle_mass = req.payload_kg + structure_mass * fuel_mass_factor

    thrust_N = req.thrust_kN * 1000
    tw = thrust_N / (vehicle_mass * g)

    # Base probability from T/W
    prob = max(0.0, min(1.0, (tw - 0.9) / 0.8))  # 0 at 0.9, ~1 at 1.7

    # Fuel reliability modifiers
    fuel_reliability = {
        "RP-1": 0.94,
        "Liquid Hydrogen": 0.9,
        "Solid": 0.85,
        "Methalox": 0.92,
    }[req.fuel_type]
    prob *= fuel_reliability

    # Weather penalties
    weather_penalty = {
        "Clear": 1.0,
        "Windy": 0.9,
        "Rain": 0.8,
        "Storm": 0.55,
    }[req.weather]
    prob *= weather_penalty

    # Payload penalty if very heavy relative to thrust
    mass_index = req.payload_kg / max(1.0, req.thrust_kN)
    if mass_index > 10:
        prob *= 0.7
    if mass_index > 20:
        prob *= 0.6

    # Clamp and add slight randomness for realism
    prob = max(0.05, min(0.98, prob * random.uniform(0.95, 1.05)))

    success = random.random() < prob

    # Simple performance estimates
    isp_map = {
        "RP-1": 300,
        "Liquid Hydrogen": 450,
        "Solid": 260,
        "Methalox": 360,
    }
    isp = isp_map[req.fuel_type]
    # Tsiolkovsky-like rough estimate: dv ~ Isp*g*ln(m0/m1); approximate ln term via T/W
    ln_term = max(1.05, min(1.8, 1.0 + (tw - 1.0) * 0.6))
    delta_v = isp * g * math.log(ln_term)

    burn_time = max(90.0, min(220.0, (vehicle_mass * g) / max(1.0, thrust_N) * 600))
    max_altitude = max(50.0, min(220.0, (delta_v / 9.81) * 10))

    reasons: List[str] = []
    if tw < 1.2:
        reasons.append("Low thrust-to-weight ratio")
    if req.weather in ("Rain", "Storm"):
        reasons.append("Adverse weather conditions")
    if req.fuel_type == "Solid":
        reasons.append("Solid motors less controllable")
    if req.payload_kg > 30000:
        reasons.append("Heavy payload")

    return SimulationResult(
        success=success,
        probability=round(prob, 3),
        reasons=reasons or (["Nominal conditions"] if success else ["Multiple minor risk factors"]),
        burn_time_s=round(burn_time, 1),
        max_altitude_km=round(max_altitude, 1),
        delta_v_ms=round(delta_v, 1),
    )


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
