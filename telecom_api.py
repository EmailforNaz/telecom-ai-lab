# ============================================================
# TELECOM AIOPS FASTAPI SERVICE
# Ready for Docker + OpenShift deployment
# ============================================================


# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# -----------------------------
# 2. CREATE FASTAPI APP
# -----------------------------
# This is your web server (like Flask but modern & faster)

app = FastAPI(title="Telecom AIOps API")


# -----------------------------
# 3. DEFINE INPUT FORMAT (REQUEST BODY)
# -----------------------------
# This defines what input API expects

class KPIInput(BaseModel):
    latency_ms: float
    packet_loss: float
    prb_utilization: float


# -----------------------------
# 4. LOAD / TRAIN SIMPLE ML MODEL
# -----------------------------
# In production: this would be pre-trained & loaded from file
# Here we simulate training once for simplicity

dummy_data = pd.DataFrame({
    "latency_ms": np.random.normal(30, 10, 1000),
    "packet_loss": np.random.normal(0.5, 0.3, 1000),
    "prb_utilization": np.random.normal(70, 15, 1000)
})

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(dummy_data)


# -----------------------------
# 5. HEALTH SCORE FUNCTION
# -----------------------------
def compute_health(latency, packet_loss, prb):

    # Start from perfect score
    score = 100

    # Apply telecom penalties
    score -= latency * 0.5
    score -= packet_loss * 10
    score -= prb * 0.2

    return round(score, 2)


# -----------------------------
# 6. STATUS CLASSIFICATION
# -----------------------------
def get_status(score):

    if score >= 90:
        return "HEALTHY"
    elif score >= 70:
        return "DEGRADED"
    else:
        return "CRITICAL"


# -----------------------------
# 7. ROOT CAUSE ENGINE
# -----------------------------
def root_cause(latency, packet_loss, prb):

    return {
        "latency_impact": latency * 0.5,
        "packet_loss_impact": packet_loss * 10,
        "prb_impact": prb * 0.2
    }


# -----------------------------
# 8. RECOMMENDATION ENGINE
# -----------------------------
def recommend(latency, packet_loss, prb):

    if prb > 85:
        return "Increase capacity (PRB congestion)"
    elif packet_loss > 1:
        return "Check transport/backhaul issue"
    elif latency > 50:
        return "Investigate RF latency issues"
    else:
        return "Network stable"


# -----------------------------
# 9. MAIN API ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: KPIInput):

    # Convert input
    latency = data.latency_ms
    packet_loss = data.packet_loss
    prb = data.prb_utilization

    # -------------------------
    # AI PIPELINE EXECUTION
    # -------------------------

    # 1. Health score
    score = compute_health(latency, packet_loss, prb)

    # 2. Status
    status = get_status(score)

    # 3. ML anomaly detection
    features = np.array([[latency, packet_loss, prb]])
    anomaly = model.predict(features)[0]

    # 4. Root cause
    cause = root_cause(latency, packet_loss, prb)

    # 5. Recommendation
    action = recommend(latency, packet_loss, prb)

    # -------------------------
    # RESPONSE (OUTPUT JSON)
    # -------------------------
    return {
        "health_score": score,
        "status": status,
        "anomaly": "ANOMALY" if anomaly == -1 else "NORMAL",
        "root_cause": cause,
        "recommendation": action
    }


# -----------------------------
# 10. HEALTH CHECK ENDPOINT
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "API_RUNNING"}