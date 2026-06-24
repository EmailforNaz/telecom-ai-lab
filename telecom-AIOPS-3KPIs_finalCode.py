# ============================================================
# TELECOM AIOps SYSTEM - FULL PRODUCTION STYLE VERSION
# ============================================================
# This script simulates a real telecom AIOps pipeline
# It is designed to scale to 100,000+ cells
# and is structured like a real OpenShift microservice backend
# ============================================================


# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
# pandas → used for structured tabular data (like SQL tables)
# numpy → used for fast mathematical + random data generation
# sklearn → used for AI anomaly detection model

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# -----------------------------
# 2. CONFIGURATION SECTION
# -----------------------------
# This is where real telecom engineers define system scale

NUM_CELLS = 100000   # simulate large telecom network (100k cells)

np.random.seed(42)    # ensures same random data every run (reproducibility)


# -----------------------------
# 3. SIMULATE TELECOM NETWORK DATA
# -----------------------------
# In real life, this comes from:
# - 5G gNodeB
# - LTE eNodeB
# - OSS systems
# - Kafka streams
# - Prometheus metrics

# We simulate realistic KPI distributions

df = pd.DataFrame({

    # Create unique cell IDs like Cell_0, Cell_1, ..., Cell_99999
    "cell_name": [f"Cell_{i}" for i in range(NUM_CELLS)],

    # Latency in ms (normal distribution around 30ms)
    # Telecom meaning: delay in network response
    "latency_ms": np.random.normal(loc=30, scale=10, size=NUM_CELLS),

    # Packet loss percentage (small values normally)
    # Telecom meaning: dropped packets in transmission
    "packet_loss": np.random.normal(loc=0.5, scale=0.3, size=NUM_CELLS),

    # PRB utilization (%)
    # Telecom meaning: radio resource congestion level
    "prb_utilization": np.random.normal(loc=70, scale=15, size=NUM_CELLS)
})


# -----------------------------
# 4. CLEAN DATA (IMPORTANT)
# -----------------------------
# Real telecom data may contain invalid values
# So we enforce realistic constraints:

# Packet loss cannot be negative
df["packet_loss"] = df["packet_loss"].clip(lower=0)

# Latency cannot be below 1 ms
df["latency_ms"] = df["latency_ms"].clip(lower=1)

# PRB utilization must be between 0 and 100%
df["prb_utilization"] = df["prb_utilization"].clip(lower=0, upper=100)


# -----------------------------
# 5. HEALTH SCORE ENGINE
# -----------------------------
# This converts raw KPIs into a single "network health score"
# Higher score = better network performance

# Formula explanation:
# Start from 100 (perfect network)
# Subtract penalties based on KPIs

df["health_score"] = (
    100
    - (df["latency_ms"] * 0.5)        # latency impact
    - (df["packet_loss"] * 10)        # packet loss impact (very important KPI)
    - (df["prb_utilization"] * 0.2)   # congestion impact
)


# -----------------------------
# 6. STATUS CLASSIFICATION ENGINE
# -----------------------------
# Convert numeric score into human-readable telecom status

# Rules:
# >= 90 → HEALTHY
# >= 70 → DEGRADED
# < 70  → CRITICAL

df["status"] = np.where(
    df["health_score"] >= 90, "HEALTHY",
    np.where(df["health_score"] >= 70, "DEGRADED", "CRITICAL")
)


# -----------------------------
# 7. ANOMALY DETECTION (AI MODEL)
# -----------------------------
# This is machine learning part
# It detects unusual behavior WITHOUT rules

features = df[["latency_ms", "packet_loss", "prb_utilization"]]

# Isolation Forest:
# - +1 → normal behavior
# - -1 → anomaly (unusual cell behavior)

model = IsolationForest(
    contamination=0.05,   # assume 5% anomalies in network
    random_state=42
)

df["anomaly"] = model.fit_predict(features)


# -----------------------------
# 8. ROOT CAUSE ENGINE
# -----------------------------
# This explains WHY a cell is performing badly

df["latency_impact"] = df["latency_ms"] * 0.5
df["packet_loss_impact"] = df["packet_loss"] * 10
df["prb_impact"] = df["prb_utilization"] * 0.2

df["total_impact"] = (
    df["latency_impact"] +
    df["packet_loss_impact"] +
    df["prb_impact"]
)

# Convert each KPI into percentage contribution

df["latency_pct"] = df["latency_impact"] / df["total_impact"] * 100
df["packet_loss_pct"] = df["packet_loss_impact"] / df["total_impact"] * 100
df["prb_pct"] = df["prb_impact"] / df["total_impact"] * 100


# -----------------------------
# 9. RECOMMENDATION ENGINE
# -----------------------------
# This tells engineers what to do

def get_recommendation(row):

    # If cell is critical → investigate deeper
    if row["status"] == "CRITICAL":

        # Check if congestion is main issue
        if row["prb_utilization"] > 85:
            return "Increase capacity / reduce congestion (PRB issue)"

        # Check if packet loss is main issue
        elif row["packet_loss"] > 1:
            return "Check transport/backhaul (packet loss issue)"

        # Otherwise likely RF or hardware
        else:
            return "Investigate RF interference or hardware fault"

    # If degraded → warning state
    elif row["status"] == "DEGRADED":
        return "Monitor closely, early degradation detected"

    # Healthy case
    else:
        return "No action required"


df["recommendation"] = df.apply(get_recommendation, axis=1)


# -----------------------------
# 10. INCIDENT TICKET GENERATOR
# -----------------------------
# This simulates ServiceNow / Jira ticket creation

def create_ticket(row):

    return {
        "incident_title": f"{row['cell_name']} Network Issue",
        "severity": "P1" if row["status"] == "CRITICAL" else "P2",
        "status": row["status"],
        "health_score": round(row["health_score"], 2),
        "recommendation": row["recommendation"],
        "anomaly_flag": int(row["anomaly"])
    }


tickets = df.apply(create_ticket, axis=1)


# -----------------------------
# 11. NETWORK DASHBOARD SUMMARY
# -----------------------------
# This is what NOC engineers actually see

total = len(df)
critical = len(df[df["status"] == "CRITICAL"])
degraded = len(df[df["status"] == "DEGRADED"])
healthy = len(df[df["status"] == "HEALTHY"])

print("\n==============================")
print(" TELECOM AIOPS FINAL DASHBOARD")
print("==============================\n")

print("Total Cells:", total)
print("Healthy Cells:", healthy)
print("Degraded Cells:", degraded)
print("Critical Cells:", critical)

print("\nNetwork Health Distribution:")
print("Healthy %:", round(healthy/total*100, 2))
print("Degraded %:", round(degraded/total*100, 2))
print("Critical %:", round(critical/total*100, 2))


# Worst performing cells (top 5 worst)
worst_cells = df.sort_values("health_score").head(5)

print("\nTOP 5 WORST CELLS:")
print(worst_cells[["cell_name", "health_score", "status"]])


# -----------------------------
# 12. PRINT SAMPLE TICKETS
# -----------------------------
# Show first 3 incident tickets

print("\nSAMPLE INCIDENT TICKETS:\n")

for i in range(3):
    print(tickets.iloc[i])
    print("\n")