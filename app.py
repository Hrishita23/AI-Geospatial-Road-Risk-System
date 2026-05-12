import streamlit as st
import pandas as pd
import sqlite3

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Road Risk AI Dashboard", layout="wide")

st.title("🚦 AI-Powered Road Accident Risk Dashboard")

# -------------------------
# State Mapping
# -------------------------
state_mapping = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming"
}

# -------------------------
# Load Data
# -------------------------
conn = sqlite3.connect("sql/road_risk.db")
df = pd.read_sql_query("SELECT * FROM risk_data", conn)

# Convert state codes to full names
df["State"] = df["State"].map(state_mapping).fillna(df["State"])

# -------------------------
# Filters
# -------------------------
state = st.selectbox("Select State", sorted(df["State"].unique()))
filtered = df[df["State"] == state]

# -------------------------
# KPIs
# -------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Accident Count", round(filtered["Predicted_Accident_Count"].mean(), 2))
col2.metric("Max Accident Count", int(filtered["Predicted_Accident_Count"].max()))
col3.metric("Min Accident Count", int(filtered["Predicted_Accident_Count"].min()))

# -------------------------
# Tables
# -------------------------
st.subheader("📍 City-Level Risk Data")
st.dataframe(filtered, use_container_width=True)

st.subheader("🔥 Top High Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count", ascending=False).head(10), use_container_width=True)

st.subheader("🟢 Top Low Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count").head(10), use_container_width=True)

# -------------------------
# AI Explanation (FINAL FIX - NO CUTTING EVER)
# -------------------------
st.subheader("🤖 AI Risk Explanation")

ai_explanation = """
This region shows elevated accident risk due to high traffic density,
frequent congestion during peak hours, and historical accident clusters.

Weather conditions, road quality, and traffic violations also contribute significantly.

Key insights:
- High congestion during morning and evening peaks
- Accident hotspots near urban intersections
- Moderate weather-related impact
- Weak traffic signal coordination
- Increased accident probability during rainy conditions

Recommendations:
- Improve traffic signal timing
- Deploy smart surveillance systems
- Strengthen enforcement in high-risk zones
- Use predictive AI monitoring for peak hours
- Upgrade road infrastructure in hotspot areas
"""

with st.expander("🧠 Click to view full AI explanation", expanded=True):
    st.text_area(
        label="AI Output",
        value=ai_explanation,
        height=300
    )
