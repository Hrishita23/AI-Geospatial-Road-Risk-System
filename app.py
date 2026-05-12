import streamlit as st
import pandas as pd
import sqlite3

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Road Risk AI Dashboard", layout="wide")

st.title("🚦 AI-Powered Road Accident Risk Dashboard")

# -------------------------
# State Mapping (Full Names)
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

# Convert abbreviations → full names
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
# City-Level Table
# -------------------------
st.subheader("📍 City-Level Risk Data")
st.dataframe(filtered, use_container_width=True)

# -------------------------
# Top Risky Cities
# -------------------------
st.subheader("🔥 Top High Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count", ascending=False).head(10), use_container_width=True)

st.subheader("🟢 Top Low Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count").head(10), use_container_width=True)

# -------------------------
# AI Explanation (FULL FIX - NO CUTTING)
# -------------------------
st.subheader("🤖 AI Risk Explanation")

# Replace this with your model/LLM output
ai_explanation = """
This region shows elevated accident risk due to high traffic density,
frequent congestion during peak hours, and historical accident clusters.

Weather conditions, road quality, and traffic violations also contribute significantly.

Key insights:
- High congestion during morning and evening peaks
- Accident hotspots near urban intersections
- Moderate weather-related impact

Recommendations:
- Improve traffic signal timing
- Install smart surveillance systems
- Increase enforcement in high-risk zones
- Use predictive monitoring for peak hours
"""

def render_full_text(text):
    return f"""
    <div style="
        white-space: pre-wrap;
        font-size: 16px;
        line-height: 1.6;
        padding: 10px;
        border-radius: 10px;
        background-color: #111827;
        color: #F9FAFB;
    ">
    {text}
    </div>
    """

with st.expander("🧠 Click to view full AI explanation", expanded=True):
    st.markdown(render_full_text(ai_explanation), unsafe_allow_html=True)
