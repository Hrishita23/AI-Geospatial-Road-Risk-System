%%writefile app.py
import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="Road Risk AI Dashboard", layout="wide")

st.title("🚦 AI-Powered Road Accident Risk Dashboard")

# -------------------------
# Load SQLite data
# -------------------------
conn = sqlite3.connect("sql/road_risk.db")

df = pd.read_sql_query("SELECT * FROM risk_data", conn)

# -------------------------
# Filters
# -------------------------
state = st.selectbox("Select State", df["State"].unique())

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
# Table
# -------------------------
st.subheader("📍 City-Level Risk Data")
st.dataframe(filtered)

# -------------------------
# Top risky cities
# -------------------------
st.subheader("🔥 Top High Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count", ascending=False).head(10))

st.subheader("🟢 Top Low Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count").head(10))