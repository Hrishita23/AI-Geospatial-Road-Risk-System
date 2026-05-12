import streamlit as st
import pandas as pd
import sqlite3


st.set_page_config(page_title="Road Risk AI Dashboard", layout="wide")

st.title(" AI-Powered Road Accident Risk Dashboard")


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


conn = sqlite3.connect("sql/road_risk.db")
df = pd.read_sql_query("SELECT * FROM risk_data", conn)


df["State"] = df["State"].map(state_mapping).fillna(df["State"])


state = st.selectbox("Select State", sorted(df["State"].unique()))
filtered = df[df["State"] == state]


st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Accident Count", round(filtered["Predicted_Accident_Count"].mean(), 2))
col2.metric("Max Accident Count", int(filtered["Predicted_Accident_Count"].max()))
col3.metric("Min Accident Count", int(filtered["Predicted_Accident_Count"].min()))


st.subheader(" City-Level Risk Data")
st.dataframe(filtered, use_container_width=True)

st.subheader(" Top High Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count", ascending=False).head(10), use_container_width=True)

st.subheader(" Top Low Risk Cities")
st.dataframe(df.sort_values("Predicted_Accident_Count").head(10), use_container_width=True)
