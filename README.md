
#  AI-Powered Geospatial Road Risk Prediction & Insight Generation System

An end-to-end **Machine Learning + Geospatial AI + Generative AI** system that analyzes road accident data, assigns each city a risklevel, predicts the accident number and generates human-readable explanations with an interactive dashboard.

This project is inspired by real-world **location intelligence systems like HERE Technologies** and demonstrates a complete production-level ML pipeline.

---

#  Dataset

We use the **US Accidents Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

#  Project Overview

This system includes **three major AI components**:

---

## 1️⃣ Classification Task (Risk Prediction)

Predicts accident risk category:

- 🟢 Low Risk  
- 🟡 Medium Risk  
- 🔴 High Risk  

### Models Used:
- Random Forest Classifier  
- XGBoost Classifier (Final Model)

---

## 2️⃣ Regression Models (Impact Prediction)

Predicts numerical accident impact:

- Number of accident cases  

### Models Used:
- Random Forest Regressor  
- XGBoost Regressor (Optimized)

---

## 3️⃣ LLM-Based Explanations (Generative AI)

Generates human-readable insights such as:

> “California is classified as High Risk due to high accident frequency, dense traffic conditions, and elevated injury rates.”

This is implemented using:
- Rule-based explanation system (baseline)
- Prompt-style structured generation (LLM-ready design)

---

## 4️⃣ Geospatial Visualization

- Interactive risk maps using **Folium**
- Accident hotspot visualization
- City-wise and state-wise risk distribution

---

## 5️⃣ SQL Analytics Layer

All processed data is stored in **SQLite database**, enabling queries like:

- Top high-risk states
- Most dangerous cities
- Accident ranking by region

---

## 6️⃣ Streamlit Dashboard

Interactive dashboard features:

- State-wise filtering
- City-level risk analysis
- KPIs (avg, max, min accident count)
- Top risky & safe cities
- AI-generated explanations

---

# ⚙️ How to Run the Project

## 📌 Step 1 — Run Full ML Pipeline

Run:

```bash
python app.py
