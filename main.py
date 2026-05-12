
!pip install -q kaggle
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import folium
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
import kagglehub
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")

print("Path to dataset files:", path)

print(os.listdir(path))

file_path = os.path.join(path, "US_Accidents_March23.csv")

df = pd.read_csv(
    file_path,
    nrows=200000,   # Load only 200k rows 
    low_memory=False
)

print("Dataset Shape:", df.shape)
df.head()

selected_columns = [
    'Severity',
    'Start_Time',
    'End_Time',
    'Start_Lat',
    'Start_Lng',
    'State',
    'City',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)'
]

df = df[selected_columns].copy()

print("Selected Columns:")
print(df.columns.tolist())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df['Precipitation(in)'] = df['Precipitation(in)'].fillna(0)

df = df.dropna()

print("\nShape After Cleaning:", df.shape)

df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])

df['Accident_Duration_Minutes'] = (
    (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
)

df = df[df['Accident_Duration_Minutes'] > 0]

df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Year'] = df['Start_Time'].dt.year

df['Weather_Risk_Score'] = (
    (100 - df['Visibility(mi)'] * 10) +
    df['Humidity(%)'] * 0.1 +
    df['Precipitation(in)'] * 50
)

q1 = df['Weather_Risk_Score'].quantile(0.33)
q2 = df['Weather_Risk_Score'].quantile(0.66)

def assign_risk(score):
    if score <= q1:
        return 'Low Risk'
    elif score <= q2:
        return 'Medium Risk'
    else:
        return 'High Risk'

df['Risk_Category'] = df['Weather_Risk_Score'].apply(assign_risk)


print("\nRisk Category Distribution:")
print(df['Risk_Category'].value_counts())

display(
    df[[
        'State',
        'City',
        'Severity',
        'Weather_Risk_Score',
        'Risk_Category',
        'Accident_Duration_Minutes'
    ]].head(10)
)


df.to_csv("processed_cleaned_data.csv", index=False)

print("\nCleaned dataset saved as processed_cleaned_data.csv")
print("Final Shape:", df.shape)

def severity_to_risk(severity):
    """
    Map original accident severity to business-friendly risk categories.
    Severity values in the US Accidents dataset are typically 1-4.
    """
    if severity == 1:
        return 'Low Risk'
    elif severity == 2:
        return 'Medium Risk'
    else:   # Severity 3 and 4
        return 'High Risk'

df['Risk_Category'] = df['Severity'].apply(severity_to_risk)

print("New Risk Category Distribution:")
print(df['Risk_Category'].value_counts())

feature_cols = [
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Hour',
    'DayOfWeek',
    'Month'
]

X = df[feature_cols].copy()
y = df['Risk_Category']

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} -> {label}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("FINAL CLASSIFICATION RESULTS")
print("="*50)
print(f"Classification Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Feature Importances:")
print(feature_importance)


joblib.dump(clf, "models/risk_classifier_xgboost.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

feature_importance.to_csv(
    "outputs/classification_feature_importance.csv",
    index=False
)

print("\nModel saved to models/risk_classifier_xgboost.pkl")
print("Label encoder saved to models/label_encoder.pkl")
print("Feature importance saved to outputs/classification_feature_importance.csv")

# Hyperparameter Tuning
!pip install -q xgboost

q1 = df['Accident_Duration_Minutes'].quantile(0.33)
q2 = df['Accident_Duration_Minutes'].quantile(0.66)

def assign_risk(duration):
    if duration <= q1:
        return 'Low Risk'
    elif duration <= q2:
        return 'Medium Risk'
    else:
        return 'High Risk'

df['Risk_Category'] = df['Accident_Duration_Minutes'].apply(assign_risk)

print("Risk Category Distribution:")
print(df['Risk_Category'].value_counts())

feature_cols = [
    'Severity',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Hour',
    'DayOfWeek',
    'Month'
]

X = df[feature_cols].copy()
y = df['Risk_Category']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

param_dist = {
    'n_estimators': [200, 300, 500, 700],
    'max_depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 2, 5]
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nStarting hyperparameter tuning...")
random_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(random_search.best_score_)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("OPTIMIZED CLASSIFICATION RESULTS")
print("=" * 60)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Feature Importances:")
print(feature_importance)

joblib.dump(best_model, "models/risk_classifier_xgboost_optimized.pkl")
joblib.dump(label_encoder, "models/label_encoder_optimized.pkl")

feature_importance.to_csv(
    "outputs/classification_feature_importance_optimized.csv",
    index=False
)

print("\nOptimized model saved successfully!")
print("Model: models/risk_classifier_xgboost_optimized.pkl")
print("Label Encoder: models/label_encoder_optimized.pkl")
print("Feature Importance: outputs/classification_feature_importance_optimized.csv")

best_model = joblib.load("models/risk_classifier_xgboost_optimized.pkl")
label_encoder = joblib.load("models/label_encoder_optimized.pkl")


feature_cols = [
    'Severity',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Hour',
    'DayOfWeek',
    'Month'
]

X_all = df[feature_cols].copy()

pred_encoded = best_model.predict(X_all)
pred_labels = label_encoder.inverse_transform(pred_encoded)


df['Predicted_Risk_Category'] = pred_labels

df.to_csv("outputs/area_risk_predictions.csv", index=False)

state_summary = (
    df.groupby(['State', 'Predicted_Risk_Category'])
      .size()
      .unstack(fill_value=0)
)

state_summary['Total_Accidents'] = state_summary.sum(axis=1)

state_summary['High_Risk_Percentage'] = (
    state_summary['High Risk'] /
    state_summary['Total_Accidents'] * 100
)

state_summary['Overall_State_Risk'] = (
    state_summary[['High Risk', 'Medium Risk', 'Low Risk']]
    .idxmax(axis=1)
)

state_summary = state_summary.sort_values(
    'High_Risk_Percentage',
    ascending=False
)

state_summary.to_csv("outputs/state_risk_summary.csv")

city_counts = df.groupby(['State', 'City']).size()
top_cities = city_counts.sort_values(ascending=False).head(500).index

city_df = df.set_index(['State', 'City']).loc[top_cities].reset_index()

city_summary = (
    city_df.groupby(['State', 'City', 'Predicted_Risk_Category'])
           .size()
           .unstack(fill_value=0)
)

city_summary['Total_Accidents'] = city_summary.sum(axis=1)

city_summary['High_Risk_Percentage'] = (
    city_summary['High Risk'] /
    city_summary['Total_Accidents'] * 100
)

city_summary['Overall_City_Risk'] = (
    city_summary[['High Risk', 'Medium Risk', 'Low Risk']]
    .idxmax(axis=1)
)

city_summary = city_summary.sort_values(
    'High_Risk_Percentage',
    ascending=False
)

city_summary.to_csv("outputs/city_risk_summary.csv")

print("=" * 70)
print("TOP 20 HIGH-RISK STATES")
print("=" * 70)
display(
    state_summary[
        ['Total_Accidents', 'High_Risk_Percentage', 'Overall_State_Risk']
    ].head(20)
)

print("=" * 70)
print("TOP 20 HIGH-RISK CITIES")
print("=" * 70)
display(
    city_summary[
        ['Total_Accidents', 'High_Risk_Percentage', 'Overall_City_Risk']
    ].head(20)
)

high_risk_states = state_summary[
    state_summary['Overall_State_Risk'] == 'High Risk'
]

medium_risk_states = state_summary[
    state_summary['Overall_State_Risk'] == 'Medium Risk'
]

low_risk_states = state_summary[
    state_summary['Overall_State_Risk'] == 'Low Risk'
]

print("\nNumber of High Risk States:", len(high_risk_states))
print("Number of Medium Risk States:", len(medium_risk_states))
print("Number of Low Risk States:", len(low_risk_states))

print("\nFiles Saved:")
print("outputs/area_risk_predictions.csv")
print("outputs/state_risk_summary.csv")
print("outputs/city_risk_summary.csv")

!pip install -q xgboost

city_agg = (
    df.groupby(['State', 'City'])
      .agg({
          'Severity': 'mean',
          'Distance(mi)': 'mean',
          'Temperature(F)': 'mean',
          'Humidity(%)': 'mean',
          'Pressure(in)': 'mean',
          'Visibility(mi)': 'mean',
          'Wind_Speed(mph)': 'mean',
          'Precipitation(in)': 'mean',
          'Accident_Duration_Minutes': 'mean',
          'Hour': 'mean',
          'DayOfWeek': 'mean',
          'Month': 'mean',
          'Is_Weekend': 'mean',
          'Quarter': 'mean',
          'Season': 'mean'
      })
      .reset_index()
)


city_counts = (
    df.groupby(['State', 'City'])
      .size()
      .reset_index(name='Accident_Count')
)


city_agg = city_agg.merge(city_counts, on=['State', 'City'])


city_agg = city_agg[city_agg['Accident_Count'] >= 20].reset_index(drop=True)

print("City-level dataset shape:", city_agg.shape)

feature_cols = [
    'Severity',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Accident_Duration_Minutes',
    'Hour',
    'DayOfWeek',
    'Month',
    'Is_Weekend',
    'Quarter',
    'Season'
]

X = city_agg[feature_cols].copy()
y = city_agg['Accident_Count'].copy()

y_log = np.log1p(y)

print("\nTarget Statistics:")
print("Min:", y.min())
print("Max:", y.max())
print("Median:", y.median())

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)


y_test_actual = np.expm1(y_test_log)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

model = XGBRegressor(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

print("\nTraining improved regression model...")
model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=100
)

y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n" + "=" * 60)
print("IMPROVED REGRESSION RESULTS")
print("=" * 60)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Feature Importances:")
print(feature_importance.head(15))

results = X_test.copy()
results['Actual_Accident_Count'] = np.round(y_test_actual).astype(int)
results['Predicted_Accident_Count'] = np.round(y_pred).astype(int)
results['Absolute_Error'] = abs(
    results['Actual_Accident_Count'] -
    results['Predicted_Accident_Count']
)
results['Percentage_Error'] = (
    results['Absolute_Error'] /
    results['Actual_Accident_Count'].clip(lower=1)
) * 100

print("\nSample Predictions:")
display(
    results[
        [
            'Actual_Accident_Count',
            'Predicted_Accident_Count',
            'Absolute_Error',
            'Percentage_Error'
        ]
    ].head(10)
)

joblib.dump(model, "models/accident_count_regressor_improved.pkl")

results.to_csv(
    "outputs/regression_predictions_improved.csv",
    index=False
)

feature_importance.to_csv(
    "outputs/regression_feature_importance_improved.csv",
    index=False
)

city_agg.to_csv(
    "outputs/city_level_aggregated_dataset.csv",
    index=False
)

print("\nImproved regression model saved successfully!")
print("models/accident_count_regressor_improved.pkl")
print("outputs/regression_predictions_improved.csv")
print("outputs/regression_feature_importance_improved.csv")

city_counts = df.groupby(['State', 'City']).size().reset_index(name='Accident_Count')

state_counts = df.groupby('State').size().reset_index(name='State_Accident_Count')

df = df.merge(city_counts, on=['State', 'City'], how='left')
df = df.merge(state_counts, on='State', how='left')

df['City_State_Ratio'] = df['Accident_Count'] / df['State_Accident_Count']
df['Log_City_Accidents'] = np.log1p(df['Accident_Count'])
df['Log_State_Accidents'] = np.log1p(df['State_Accident_Count'])

city_agg = df.groupby(['State', 'City']).agg({

    
    'Severity': 'mean',
    'Distance(mi)': 'mean',
    'Temperature(F)': 'mean',
    'Humidity(%)': 'mean',
    'Pressure(in)': 'mean',
    'Visibility(mi)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Precipitation(in)': 'mean',

    
    'Hour': 'mean',
    'DayOfWeek': 'mean',
    'Month': 'mean',

    
    'City_State_Ratio': 'mean',
    'Log_State_Accidents': 'mean',

    
    'Accident_Duration_Minutes': 'mean'
}).reset_index()

city_counts = df.groupby(['State', 'City']).size().reset_index(name='Accident_Count')

city_agg = city_agg.merge(city_counts, on=['State', 'City'])

city_agg = city_agg[city_agg['Accident_Count'] >= 20].reset_index(drop=True)

print("Final dataset shape:", city_agg.shape)

feature_cols = [
    'Severity',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Hour',
    'DayOfWeek',
    'Month',
    'City_State_Ratio',
    'Log_State_Accidents',
    'Accident_Duration_Minutes'
]

X = city_agg[feature_cols]
y = np.log1p(city_agg['Accident_Count'])  

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_alpha=0.05,
    reg_lambda=1.5,
    gamma=0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

print("\nTraining optimized regression model...")

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

y_pred = np.maximum(y_pred, 0)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n" + "=" * 60)
print("FINAL REGRESSION RESULTS (OPTIMIZED)")
print("=" * 60)
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.2f}")
print(f"MAE      : {mae:.2f}")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Features:")
print(feature_importance)

results = X_test.copy()
results['Actual_Accidents'] = np.round(y_true).astype(int)
results['Predicted_Accidents'] = np.round(y_pred).astype(int)
results['Error'] = abs(results['Actual_Accidents'] - results['Predicted_Accidents'])

print("\nSample Predictions:")
print(results[['Actual_Accidents', 'Predicted_Accidents', 'Error']].head(10))

joblib.dump(model, "models/accident_count_regressor_final.pkl")

results.to_csv("outputs/regression_predictions_final.csv", index=False)
feature_importance.to_csv("outputs/regression_feature_importance_final.csv", index=False)
city_agg.to_csv("outputs/city_level_dataset_final.csv", index=False)

print("\nFINAL MODEL SAVED SUCCESSFULLY")
print("models/accident_count_regressor_final.pkl")
print("outputs/regression_predictions_final.csv")
print("outputs/regression_feature_importance_final.csv")
print("outputs/city_level_dataset_final.csv")

model = joblib.load("models/accident_count_regressor_final.pkl")

city_counts = df.groupby(['State', 'City']).size().reset_index(name='Accident_Count')

state_counts = df.groupby('State').size().reset_index(name='State_Accident_Count')

city_agg = city_counts.merge(state_counts, on='State', how='left')

city_agg['City_State_Ratio'] = (
    city_agg['Accident_Count'] / city_agg['State_Accident_Count']
)

city_agg['Log_State_Accidents'] = np.log1p(city_agg['State_Accident_Count'])

weather_features = df.groupby(['State', 'City']).agg({
    'Severity': 'mean',
    'Distance(mi)': 'mean',
    'Temperature(F)': 'mean',
    'Humidity(%)': 'mean',
    'Pressure(in)': 'mean',
    'Visibility(mi)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Precipitation(in)': 'mean',
    'Hour': 'mean',
    'DayOfWeek': 'mean',
    'Month': 'mean',
    'Accident_Duration_Minutes': 'mean'
}).reset_index()

city_agg = city_agg.merge(weather_features, on=['State', 'City'])

feature_cols = [
    'Severity',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Hour',
    'DayOfWeek',
    'Month',
    'City_State_Ratio',
    'Log_State_Accidents',
    'Accident_Duration_Minutes'
]

X = city_agg[feature_cols]

y_log_pred = model.predict(X)

city_agg['Predicted_Accident_Count'] = np.expm1(y_log_pred)
city_agg['Predicted_Accident_Count'] = np.maximum(city_agg['Predicted_Accident_Count'], 0)
city_agg['Predicted_Accident_Count'] = city_agg['Predicted_Accident_Count'].round().astype(int)

top_cities = city_agg.sort_values('Predicted_Accident_Count', ascending=False)

print("\n TOP 20 ACCIDENT HOTSPOTS")
print(top_cities[['State', 'City', 'Predicted_Accident_Count']].head(20))

city_agg.to_csv("outputs/predicted_accident_counts_final.csv", index=False)

print("\nPrediction completed successfully")

df_map = city_agg.copy()

geo = df.groupby(['State', 'City']).agg({
    'Start_Lat': 'mean',
    'Start_Lng': 'mean'
}).reset_index()

df_map = df_map.merge(geo, on=['State', 'City'], how='left')

df_map = df_map.dropna(subset=['Start_Lat', 'Start_Lng'])

def risk_level(x):
    if x >= np.percentile(df_map['Predicted_Accident_Count'], 75):
        return 'High Risk'
    elif x >= np.percentile(df_map['Predicted_Accident_Count'], 40):
        return 'Medium Risk'
    else:
        return 'Low Risk'

df_map['Risk_Category'] = df_map['Predicted_Accident_Count'].apply(risk_level)

m = folium.Map(location=[df_map['Start_Lat'].mean(),
                         df_map['Start_Lng'].mean()],
               zoom_start=5)

def color(risk):
    if risk == "High Risk":
        return "red"
    elif risk == "Medium Risk":
        return "orange"
    else:
        return "green"

for _, row in df_map.iterrows():
    folium.CircleMarker(
        location=[row['Start_Lat'], row['Start_Lng']],
        radius=5,
        popup=f"{row['City']} | Predicted: {row['Predicted_Accident_Count']}",
        color=color(row['Risk_Category']),
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

m.save("outputs/accident_risk_map.html")

print("Map saved: outputs/accident_risk_map.html")

from google.colab import files
files.download("outputs/accident_risk_map.html")

df_explain = city_agg.copy()

assert 'Predicted_Accident_Count' in df_explain.columns

q1 = df_explain['Predicted_Accident_Count'].quantile(0.33)
q2 = df_explain['Predicted_Accident_Count'].quantile(0.66)

def risk_label(x):
    if x <= q1:
        return "Low Risk"
    elif x <= q2:
        return "Medium Risk"
    else:
        return "High Risk"

df_explain["Risk_Category"] = df_explain["Predicted_Accident_Count"].apply(risk_label)

def generate_explanation(row):
    city = row["City"]
    state = row["State"]
    risk = row["Risk_Category"]
    acc = row["Predicted_Accident_Count"]

    if risk == "High Risk":
        return f"{city}, {state} is classified as HIGH RISK due to a very high predicted accident count of {int(acc)}, indicating dense traffic conditions and elevated road exposure."

    elif risk == "Medium Risk":
        return f"{city}, {state} is classified as MEDIUM RISK with a moderate predicted accident count of {int(acc)}, suggesting average traffic safety conditions."

    else:
        return f"{city}, {state} is classified as LOW RISK due to a low predicted accident count of {int(acc)}, indicating relatively safer road conditions."

df_explain["AI_Explanation"] = df_explain.apply(generate_explanation, axis=1)

df_explain[["State", "City", "Predicted_Accident_Count", "Risk_Category", "AI_Explanation"]].head(10)

df_explain = city_agg.copy()

q1 = df_explain["Predicted_Accident_Count"].quantile(0.33)
q2 = df_explain["Predicted_Accident_Count"].quantile(0.66)

def risk_label(x):
    if x <= q1:
        return "Low Risk"
    elif x <= q2:
        return "Medium Risk"
    else:
        return "High Risk"

df_explain["Risk_Category"] = df_explain["Predicted_Accident_Count"].apply(risk_label)

def simple_explanation(row):
    city = row["City"]
    state = row["State"]
    risk = row["Risk_Category"]
    acc = int(row["Predicted_Accident_Count"])

    if risk == "High Risk":
        return (
            f"{city}, {state} has a HIGH chance of road accidents. "
            f"The model predicts around {acc} accidents here. "
            f"This means the area is busy and drivers should be extra careful."
        )

    elif risk == "Medium Risk":
        return (
            f"{city}, {state} has a MODERATE chance of road accidents. "
            f"The model predicts around {acc} accidents here. "
            f"Some caution is needed while driving in this area."
        )

    else:
        return (
            f"{city}, {state} has a LOW chance of road accidents. "
            f"The model predicts around {acc} accidents here. "
            f"This area is relatively safer for road travel."
        )

df_explain["AI_Explanation"] = df_explain.apply(simple_explanation, axis=1)

pd.set_option("display.max_colwidth", None)

print("\n=== SAMPLE OUTPUT ===\n")

display(
    df_explain[
        ["State", "City", "Predicted_Accident_Count", "Risk_Category", "AI_Explanation"]
    ].head(10)
)

os.makedirs("outputs", exist_ok=True)

df_explain.to_csv("outputs/risk_explanations_final.csv", index=False)

print("\nSTEP 7 COMPLETED SUCCESSFULLY")
print("File saved: outputs/risk_explanations_final.csv")

df_sql = df_explain.copy()

os.makedirs("sql", exist_ok=True)

conn = sqlite3.connect("sql/road_risk.db")

df_sql.to_sql("risk_data", conn, if_exists="replace", index=False)

print("Data stored in SQLite database successfully")

def run_query(query):
    return pd.read_sql_query(query, conn)

query1 = """
SELECT State, City, Predicted_Accident_Count, Risk_Category
FROM risk_data
WHERE Risk_Category = 'High Risk'
ORDER BY Predicted_Accident_Count DESC
LIMIT 10;
"""

print("\nTOP 10 HIGH RISK CITIES")
display(run_query(query1))

query2 = """
SELECT State, City, Predicted_Accident_Count, Risk_Category
FROM risk_data
WHERE Risk_Category = 'Low Risk'
ORDER BY Predicted_Accident_Count ASC
LIMIT 10;
"""

print("\nTOP 10 LOW RISK CITIES")
display(run_query(query2))

query3 = """
SELECT State,
       AVG(Predicted_Accident_Count) AS Avg_Accidents
FROM risk_data
GROUP BY State
ORDER BY Avg_Accidents DESC
LIMIT 10;
"""

print("\nTOP STATES BY AVERAGE ACCIDENTS")
display(run_query(query3))

conn.close()

print("\nSTEP 8 COMPLETED SUCCESSFULLY")

pip install streamlit
st.set_page_config(page_title="Road Risk AI System", layout="wide")

st.title("AI-Powered Road Accident Risk Prediction System")
st.markdown("Geospatial AI system for accident prediction and risk analysis")

conn = sqlite3.connect("sql/road_risk.db")
df = pd.read_sql_query("SELECT * FROM risk_data", conn)

st.sidebar.header("Filter Data")

state = st.sidebar.selectbox("Select State", ["All"] + sorted(df["State"].unique().tolist()))

if state != "All":
    df = df[df["State"] == state]

city = st.sidebar.selectbox("Select City", ["All"] + sorted(df["City"].unique().tolist()))

if city != "All":
    df = df[df["City"] == city]

col1, col2, col3 = st.columns(3)

col1.metric("Total Cities", len(df))
col2.metric("Avg Accident Count", round(df["Predicted_Accident_Count"].mean(), 2))
col3.metric("High Risk Areas", len(df[df["Risk_Category"] == "High Risk"]))

st.subheader("Prediction Results")

st.dataframe(
    df[[
        "State",
        "City",
        "Predicted_Accident_Count",
        "Risk_Category",
        "AI_Explanation"
    ]],
    use_container_width=True
)

st.subheader("High Risk Areas")

high_risk = df[df["Risk_Category"] == "High Risk"] \
    .sort_values(by="Predicted_Accident_Count", ascending=False) \
    .head(10)

st.table(high_risk[["State", "City", "Predicted_Accident_Count"]])

st.subheader("Low Risk Areas")

low_risk = df[df["Risk_Category"] == "Low Risk"] \
    .sort_values(by="Predicted_Accident_Count", ascending=True) \
    .head(10)

st.table(low_risk[["State", "City", "Predicted_Accident_Count"]])

st.subheader("AI Explanation Example")

if len(df) > 0:
    st.write(df["AI_Explanation"].iloc[0])


st.markdown("---")
st.markdown("Built using Machine Learning + Geospatial AI + SQL")

!streamlit run app.py

