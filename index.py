import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Irrigation System", layout="wide")

st.title("🌱 Smart Irrigation System Dashboard")
st.markdown("AI System to predict **Water / No Water** using soil & weather conditions")

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("dataset2.csv")

def irrigation_logic(row):
    if row["SoilMoisture"] < 300:
        return 1 
    elif row["SoilMoisture"] < 500 and row["temperature"] > 30:
        return 1
    elif row["Humidity"] < 30:
        return 1
    else:
        return 0 

data["IrrigationNeeded"] = data.apply(irrigation_logic, axis=1)

X = data[["CropDays", "SoilMoisture", "temperature", "Humidity"]]
y = data["IrrigationNeeded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🌾 Input Conditions")

crop_days = st.sidebar.slider("Crop Days", 1, 30, 10)
soil_moisture = st.sidebar.slider("Soil Moisture", 0, 1000, 300)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)

# =========================
# PREDICTION
# =========================
def predict(crop_days, soil_moisture, temp, humidity):
    input_data = np.array([[crop_days, soil_moisture, temp, humidity]])
    pred = model.predict(input_data)[0]
    return pred

if st.sidebar.button("🔍 Analyze"):
    result = predict(crop_days, soil_moisture, temperature, humidity)

    st.subheader(" Prediction Result")

    if result == 1:
        st.error("🚨 WATER REQUIRED")
        st.markdown("Plants need irrigation immediately.")
    else:
        st.success("✅ NO WATER NEEDED")
        st.markdown("Soil conditions are sufficient.")

# =========================
# METRICS SECTION
# =========================
st.markdown("---")
st.subheader("📊 System Performance")

col1, col2 = st.columns(2)

with col1:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Feature Importance")

    importances = model.feature_importances_
    features = X.columns

    fig2, ax2 = plt.subplots()
    ax2.barh(features, importances)
    st.pyplot(fig2)

# =========================
# DATA INSIGHT
# =========================
st.markdown("---")
st.subheader("📈 Dataset Overview")

st.write(data.describe())

st.bar_chart(data["IrrigationNeeded"].value_counts())

# =========================
# EXPLANATION
# =========================
st.markdown("---")
st.subheader("🧠 How the System Works")

st.info("""
This AI system uses:
- Soil Moisture
- Temperature
- Humidity
- Crop Age

Then a Random Forest model predicts whether irrigation is needed or not.

Decision:
- 1 → WATER REQUIRED
- 0 → NO WATER NEEDED
""")