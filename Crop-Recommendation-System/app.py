"""
Crop Recommendation System — Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="centered")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@st.cache_resource
def load_model():
    model = joblib.load("model/crop_model.pkl")
    le = joblib.load("model/label_encoder.pkl")
    return model, le


model, le = load_model()

st.title("🌱 Crop Recommendation System")
st.write(
    "Enter your soil nutrients and weather conditions to get the best crop "
    "recommendation, powered by a Random Forest model."
)

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen — N (kg/ha)", min_value=0.0, max_value=200.0, value=90.0)
    P = st.number_input("Phosphorus — P (kg/ha)", min_value=0.0, max_value=150.0, value=42.0)
    K = st.number_input("Potassium — K (kg/ha)", min_value=0.0, max_value=250.0, value=43.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=26.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=210.0)

if st.button("Recommend Crop", type="primary"):
    sample = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall,
    }])[FEATURES]

    probs = model.predict_proba(sample)[0]
    top_idx = probs.argsort()[::-1][:3]

    best_crop = le.classes_[top_idx[0]]
    st.success(f"### 🏆 Best match: **{best_crop.title()}**")

    st.write("#### Top 3 recommendations")
    for rank, idx in enumerate(top_idx, start=1):
        crop = le.classes_[idx]
        st.write(f"{rank}. **{crop.title()}** — {probs[idx] * 100:.1f}%")
        st.progress(float(probs[idx]))

st.divider()
st.caption("Model: Random Forest • Trained on soil (N, P, K, pH) and weather (temperature, humidity, rainfall) data.")
