import streamlit as st
import joblib
import numpy as np

# Load tuned Random Forest model
model = joblib.load("wine_quality_model.pkl")

st.set_page_config(page_title="Smart Wine Quality Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Smart Wine Quality Predictor")
st.markdown("### Predict the quality of wine using key chemical properties")

st.sidebar.header("Enter Wine Parameters")

# Sidebar inputs
fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.08)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 50.0)
density = st.sidebar.number_input("Density", 0.9900, 1.0100, 0.9978)
pH = st.sidebar.number_input("pH", 2.0, 5.0, 3.2)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.6)
alcohol = st.sidebar.number_input("Alcohol", 5.0, 15.0, 10.0)

# Predict button
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                          pH, sulphates, alcohol]])
    prediction = model.predict(features)[0]

    quality_labels = {
        0: "Poor",
        1: "Below Average",
        2: "Average",
        3: "Good",
        4: "Very Good",
        5: "Excellent"
    }

    st.markdown("### 🧾 Prediction Result")
    st.success(f"Predicted Wine Quality: **{quality_labels.get(prediction, prediction)}**")
