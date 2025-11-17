import streamlit as st
import numpy as np
from joblib import load

st.title("Heart Disease Prediction")

# Load model + scaler
model = load("knn_model.joblib")
scaler = load("scaler.joblib")

st.header("Enter Patient Health Values")

# Gender dropdown instead of numeric input
sex = st.selectbox(
    "Sex",
    options=["Female", "Male"],
    help="Select the patient's biological sex."
)

# Convert to 0/1 for model
sex_value = 1 if sex == "Male" else 0

# Use sliders / number inputs
age = st.slider("Age", min_value=20, max_value=90, step=1)
cp = st.slider("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80.0, max_value=200.0, step=1.0)
chol = st.number_input("Cholesterol (chol)", min_value=100.0, max_value=600.0, step=1.0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.slider("Resting ECG Results (restecg)", min_value=0, max_value=2, step=1)
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60.0, max_value=250.0, step=1.0)
exang = st.selectbox("Exercise-Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=7.0, step=0.1)
slope = st.slider("Slope of ST Segment (slope)", min_value=0, max_value=2, step=1)
ca = st.slider("Number of Major Vessels (0-3)", min_value=0, max_value=3, step=1)
thal = st.slider("Thal (1 = normal, 2 = fixed defect, 3 = reversible)", min_value=1, max_value=3, step=1)

# Arrange inputs into array
features = np.array([[age, sex_value, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Scale
scaled_features = scaler.transform(features)

# Predict
if st.button("Predict"):
    pred = model.predict(scaled_features)[0]
    
    if pred == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
