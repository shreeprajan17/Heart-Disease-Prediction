import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json

st.title("Heart Disease Prediction App (KNN Model)")

# Load saved model and scaler
model = load("knn_model.joblib")
scaler = load("scaler.joblib")

# Load feature names
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

st.write("### Enter Patient Health Values")

# Create input fields dynamically
inputs = {}

for col in feature_names:

    # Only change: sex becomes dropdown instead of number
    if col == "sex":
        gender = st.selectbox("sex", ["Female", "Male"])
        inputs[col] = 1 if gender == "Male" else 0
    
    else:
        inputs[col] = st.number_input(col, value=0.0)

# Predict button
if st.button("Predict Heart Disease"):
    # Convert to DataFrame
    df = pd.DataFrame([inputs])

    # Scale input
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]

    # Output
    if prediction == 1:
        st.error("High Risk: Heart Disease Detected")
    else:
        st.success("Low Risk: No Heart Disease Detected")
