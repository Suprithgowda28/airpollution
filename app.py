import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model_filename = "air_pollution_model.pkl"
model = joblib.load(model_filename)

st.title("Air Pollution Prediction App")
st.write("Enter environmental conditions to predict PM2.5 levels.")

# User inputs
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=5.0)
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0)
no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=40.0)
so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=100.0, value=20.0)
co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=20.0, value=1.0)
o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=200.0, value=50.0)

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    "temperature": [temperature],
    "humidity": [humidity],
    "wind_speed": [wind_speed],
    "pm10": [pm10],
    "no2": [no2],
    "so2": [so2],
    "co": [co],
    "o3": [o3]
})

# Predict PM2.5
if st.button("Predict PM2.5"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted PM2.5 Level: {prediction:.2f} µg/m³")

st.write("This app uses a machine learning model to estimate PM2.5 levels based on environmental conditions.")