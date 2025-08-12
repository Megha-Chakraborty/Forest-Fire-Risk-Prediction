##FOR STREAMLIT BASED WEB APPLICATION

import streamlit as st
import pickle
import numpy as np

# Load your trained model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

st.set_page_config(page_title="Fire Weather Index Prediction", page_icon="🔥", layout="centered")

st.title("🔥 Fire Weather Index Prediction 🔥")
st.markdown("Enter the weather and environmental parameters below:")

# Input fields
Temperature = st.number_input("Temperature (°C)", value=20.0)
RH = st.number_input("Relative Humidity (%)", value=50.0)
Ws = st.number_input("Wind Speed (km/h)", value=10.0)
Rain = st.number_input("Rain (mm)", value=0.0)
FFMC = st.number_input("FFMC", value=80.0)
DMC = st.number_input("DMC", value=20.0)
ISI = st.number_input("ISI", value=5.0)
Classes = st.number_input("Classes (numeric encoding)", value=0.0)
Region = st.number_input("Region (numeric encoding)", value=0.0)

if st.button("Predict"):
    # Prepare and scale input
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    input_scaled = standard_scaler.transform(input_data)
    prediction = ridge_model.predict(input_scaled)[0]
    st.success(f"THE FWI PREDICTION IS: {prediction:.2f}")
    if prediction < 5:
        st.info("Low fire danger.")
    elif prediction < 30:
        st.warning("Moderate fire danger.")
    elif prediction < 80:
        st.error("High fire danger!")
    else:
        st.error("Extreme fire danger! Take precautions.")

st.markdown("---")
st.caption("Project created by Megha Chakraborty")