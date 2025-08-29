import streamlit as st
import pickle
import numpy as np

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="centered"  # Use a centered layout for a clean, focused look
)

# --- Model Loading (cached for performance) ---
@st.cache_resource
def load_models():
    """Loads all models and the scaler from disk."""
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    models = {
        "Ridge Regression": pickle.load(open("models/ridge.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/dt.pkl", "rb")),
        "Random Forest": pickle.load(open("models/rf.pkl", "rb")),
        "SVR": pickle.load(open("models/svr.pkl", "rb"))
    }
    return scaler, models

scaler, models = load_models()

# --- Custom CSS for a polished, modern look ---
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e6e6e6;
    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.75em 1em;
        font-weight: 600;
        color: #ffffff;
        background-color: #ff4b4b;
        transition: background-color 0.25s;
    }
    .stButton>button:hover {
        background-color: #ff3030;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Prediction Inputs")

    model_choice = st.selectbox(
        "Select Machine Learning Model",
        options=list(models.keys())
    )
    st.markdown("---")

    st.subheader("Meteorological Data")
    col1, col2 = st.columns(2)
    with col1:
        Temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0, 0.5)
        Ws = st.slider("Wind Speed (km/h)", 5.0, 30.0, 15.0)
        FFMC = st.slider("FFMC", 25.0, 100.0, 85.0)
    with col2:
        RH = st.slider("Humidity (%)", 20.0, 100.0, 50.0)
        Rain = st.slider("Rain (mm)", 0.0, 20.0, 0.0, 0.1)
        DMC = st.slider("DMC", 1.0, 70.0, 20.0)

    ISI = st.slider("Initial Spread Index (ISI)", 0.0, 20.0, 5.0)
    st.markdown("---")

    st.subheader("Contextual Information")
    Classes = st.selectbox("Fire Occurrence", options=["Fire", "Not Fire"])
    Region = st.selectbox("Region", options=["Bejaia", "Sidi Bel-abbes"])

    # Convert categorical inputs to numerical for the model
    Classes_val = 1 if Classes == "Fire" else 0
    Region_val = 0 if Region == "Bejaia" else 1

    predict_button = st.button("🔥 Predict Fire Risk")

# --- Main Page Content ---
st.title("🔥 Algerian Forest Fire Risk Prediction")
#st.markdown("Use the sidebar on the left to input data and get a Fire Weather Index (FWI) prediction.")

# --- Prediction Logic and Display ---
if predict_button:
    # Prepare features and scale them
    features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes_val, Region_val]])
    features_scaled = scaler.transform(features)

    # Select model and predict
    selected_model = models[model_choice]
    prediction = selected_model.predict(features_scaled)[0]

    # Determine risk level
    if prediction < 5:
        risk_level = "Low"
    elif prediction < 30:
        risk_level = "Moderate"
    elif prediction < 80:
        risk_level = "High"
    else:
        risk_level = "Extreme"

    # Display results in a clean, organized layout
    st.markdown("---")
    st.header("Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Fire Weather Index (FWI)", value=f"{prediction:.2f}")
    with col2:
        st.metric(label="Risk Level", value=risk_level)

    st.progress(min(int(prediction), 100), text=f"Risk Level: {risk_level}")

    # Show a clear conclusion message with an appropriate icon
    if risk_level == "Low":
        st.success(f"**Conclusion:** The predicted fire risk is **{risk_level}**. Conditions are not favorable for a significant fire event. 🌿")
    elif risk_level == "Moderate":
        st.info(f"**Conclusion:** The predicted fire risk is **{risk_level}**. Caution is advised. Monitor conditions closely. ⚠️")
    elif risk_level == "High":
        st.warning(f"**Conclusion:** The predicted fire risk is **{risk_level}**. Conditions are favorable for fire spread. High alert is necessary. 🔥")
    else:
        st.error(f"**Conclusion:** The predicted fire risk is **{risk_level}**. Extreme fire danger. Avoid any activity that could start a fire. 🚨")
else:
    st.info("Awaiting input...to 'Predict Fire Risk'.")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Megha Chakraborty")