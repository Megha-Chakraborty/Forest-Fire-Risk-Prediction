import streamlit as st
import pickle
import numpy as np

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="FWI Prediction | Showcase",
    page_icon="🔥",
    layout="wide"
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

# --- Custom CSS for a Professional, LinkedIn-Ready Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* General App Styling */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #1a1a1a; /* Dark background */
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #262626;
        border-right: 1px solid #333;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .st-emotion-cache-1gulkj5 {
        color: #fafafa;
    }

    /* Main Title with Gradient */
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #ff4b1f, #ff9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
    }

    /* Main content card styling */
    [data-testid="stAppViewContainer"] > .main > div:first-child {
        background-color: #262626;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #333;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.8em 1em;
        font-weight: 600;
        color: #ffffff;
        background-image: linear-gradient(45deg, #ff4b1f 0%, #ff9068 100%);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255, 118, 77, 0.5);
    }

    /* Metric Styling */
    [data-testid="stMetric"] {
        background-color: #333333;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #444;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0a0;
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
    Region = st.selectbox("Region", options=["Bejaia", "Sidi Bel-abbes"])

    # Convert Region to numerical for the model
    Region_val = 0 if Region == "Bejaia" else 1
    
    # Set a fixed value for 'Classes' to prevent data leakage from the UI
    Classes_val = 1 # Corresponds to "Fire"

    predict_button = st.button("Predict Fire Risk")

# --- Main Page Content ---
st.markdown('<h1 class="title-gradient">Algerian Forest Fire Risk Prediction</h1>', unsafe_allow_html=True)
# st.markdown("This dashboard predicts the **Fire Weather Index (FWI)**, a key indicator of forest fire danger, using machine learning.")

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
        st.metric(label="Calculated Risk Level", value=risk_level)

    st.progress(min(int(prediction), 100))

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
    st.info("Awaiting input...")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Megha Chakraborty")