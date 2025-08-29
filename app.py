import streamlit as st
import pickle
import numpy as np

# Load models
scaler = pickle.load(open("models/scaler.pkl", "rb"))
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
dt_model = pickle.load(open("models/dt.pkl", "rb"))
rf_model = pickle.load(open("models/rf.pkl", "rb"))
svr_model = pickle.load(open("models/svr.pkl", "rb"))

st.set_page_config(page_title="Forest Fire Prediction using FWI", page_icon="🔥", layout="wide")

# Custom CSS - Updated header animation
st.markdown("""
<style>
    .main {padding: 0rem;}
    .block-container {padding-top: 1rem;}
    
    /* Animated Header Styling */
    .header {
        background: linear-gradient(45deg, #1a1a1a, #434343);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Fire animation */
    .fire-animation {
        position: absolute;
        top: 50%;
        left: 0;
        width: 100%;
        height: 100%;
        transform: translateY(-50%);
        background: 
            radial-gradient(ellipse at center, rgba(255,160,0,0.4) 0%, rgba(255,160,0,0) 70%),
            radial-gradient(ellipse at center, rgba(255,69,0,0.4) 0%, rgba(255,69,0,0) 70%);
        background-size: 200% 200%, 150% 150%;
        animation: fire-effect 8s ease infinite;
        opacity: 0.7;
        mix-blend-mode: overlay;
    }
    
    .fire-particles {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at center, #ff6b08 1px, transparent 1px),
            radial-gradient(circle at center, #ff4500 1px, transparent 1px);
        background-size: 24px 24px, 16px 16px;
        animation: particle-rise 4s linear infinite;
        opacity: 0.3;
    }
    
    @keyframes fire-effect {
        0%, 100% {
            background-position: 0% 50%, 0% 50%;
        }
        50% {
            background-position: 100% 50%, 100% 50%;
        }
    }
    
    @keyframes particle-rise {
        from {
            transform: translateY(100%);
        }
        to {
            transform: translateY(-100%);
        }
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(to right, #ff4b2b, #ff416c);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,75,43,0.4);
    }

    /* Make prediction section transparent */
    .prediction-section {
        text-align: center;
        padding: 1rem;
    }

    /* Remove white backgrounds from selectbox and sliders */
    .stSelectbox, .stSlider {
        background: transparent !important;
    }

    /* Add these new styles for selectbox cursor */
    .stSelectbox > div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] input {
        cursor: pointer !important;
    }

    /* This ensures the dropdown options also show pointer cursor */
    div[role="listbox"] ul {
        cursor: pointer !important;
    }
    
    div[role="listbox"] li {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

# Updated header with new fire animation
st.markdown("""
<div class="header">
    <div class="fire-animation"></div>
    <div class="fire-particles"></div>
    <h1 style='color: #ffd700; text-align: center; font-size: 2.8rem; position: relative; z-index: 1; 
               text-shadow: 0 2px 15px rgba(255,69,0,0.5), 0 -2px 15px rgba(255,160,0,0.5);'>
        Forest Fire Risk Prediction using Fire Weather Index
    </h1>
    <p style='color: #ffebcd; text-align: center; font-size: 1.3rem; position: relative, z-index: 1; 
              text-shadow: 0 1px 8px rgba(255,69,0,0.3);'>
        Predict Forest Fire Risk
    </p>
</div>
""", unsafe_allow_html=True)

# Add this JavaScript for smooth scrolling after the existing CSS
st.markdown("""
<script>
    function smoothScroll(elementId) {
        const element = document.getElementById(elementId);
        element.scrollIntoView({behavior: 'smooth', block: 'center'});
    }
</script>
""", unsafe_allow_html=True)

# Main Content - removed unnecessary white boxes
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("#### Select Machine Learning Model")
    model_choice = st.selectbox(
        "",
        ("Ridge Regression", "Decision Tree", "Random Forest", "SVR")
    )
    
    st.markdown("#### Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        Temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
        RH = st.slider("Relative Humidity (%)", 0.0, 100.0, 50.0)
        Ws = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)
        Rain = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0)
        FFMC = st.slider("Fine Fuel Moisture Code", 0.0, 100.0, 85.0)
    
    with col2:
        DMC = st.slider("Duff Moisture Code", 0.0, 100.0, 20.0)
        ISI = st.slider("Initial Spread Index", 0.0, 100.0, 5.0)
        Classes = st.selectbox("Fire Occurrence", [0, 1], 
                             format_func=lambda x: "Fire Present" if x == 0 else "No Fire")
        
        # Modified Region selectbox with session state for scrolling
        if 'previous_region' not in st.session_state:
            st.session_state.previous_region = None
        
        Region = st.selectbox("Region", [0, 1], 
                            format_func=lambda x: "Bejaia Region" if x == 0 else "Sidi Bel-abbes Region",
                            key="region_select")
        
        # Check if Region value changed
        if Region != st.session_state.previous_region:
            st.session_state.previous_region = Region
            st.markdown("""
                <script>
                    setTimeout(function() {
                        document.querySelector('button[kind="primary"]').scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        });
                    }, 100);
                </script>
                """, unsafe_allow_html=True)

with right_col:
    st.markdown("#### Prediction Results")
    
    features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    features_scaled = scaler.transform(features)

    if st.button("Predict", key="predict_button"):
        if model_choice == "Ridge Regression":
            model = ridge_model
        elif model_choice == "Decision Tree":
            model = dt_model
        elif model_choice == "Random Forest":
            model = rf_model
        else:
            model = svr_model

        prediction = model.predict(features_scaled)[0]

        # Add ID to results and scroll to it
        st.markdown('<div id="prediction_results">', unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align:center; color:#ff5722;'>Predicted Fire Weather Index (FWI): {prediction:.2f}</h2>",
            unsafe_allow_html=True
        )

        st.markdown("#### Fire Danger Meter")
        if prediction < 5:
            st.progress(10)
            st.success("Low fire danger")
        elif prediction < 30:
            st.progress(40)
            st.info("Moderate fire danger")
        elif prediction < 80:
            st.progress(70)
            st.warning("High fire danger")
        else:
            st.progress(100)
            st.error("Extreme fire danger! 🚨🔥")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Modified scroll to results
        st.markdown("""
            <script>
                setTimeout(function() {
                    document.querySelector('div[data-testid="stVerticalBlock"]').scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }, 100);
            </script>
            """, unsafe_allow_html=True)
        
# Footer
st.markdown("""
<div style='text-align: center; margin-top: 2rem;'>
    <p style='color: #666; font-size: 0.9rem;'>Developed by Megha Chakraborty</p>
</div>
""", unsafe_allow_html=True)