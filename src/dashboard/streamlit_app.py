"""
Streamlit Dashboard for Heart Disease Prediction Model
Make predictions using patient medical data
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="pulse",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Heart Disease Prediction")
st.markdown("Enter patient information to get heart disease prediction")
st.markdown("---")

# API URL configuration
API_URL = "http://localhost:8000"

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["random_forest", "logistic_regression"],
    format_func=lambda x: "Random Forest" if x == "random_forest" else "Logistic Regression"
)

st.info(f"Using **{model_type.replace('_', ' ').title()}** model for predictions")
st.markdown("---")

# Feature input columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age (years)", 29, 77, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 60, 202, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)

with col3:
    slope = st.selectbox("ST Slope", options=[0, 1, 2])
    ca = st.selectbox("Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])
    st.write("")
    st.write("")
    predict_button = st.button("Predict", use_container_width=True)

if predict_button:
    # Prepare input data
    patient_data = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": float(fbs),
        "restecg": float(restecg),
        "thalach": float(thalach),
        "exang": float(exang),
        "oldpeak": float(oldpeak),
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal),
        "model_type": model_type
    }
    
    try:
        # Make API call
        response = requests.post(
            f"{API_URL}/predict",
            json=patient_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", 0)
            confidence = result.get("confidence", 0)
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 0:
                    st.success("No Heart Disease Detected")
                    st.metric("Risk Level", "Low", delta="Safe")
                else:
                    st.error("Heart Disease Likely")
                    st.metric("Risk Level", "High", delta="Alert")
            
            with result_col2:
                st.metric("Model Confidence", f"{confidence*100:.2f}%")
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 100], 'color': "gray"}],
                           'threshold': {
                               'line': {'color': "red", 'width': 4},
                               'thickness': 0.75,
                               'value': 90}}
                ))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.sidebar.markdown("Heart Disease Prediction MLOps Pipeline")
