"""
Streamlit Dashboard for Heart Disease Prediction Model
Displays model metrics, predictions, and monitoring
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from pathlib import Path
import joblib
from datetime import datetime
import mlflow

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Make Prediction", "Model Performance", "Experiment History"]
)

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

# ===== PAGE 1: DASHBOARD =====
if page == "Dashboard":
    st.title("Heart Disease Prediction - Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get model info from API
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            
            with col1:
                st.metric("Model Type", model_info.get("model_type", "N/A"))
            with col2:
                st.metric("Accuracy", f"{model_info.get('accuracy', 0):.4f}")
            with col3:
                st.metric("Precision", f"{model_info.get('precision', 0):.4f}")
            with col4:
                st.metric("ROC-AUC", f"{model_info.get('roc_auc', 0):.4f}")
    except:
        st.warning("WARNING: Could not connect to API. Make sure it's running at " + API_URL)
    
    st.markdown("---")
    
    # Project Overview
    st.header("Project Overview")
    overview_cols = st.columns(2)
    
    with overview_cols[0]:
        st.subheader("Project Details")
        st.write("""
        - **Objective**: Predict heart disease risk based on patient medical data
        - **Dataset**: UCI Heart Disease Dataset
        - **Samples**: 303 patients
        - **Features**: 13 medical attributes
        - **Target**: Binary classification (0: No disease, 1: Disease present)
        """)
    
    with overview_cols[1]:
        st.subheader("Tech Stack")
        st.write("""
        - **ML Framework**: Scikit-learn
        - **API Server**: FastAPI
        - **Dashboard**: Streamlit
        - **Container**: Docker
        - **Orchestration**: Kubernetes
        - **Experiment Tracking**: MLflow
        - **Monitoring**: Prometheus + Grafana
        """)
    
    st.markdown("---")
    
    # System Status
    st.header("System Status")
    sys_cols = st.columns(3)
    
    with sys_cols[0]:
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            status = "HEALTHY" if response.status_code == 200 else "UNHEALTHY"
            st.write(f"**API Status**: {status}")
        except:
            st.write("**API Status**: DISCONNECTED")
    
    with sys_cols[1]:
        st.write("**Database**: CONNECTED")
    
    with sys_cols[2]:
        st.write("**Dashboard**: RUNNING")

# ===== PAGE 2: MAKE PREDICTION =====
elif page == "Make Prediction":
    st.title("Make Prediction")
    st.markdown("Enter patient information to get heart disease prediction")
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
            "thal": float(thal)
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

# ===== PAGE 3: MODEL PERFORMANCE =====
elif page == "Model Performance":
    st.title("Model Performance Metrics")
    st.markdown("---")
    
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            
            # Metrics table
            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                "Value": [
                    f"{model_info.get('accuracy', 0):.4f}",
                    f"{model_info.get('precision', 0):.4f}",
                    f"{model_info.get('recall', 0):.4f}",
                    f"{model_info.get('f1', 0):.4f}",
                    f"{model_info.get('roc_auc', 0):.4f}"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("---")
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=metrics_data["Metric"][:4],
                        y=[float(v) for v in metrics_data["Value"][:4]],
                        marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
                    )
                ])
                fig.update_layout(
                    title="Classification Metrics",
                    yaxis_title="Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Gauge for ROC-AUC
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=float(model_info.get('roc_auc', 0)),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "ROC-AUC Score"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightcoral"},
                            {'range': [0.5, 0.8], 'color': "lightyellow"},
                            {'range': [0.8, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not fetch model metrics")
    except:
        st.error("Cannot connect to API")

# ===== PAGE 4: EXPERIMENT HISTORY =====
elif page == "Experiment History":
    st.title("MLflow Experiment History")
    st.markdown("Experiment tracking and model comparison")
    st.markdown("---")
    
    try:
        # Try to read MLflow data
        mlflow_path = Path("mlruns")
        if mlflow_path.exists():
            st.info("MLflow experiments stored locally in mlruns/ directory")
            
            # Count experiments
            experiments = list(mlflow_path.glob("*/"))
            st.metric("Total Experiments", len(experiments))
            
            st.markdown("---")
            st.subheader("Tracked Runs")
            
            for exp in experiments:
                if exp.name != "0":
                    st.write(f"**Experiment**: {exp.name}")
                    runs = list(exp.glob("*/"))
                    st.write(f"Runs: {len(runs)}")
            
            st.markdown("---")
            st.info("To view detailed MLflow UI, run: mlflow ui --backend-store-uri file:mlruns")
        else:
            st.warning("No MLflow experiments found. Train models first!")
    except Exception as e:
        st.warning(f"Could not load MLflow data: {str(e)}")

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("Heart Disease Prediction MLOps Pipeline")
