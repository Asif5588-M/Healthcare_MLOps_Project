import streamlit as pd
import streamlit as st
import requests

# Page UI configurations
st.set_page_config(
    page_title="Diabetes Risk Assessment Portal",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Patient Health Diagnostic Portal")
st.markdown("### Real-time Machine Learning Classification Diagnostics")
st.write("Input patient laboratory metrics below to securely fetch diagnostic predictions from our FastAPI microservice backend framework.")

st.markdown("---")

# Creating structural form container inputs
with st.form("diagnostic_entry_form"):
    st.subheader("Patient Clinical Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
        blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0.0, max_value=150.0, value=70.0)
        skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
        
    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=79.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function Score", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age (Years)", min_value=1.0, max_value=120.0, value=30.0, step=1.0)

    # Submission Action Button
    submit_button = st.form_submit_button(label="Analyze Diagnostic Data")

# When the user submits the metrics
if submit_button:
    # Packaging request body structural JSON schema to match FastAPI specifications
    payload = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "age": age
    }
    
    # Target endpoint URL mapping
    API_URL = "http://127.0.0"
    
    try:
        with st.spinner("Communicating with model server endpoints... Please wait."):
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result_data = response.json()
                diagnosis = result_data.get("diagnosis")
                
                st.markdown("---")
                st.subheader("Diagnostic Evaluation Result")
                
                # Visual conditional rendering depending on classification output status
                if diagnosis == "Diabetic":
                    st.error(f"Prediction Alert: The patient is classified as **{diagnosis}**.")
                else:
                    st.success(f"Prediction Confirmed: The patient is classified as **{diagnosis}**.")
            else:
                st.error(f"API Error: Received invalid response status code {response.status_code}")
                
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Please ensure your FastAPI server (`app.py`) is running actively on port 8000!")
