import streamlit as st
from src.pipeline.prediction_pipeline import PredictionPipeline

# Page UI configurations
st.set_page_config(
    page_title="Diabetes Risk Assessment Portal",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Patient Health Diagnostic Portal")
st.markdown("### Real-time Machine Learning Classification Diagnostics")
st.write("Input patient laboratory metrics below to securely fetch diagnostic predictions directly from our integrated MLOps inference pipeline.")

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
    # Packaging request features list for direct prediction pipeline execution
    features_list = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]
    
    try:
        with st.spinner("Executing live inference layer models... Please wait."):
            # Invoking prediction pipeline directly (Cloud Architecture Alignment)
            pipeline = PredictionPipeline()
            prediction_result = pipeline.predict(features_list)
            
            diagnosis = "Diabetic" if prediction_result == 1 else "Healthy"
            
            st.markdown("---")
            st.subheader("Diagnostic Evaluation Result")
            
            # Visual conditional rendering depending on classification output status
            if diagnosis == "Diabetic":
                st.error(f"Prediction Alert: The patient is classified as **{diagnosis}**.")
            else:
                st.success(f"Prediction Confirmed: The patient is classified as **{diagnosis}**.")
                
    except Exception as e:
        st.error(f"Inference Error: Runtime pipeline execution failed: {str(e)}")
