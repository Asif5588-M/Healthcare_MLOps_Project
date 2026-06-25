import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipeline.prediction_pipeline import PredictionPipeline

# Initializing FastAPI application
app = FastAPI(
    title="Healthcare Diabetes Prediction MLOps Service",
    description="Production-grade API endpoint for real-time patient diagnosis.",
    version="1.0.0"
)

# Initializing the inference prediction pipeline layer
prediction_pipeline = PredictionPipeline()

# Defining the secure Pydantic request data structure validation schema
class PatientDataSchema(BaseModel):
    pregnancies: float = Field(..., description="Number of times pregnant", example=1.0)
    glucose: float = Field(..., description="Plasma glucose concentration", example=85.0)
    blood_pressure: float = Field(..., description="Diastolic blood pressure (mm Hg)", example=66.0)
    skin_thickness: float = Field(..., description="Triceps skin fold thickness (mm)", example=29.0)
    insulin: float = Field(..., description="2-Hour serum insulin (mu U/ml)", example=0.0)
    bmi: float = Field(..., description="Body mass index (weight in kg/(height in m)^2)", example=26.6)
    diabetes_pedigree: float = Field(..., description="Diabetes pedigree function score", example=0.351)
    age: float = Field(..., description="Age in years", example=31.0)


@app.get("/")
async def root_endpoint():
    """Health check monitoring endpoint"""
    return {"status": "healthy", "service": "Healthcare Diabetes Prediction API"}


@app.post("/predict")
async def predict_endpoint(patient_data: PatientDataSchema):
    """
    Inference endpoint to compute predictive classification metrics for patients.
    Returns: 1 for Diabetic status, 0 for Healthy status.
    """
    try:
        # Extracting raw values structured inside the validated request body schema
        features_list = [
            patient_data.pregnancies,
            patient_data.glucose,
            patient_data.blood_pressure,
            patient_data.skin_thickness,
            patient_data.insulin,
            patient_data.bmi,
            patient_data.diabetes_pedigree,
            patient_data.age
        ]
        
        # Invoking prediction pipeline execution step
        prediction_result = prediction_pipeline.predict(features_list)
        
        # Formatting human-readable structural response mapping
        diagnosis_label = "Diabetic" if prediction_result == 1 else "Healthy"
        
        return {
            "prediction": prediction_result,
            "diagnosis": diagnosis_label,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal inference failure during parsing execution: {str(e)}"
        )

if __name__ == "__main__":
    # Launching local development web server routing architecture
    uvicorn.run("app.py:app", host="127.0.0.1", port=8000, reload=True)
