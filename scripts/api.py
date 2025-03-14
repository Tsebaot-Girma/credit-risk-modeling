# scripts/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# Load model
model_path = Path(__file__).parent.parent / "models" / "credit_risk_model.pkl"
model = joblib.load(model_path)

# Debug endpoint (MUST come after model loading)
@app.get("/model_info")
def get_model_info():
    return {
        "feature_names": model.feature_names_in_.tolist(),  # Requires scikit-learn >= 1.0
        "n_features": model.n_features_in_
    }

# Prediction endpoint
class CustomerData(BaseModel):
    Amount: float
    TransactionCount: int
    Recency: int

@app.post("/predict")
def predict(data: CustomerData):
    features = [[data.Amount, data.TransactionCount, data.Recency]]
    prediction = model.predict(features)
    return {"risk_score": int(prediction[0])}