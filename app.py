import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from model_pipeline import load_model

MODEL_PATH = "classifier.joblib"
FEATURES = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

app = FastAPI(title="Churn prediction API")

try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"WARNING: {MODEL_PATH} not found. Train the model first.")


class Customer(BaseModel):
    """Features of one customer (Gender: 0 = Female, 1 = Male)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 619,
                "Gender": 0,
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            }
        }
    )

    CreditScore: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def home():
    """Check that the API is running."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    """Predict if a customer leaves the bank (1) or stays (0).

    Args:
        customer: the features of the customer.
    Returns:
        dict with the prediction and the probability of leaving.
    Raises:
        HTTPException: if the model is missing or the prediction fails.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        data = pd.DataFrame([customer.model_dump()], columns=FEATURES)
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")
    return {"prediction": prediction, "churn_probability": round(probability, 3)}
