from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Fraud Detection API",
    description="LSTM-Autoencoder based credit card fraud detection",
    version="1.0.0"
)

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float

class SequencePayload(BaseModel):
    transactions: List[Transaction]

# Load artifacts at startup
print("Loading model artifacts...")
model = load_model("models/autoencoder.keras", compile=False)
scaler = joblib.load("models/scaler.pkl")
threshold = float(open("models/threshold.txt").read())
print("Model loaded successfully.")

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running. Go to /docs for interactive documentation.",
        "status": "online",
        "threshold": threshold
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(payload: SequencePayload):
    if len(payload.transactions) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 transactions required.")

    # Convert to DataFrame
    df = pd.DataFrame([t.dict() for t in payload.transactions])
    df['Amount_scaled'] = scaler.transform(df[['Amount']])
    features = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']
    X = df[features].values.reshape(1, 10, 29)

    # Reconstruct and compute error
    reconstruction = model.predict(X, verbose=0)
    error = float(np.mean(np.square(X - reconstruction)))

    is_fraud = error > threshold
    return {
        "is_fraud": is_fraud,
        "score": error,
        "threshold": threshold,
        "message": "Fraud detected" if is_fraud else "Transaction approved"
    }
