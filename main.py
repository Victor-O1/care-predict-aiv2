from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
import os
import uvicorn

# -----------------------------
# Configuration
# -----------------------------
MODELS_DIR = "saved_models"
target_cols = ["diabetes_risk", "hypertension_risk", "cvd_risk", "kidney_risk"]

# -----------------------------
# Load models and preprocessors (on startup)
# -----------------------------
try:
    models = {
        target: joblib.load(os.path.join(MODELS_DIR, f"{target}_xgb_model.pkl"))
        for target in target_cols
    }
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
except Exception as e:
    raise RuntimeError(f"❌ Failed to load models or preprocessors: {e}")

app = FastAPI(title="Health Risk Prediction API")

# -----------------------------
# Helper: preprocess new data
# -----------------------------
# Columns you dropped during training
target_cols = ["diabetes_risk", "hypertension_risk", "cvd_risk", "kidney_risk"]
drop_cols = [
    "Name", "location", "allergies", "Existing conditions",
    "past treatments", "Names", "dosages", "frequency", "side effect"
]
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    # Drop irrelevant + target columns
    df = df.drop(columns=target_cols + drop_cols, errors="ignore")

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: x if x in le.classes_ else "<UNK>")
            if "<UNK>" not in le.classes_:
                le.classes_ = np.append(le.classes_, "<UNK>")
            df[col] = le.transform(df[col].astype(str))

    # Scale numeric features
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )
    return df_scaled

# -----------------------------
# API Endpoint: Predict
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)

        if df.shape[0] != 1:
            raise HTTPException(status_code=400, detail="CSV must contain exactly 1 row.")

        # Preprocess
        X_new = preprocess_input(df)

        # Predict probabilities with each model
        results = {}
        for target, model in models.items():
            prob = model.predict_proba(X_new)[:, 1][0]
            results[target] = float(prob)

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# -----------------------------
# Entry Point (for running directly)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
