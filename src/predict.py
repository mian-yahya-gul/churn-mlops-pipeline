"""
predict.py
Loads the latest registered MLflow model and runs inference
on new customer records.

Run:
    python src/predict.py
"""

import pandas as pd
import joblib
import mlflow

MODEL_NAME = "churn-rf-classifier"
ENCODERS_PATH = "models/label_encoders.pkl"


def load_latest_model(model_name: str = MODEL_NAME):
    """Loads the latest version of the registered model from the MLflow registry."""
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def preprocess_new_record(record: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df


def predict_churn(record: dict):
    model = load_latest_model()
    encoders = joblib.load(ENCODERS_PATH)

    df = preprocess_new_record(record, encoders)
    # ensure column order matches training
    expected_cols = [
        "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "internet_service", "tech_support",
        "payment_method", "num_support_calls"
    ]
    df = df[expected_cols]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    sample_customer = {
        "tenure_months": 4,
        "monthly_charges": 85.50,
        "total_charges": 342.00,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "payment_method": "Electronic check",
        "num_support_calls": 5,
    }

    result = predict_churn(sample_customer)
    print(f"Sample prediction: {result}")
