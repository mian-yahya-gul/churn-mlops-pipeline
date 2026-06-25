"""
preprocess.py
Loads raw churn data and prepares train/test splits with encoding.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

RAW_DATA_PATH = "data/customer_churn.csv"
PROCESSED_DIR = "data/processed"
ENCODERS_PATH = "models/label_encoders.pkl"

CATEGORICAL_COLS = ["contract_type", "internet_service", "tech_support", "payment_method"]
TARGET_COL = "churn"
DROP_COLS = ["customer_id"]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    return df, encoders


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    df = load_raw_data()
    df = df.drop(columns=DROP_COLS)

    df, encoders = encode_categoricals(df, fit=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    joblib.dump(encoders, ENCODERS_PATH)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f} | Test churn rate: {y_test.mean():.3f}")
