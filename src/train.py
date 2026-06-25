"""
train.py
Trains a churn classifier with full MLflow experiment tracking:
params, metrics, model artifact, and confusion matrix.

Run:
    python src/train.py --n_estimators 200 --max_depth 8
"""

import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

PROCESSED_DIR = "data/processed"
EXPERIMENT_NAME = "customer-churn-prediction"


def load_processed_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, out_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Churn Prediction")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def train(n_estimators: int, max_depth: int, min_samples_split: int):
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_processed_data()

    with mlflow.start_run():
        # ---- log params ----
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # ---- train ----
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ---- evaluate ----
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # ---- artifacts ----
        os.makedirs("models", exist_ok=True)
        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)

        feature_importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        feature_importance.to_csv("models/feature_importance.csv")
        mlflow.log_artifact("models/feature_importance.csv")

        # ---- log + register model ----
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn-rf-classifier",
        )

        print(f"Run complete. Metrics: {metrics}")
        print(f"Top features:\n{feature_importance.head(5)}")

        return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model with MLflow tracking")
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=4)
    args = parser.parse_args()

    train(args.n_estimators, args.max_depth, args.min_samples_split)
