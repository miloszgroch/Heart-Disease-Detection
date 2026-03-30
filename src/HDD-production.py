import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_experiment("heart-disease-classification")
mlflow.sklearn.autolog()

# Log Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Reading Data from: {path}")
    try:
        data = pd.read_csv(path)
        logging.info(f"Read {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        logging.error(f"Error while reading data: {e}")
        raise


def perform_eda(data: pd.DataFrame) -> None:
    logger.info("(EDA)...")

    print("\n Basic statistics:\n", data.describe())
    print("\n NAs:\n", data.isnull().sum())

    numerical_data = data.select_dtypes(include=["number"])

    print("\n Correlations (top 10):\n", numerical_data.corr().abs().unstack()
          .sort_values(ascending=False).drop_duplicates().head(10))

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(numerical_data.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation matrix (Numerical features)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()



def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    logger.info("Split X i y")

    
    X = data.drop(columns=["target"])
    y = data["target"]

   
    X_numeric = X.select_dtypes(include=[np.number])

    logger.info("Train, test split")
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    logger.info("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_numeric.columns.tolist()



def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, GradientBoostingClassifier]:
    logging.info("Train Random Forest model")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    logging.info("Train Gradient Boosting model")
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)

    return rf, gb


def tune_hyperparameters(model, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray):
    logging.info("Start GridSearch")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> float:
    logging.info(f"Model Evaluation: {model_name}")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info(f"{model_name} Accuracy: {acc:.4f}")
    logging.info(f"{model_name} Classification Report:\n{report}")

    mlflow.log_metric(f"{model_name}_accuracy", acc)

    return acc


def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    logging.info(f"Draw feature importance: {model_name}")
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[sorted_idx], y=sorted_features)
            plt.title(f"Feature importance: {model_name}")
            plt.tight_layout()
            plt.savefig(f"feature_importance_{model_name}.png")
            mlflow.log_artifact(f"feature_importance_{model_name}.png")
            plt.close()
        else:
            logging.warning(f"Model {model_name} is not using feature_importances_.")
    except Exception as e:
        logging.error(f"Error drawing feature importance: {e}")



def main(data_path: str) -> None:
    data = load_data(data_path)
    perform_eda(data)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)

    # RANDOM FOREST BASELINE
    with mlflow.start_run(run_name="RandomForest_baseline"):
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        evaluate_model(rf_model, X_test, y_test, "rf_baseline")
        plot_feature_importance(rf_model, feature_names, "rf_baseline")

        mlflow.sklearn.log_model(rf_model, "model")

    # GRADIENT BOOSTING
    with mlflow.start_run(run_name="GradientBoosting"):
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train, y_train)

        evaluate_model(gb_model, X_test, y_test, "gb")
        plot_feature_importance(gb_model, feature_names, "gb")

        mlflow.sklearn.log_model(gb_model, "model")

    # TUNED RANDOM FOREST
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    with mlflow.start_run(run_name="RandomForest_tuned"):
        rf_best = tune_hyperparameters(
            RandomForestClassifier(random_state=42),
            rf_params,
            X_train,
            y_train
        )

        evaluate_model(rf_best, X_test, y_test, "rf_tuned")
        plot_feature_importance(rf_best, feature_names, "rf_tuned")

        mlflow.sklearn.log_model(rf_best, "model")


if __name__ == "__main__":
    main(r"C:/Users/mgroch/OneDrive - Interpublic/Desktop/Other/HDD/heart.csv")
