import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

st.title("Heart Disease Detection")

st.write("""
Explore dataset, visualize features, and run predictions using trained models.
""")

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "heart.csv"
    return pd.read_csv(data_path)

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ===============================
# EDA
# ===============================
st.subheader("Basic Statistics")
st.dataframe(data.describe())

st.subheader("Missing Values")
st.dataframe(data.isnull().sum())

numeric_data = data.select_dtypes(include=[np.number])

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ===============================
# Preprocessing
# ===============================
X = data.drop(columns=["target"])
y = data["target"]

X_numeric = X.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# Load models from MLflow
# ===============================
def load_model(run_name: str):
    experiment = mlflow.get_experiment_by_name("heart-disease-classification")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    run_id = runs[runs["tags.mlflow.runName"] == run_name].iloc[0].run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

rf_model = load_model("RandomForest_tuned")
gb_model = load_model("GradientBoosting")

st.subheader("Model Performance")

for name, model in {
    "Random Forest (tuned)": rf_model,
    "Gradient Boosting": gb_model
}.items():
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    st.write(f"{name} Accuracy: {acc:.4f}")

# ===============================
# Feature Importance
# ===============================
st.subheader("Feature Importance - Random Forest")

if hasattr(rf_model, "feature_importances_"):
    feature_names = X_numeric.columns
    importances = rf_model.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ===============================
# ROC Curve
# ===============================
st.subheader("ROC Curve")

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} AUC: {auc:.4f}")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# ===============================
# Prediction
# ===============================
st.subheader("Predict Heart Disease")

age = st.slider("Age", int(data.age.min()), int(data.age.max()), 54)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", int(data.trestbps.min()), int(data.trestbps.max()), 130)
chol = st.slider("Cholesterol", int(data.chol.min()), int(data.chol.max()), 250)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", int(data.thalach.min()), int(data.thalach.max()), 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", float(data.oldpeak.min()), float(data.oldpeak.max()), 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.slider("CA", 0, 3, 0)
thal = st.selectbox("Thal", [1, 2, 3])

input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

input_scaled = scaler.transform(input_df)

prediction = rf_model.predict(input_scaled)
prob = rf_model.predict_proba(input_scaled)[0, 1]

st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
st.write(f"Probability: {prob:.2f}")