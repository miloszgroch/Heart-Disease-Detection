# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import joblib

# ===============================
# Streamlit App Title
# ===============================
st.title("Heart Disease Detection")

st.write("""
This interactive app allows you to:
- Explore the Heart Disease dataset
- Train and test models (Random Forest, Gradient Boosting)
- See feature importance and ROC curves
""")

# ===============================
# 1️⃣ Load Data
# ===============================
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data

data = load_data(r'C:\Users\mgroch\OneDrive - Interpublic\Desktop\Other\HDD\heart.csv')

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ===============================
# 2️⃣ EDA
# ===============================
st.subheader("Basic Statistics")
st.dataframe(data.describe())

st.subheader("Missing Values")
st.dataframe(data.isnull().sum())


numeric_data = data.select_dtypes(include=[np.number])
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ===============================
# 3️⃣ Preprocessing
# ===============================
X = data.drop(columns=["target"])
y = data["target"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=np.number))

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===============================
# 4️⃣ Model Training
# ===============================
st.subheader("Train Models")

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
st.write(f"Random Forest Accuracy: {rf_acc:.4f}")

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
st.write(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# ===============================
# 5️⃣ Feature Importance
# ===============================
st.subheader("Feature Importance - Random Forest")

if hasattr(rf_model, "feature_importances_"):
    feature_names = X.select_dtypes(include=np.number).columns
    importances = rf_model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

# ===============================
# 6️⃣ ROC Curve
# ===============================
st.subheader("ROC Curve")

fig, ax = plt.subplots(figsize=(8,6))

for name, model in {"Random Forest": rf_model, "Gradient Boosting": gb_model}.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} AUC: {auc:.4f}")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve")
ax.legend(loc="best")
st.pyplot(fig)

# ===============================
# 7️⃣ Interactive Prediction
# ===============================
st.subheader("Predict Heart Disease for a New Patient")

age = st.slider("Age", int(data.age.min()), int(data.age.max()), 54)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.slider("Resting Blood Pressure", int(data.trestbps.min()), int(data.trestbps.max()), 130)
chol = st.slider("Cholesterol", int(data.chol.min()), int(data.chol.max()), 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", [0,1])
restecg = st.selectbox("Resting ECG results (0,1,2)", [0,1,2])
thalach = st.slider("Max Heart Rate", int(data.thalach.min()), int(data.thalach.max()), 150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0,1])
oldpeak = st.slider("ST depression induced by exercise", float(data.oldpeak.min()), float(data.oldpeak.max()), 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (0,1,2)", [0,1,2])
ca = st.slider("Number of major vessels (0-3) colored by fluoroscopy", 0, 3, 0)
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1,2,3])

input_df = pd.DataFrame({
    "age":[age],
    "sex":[sex],
    "cp":[cp],
    "trestbps":[trestbps],
    "chol":[chol],
    "fbs":[fbs],
    "restecg":[restecg],
    "thalach":[thalach],
    "exang":[exang],
    "oldpeak":[oldpeak],
    "slope":[slope],
    "ca":[ca],
    "thal":[thal]
})

prediction = rf_model.predict(input_df)
prob = rf_model.predict_proba(input_df)[0,1]

st.write(f"Prediction: {'Heart Disease' if prediction[0]==1 else 'No Heart Disease'}")
st.write(f"Probability: {prob:.2f}")
