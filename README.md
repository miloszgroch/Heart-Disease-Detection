# Heart Disease Detection Project

## Overview

This project aims to predict the presence of heart disease using clinical features. It showcases a complete ML pipeline from exploratory data analysis (EDA) to model evaluation, suitable for GitHub presentation.

## Problem

Heart disease is one of the leading causes of death worldwide.
This project aims to assist in early detection using machine learning models trained on clinical data.

---

**Dataset:** Kaggle Heart Disease Dataset

**Models:** Random Forest, Gradient Boosting

**Notebook:** `notebooks/01_heart_disease_analysis.ipynb`

**Dashboard:** Streamlit app in `app/streamlit_app.py`  

---

## Project Structure

```
heart-disease-detection/
├── data/
│   └── heart.csv                    
├── notebooks/
│   └── 01_heart_disease_analysis.ipynb  
├── src/
│   ├── train.py                                    
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── correlation_heatmap.png               
├── models/
│   └── best_model.pkl               
├── requirements.txt                  
├── README.md
├── Dockerfile
└── .gitignore

<br>

---

<br>

## 📊 Results

The models were evaluated on a held-out test set (20% of the data, 205 samples).

### 🏆 Best Model: Random Forest (Tuned)

- **Accuracy:** 0.985  
- **Precision (class 1):** 1.00  
- **Recall (class 1):** 0.97  
- **F1-score:** 0.99  

The tuned Random Forest model achieved the best performance, with near-perfect classification results and a strong balance between precision and recall.

---

### 📈 Model Comparison

| Model                     | Accuracy | Precision (avg) | Recall (avg) | F1-score (avg) |
|--------------------------|----------|-----------------|--------------|----------------|
| Random Forest (baseline) | 0.985    | 0.99            | 0.99         | 0.99           |
| Gradient Boosting        | 0.932    | 0.93            | 0.93         | 0.93           |
| Random Forest (tuned)    | 0.985    | 0.99            | 0.99         | 0.99           |

---

### 🔍 Key Insights

- Random Forest significantly outperformed Gradient Boosting on this dataset.
- Hyperparameter tuning did not improve accuracy further, indicating that the baseline model was already well-fitted.
- The model shows **very high predictive performance**, but this may suggest:
  - potential dataset simplicity
  - or slight risk of overfitting

---

### 🧠 Feature Importance

The most influential features for prediction include:
- `oldpeak` (ST depression)
- `cp` (chest pain type)
- `thalach` (maximum heart rate achieved)
- `ca` (number of major vessels)
- `exang` (exercise-induced angina)

These features align well with known clinical indicators of heart disease.

<br>

---

<br>

## 🐳 Run with Docker

This project can be easily run using Docker, ensuring a consistent environment.

### 1. Clone repository

```bash
git clone https://github.com/miloszgroch/Heart-Disease-Detection.git
cd Heart-Disease-Detection
```

### 2. Build the Docker image

```bash
docker build -t heart-disease-app .
```

### 3. Run the container

```bash
docker run -p 8501:8501 heart-disease-app
```

### 4. Open the app

Go to:
http://localhost:8501


<br>

---

<br>

## How to Run without Docker

1. Clone the repository.
2. Make sure Python 3.x is installed.
3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Open the notebook:

```bash
jupyter notebook notebooks/01_heart_disease_analysis.ipynb
```

5. Follow the notebook cells step by step.

<br>

---

<br>

## Streamlit Web Application

An interactive Streamlit application is included to present the project results
in a user-friendly way.

Features:
- Dataset preview and statistics
- Correlation heatmap
- Model training and evaluation
- Feature importance visualization
- ROC curve comparison
- Interactive heart disease prediction for a new patient


1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

3. Open your browser at:
http://localhost:8501

<br>

---

<br>


## MLflow Experiment Tracking

The project uses MLflow to track experiments:

- Random Forest (baseline)
- Gradient Boosting
- Random Forest (tuned with GridSearch)

Tracked:
- metrics (accuracy)
- models
- feature importance plots

Run:
mlflow ui
<br>

---

<br>
## Key Steps in Notebook/Dashboard

1. **Data Loading**: Read the CSV dataset.
2. **EDA**: Summary statistics, missing value check, correlation heatmap.
3. **Preprocessing**: Split data into train/test, scale numerical features.
4. **Model Training: Scaled vs Non-Scaled Models**: Train models on scaled features and compare their performance with scale-insensitive models
5. **Hyperparameter Tuning**: Tune Random Forest using GridSearchCV.
6. **Evaluation**: Accuracy, classification report.
7. **Feature Importance**: Visualize feature contribution.
8. **Summary & Insights**: Key results and possible improvements.

<br>

---

<br>
## Possible Extensions

* ROC AUC and confusion matrix visualizations.
* Focus on recall/precision for medical relevance.
* Use SHAP or LIME for model explainability.
* Cross-validation for robust evaluation.

<br>

---

<br>
**Author:** Miłosz Groch
**Date:** 2026-01-27
