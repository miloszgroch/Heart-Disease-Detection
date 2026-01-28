# Heart Disease Detection Project

## Overview

This project aims to predict the presence of heart disease using clinical features. It showcases a complete ML pipeline from exploratory data analysis (EDA) to model evaluation, suitable for GitHub presentation.

**Dataset:** Kaggle Heart Disease Dataset

**Models:** Random Forest, Gradient Boosting

**Notebook:** `notebooks/01_heart_disease_analysis.ipynb`

**Dashboard:** Streamlit app in `app/HDD-DEPLOY.py`  

---

## Project Structure

```
heart-disease-detection/
├── data/
│   └── heart.csv                    
├── notebooks/
│   └── 01_heart_disease_analysis.ipynb  
├── src/
│   ├── data.py                       
│   ├── models.py                     
│   └── visualizations.py             
├── app/
│   └── HDD-DEPLOY.py               
├── models/
│   └── best_model.pkl               
├── requirements.txt                  
├── README.md
└── .gitignore

```

---

## How to Run

1. Clone the repository.
2. Make sure Python 3.x is installed.
3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Open the notebook:

```bash
jupyter notebook notebooks/HDD - Portfolio.ipynb
```

5. Follow the notebook cells step by step.

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

## How to Run the App

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at:
http://localhost:8501

---

## Key Steps in Notebook/Dashboard

1. **Data Loading**: Read the CSV dataset.
2. **EDA**: Summary statistics, missing value check, correlation heatmap.
3. **Preprocessing**: Split data into train/test, scale numerical features.
4. **4. Model Training: Scaled vs Non-Scaled Models**: Train models on scaled features and compare their performance with scale-insensitive models
5. **Hyperparameter Tuning**: Tune Random Forest using GridSearchCV.
6. **Evaluation**: Accuracy, classification report.
7. **Feature Importance**: Visualize feature contribution.
8. **Summary & Insights**: Key results and possible improvements.

---

## Possible Extensions

* ROC AUC and confusion matrix visualizations.
* Focus on recall/precision for medical relevance.
* Use SHAP or LIME for model explainability.
* Cross-validation for robust evaluation.

---

**Author:** Miłosz Groch
**Date:** 2026-01-27
