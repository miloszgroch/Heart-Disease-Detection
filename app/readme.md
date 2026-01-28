# Heart Disease Detection Project

## Overview

This project predicts the presence of heart disease using clinical data.
It demonstrates a complete Machine Learning workflow including:
- Exploratory Data Analysis (EDA)
- Feature engineering and scaling
- Model training and evaluation
- Model explainability
- Interactive deployment using Streamlit

Dataset: Kaggle Heart Disease Dataset

Models used:
- Random Forest
- Gradient Boosting

---

## Project Structure

heart-disease-detection/
├── data/
│   └── heart.csv
├── notebooks/
│   └── 01_heart_disease_analysis.ipynb
├── app.py
├── requirements.txt
└── README.md

---

## Notebook

The Jupyter Notebook focuses on:
- Data exploration and visualization
- Correlation analysis
- Scaled vs non-scaled model comparison
- Model evaluation
- Feature importance and SHAP explanations
- Conclusions and possible improvements

To run the notebook:

```bash
jupyter notebook notebooks/01_heart_disease_analysis.ipynb
```

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

## Notes

- The app uses relative paths (data/heart.csv) for portability.
- Models are trained at runtime for demonstration purposes.
- Feature scaling is applied only where necessary.

## Possible Improvements

- Add cross-validation
- Optimize hyperparameters
- Improve recall for medical relevance
- Deploy to Streamlit Cloud
- Add model monitoring
