Smart Pricing & Risk Assessment using Machine Learning
Overview

This project implements an end-to-end machine learning pipeline for risk-based insurance pricing using customer financial and behavioral data.

The system combines unsupervised learning (K-Means clustering) for customer risk segmentation with supervised regression models (Random Forest & XGBoost) to predict optimal pricing.

An interactive dashboard built with Streamlit allows users to input customer details and obtain:

predicted insurance price

risk category

model explanation

analytics dashboard

This project demonstrates how hybrid ML systems can support pricing decisions in financial and insurance domains.

Key Features

End-to-end ML pipeline

Synthetic dataset generation

Customer risk segmentation using K-Means clustering

Price prediction using Random Forest & XGBoost

Hyperparameter tuning with GridSearchCV

Model explainability using SHAP

Interactive Streamlit dashboard

Prediction history tracking

Price distribution analytics

Feature importance visualization

Technologies Used

Python

Scikit-learn

XGBoost

Pandas

NumPy

Matplotlib

SHAP (Explainable AI)

Streamlit

Dataset

The project uses a synthetic but statistically realistic dataset designed to simulate insurance pricing scenarios.

Features include:

Age

Income

Credit Score

Claim History

Vehicle Value

Target variable:

Risk-adjusted insurance price

Synthetic data generation ensures reproducibility while preserving real-world patterns.

Project Architecture
smart-pricing-risk-ml
│
├── data/
│   ├── generate_dataset.py
│   └── pricing_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── clustering.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── models.py
│   └── tuning.py
│
├── artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── clustering_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── main.py
├── requirements.txt
└── README.md
Machine Learning Pipeline

Data Generation

Synthetic dataset simulating insurance customers

Data Preprocessing

Feature scaling

Data preparation

Risk Segmentation

K-Means clustering groups customers into risk segments

Model Training

Random Forest Regressor

XGBoost Regressor

Hyperparameter Tuning

GridSearchCV optimization

Model Evaluation

MAE

RMSE

R² score

Explainability

SHAP analysis for prediction interpretation

Dashboard (Streamlit)

The interactive dashboard allows users to:

input customer details

predict insurance pricing

view risk classification

see SHAP-based explanation

track prediction history

analyze price distribution

Example workflow:

User Input
   ↓
ML Model Prediction
   ↓
Risk Classification
   ↓
SHAP Explanation
   ↓
Analytics Dashboard
Installation

Clone the repository:

git clone https://github.com/Ghanashyam-kisan/smart-pricing-risk-ml.git
cd smart-pricing-risk-ml

Install dependencies:

pip install -r requirements.txt
Run the ML Pipeline

Train models and generate artifacts:

python main.py
Run the Dashboard
streamlit run app/streamlit_app.py

Then open:

http://localhost:8501
Example Output

The system produces:

Predicted Insurance Price

Risk Category (Low / Medium / High)

SHAP explanation for feature contributions

Prediction history dashboard

Price distribution analytics

Learning Outcomes

This project demonstrates:

ML pipeline design

Feature engineering

Hybrid ML systems (clustering + regression)

Explainable AI

Interactive ML dashboards

End-to-end deployment workflow

Author

Ghanashyam Kisan

B.Tech Final Year | Machine Learning & Data Science Enthusiast

GitHub:
https://github.com/Ghanashyam-kisan