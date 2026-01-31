# Smart Pricing & Risk Assessment using Machine Learning

## 📌 Overview
This project implements an end-to-end machine learning pipeline for
risk-based pricing using customer financial and behavioral data.
The system combines unsupervised learning for risk segmentation with
supervised regression models to predict optimal pricing.

The goal is to demonstrate how hybrid ML techniques can improve pricing
accuracy and explainability in real-world decision-making systems.

---

## 🧠 Key Concepts Used
- Exploratory Data Analysis (EDA)
- Feature Scaling & Preprocessing
- Risk Segmentation using K-Means Clustering
- Regression Models for Pricing Prediction
- Random Forest & XGBoost
- Hyperparameter Tuning (GridSearchCV)
- Model Explainability using Feature Importance
- Comparative Performance Analysis

---

## 📊 Dataset
- Synthetic but statistically realistic dataset
- Simulates insurance/financial pricing scenarios
- Features include:
  - Age
  - Income
  - Credit Score
  - Claim History
  - Vehicle Value
- Target:
  - Risk-adjusted Price

Synthetic data generation ensures reproducibility while preserving
real-world patterns.

---

## ⚙️ Project Architecture

data/ → Dataset generation & storage
notebooks/ → Exploratory Data Analysis (EDA)
src/ → Modular ML pipeline
main.py → End-to-end execution