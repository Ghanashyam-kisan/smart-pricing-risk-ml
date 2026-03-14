import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Smart Insurance Pricing", layout="centered")

st.title("🚗 Smart Insurance Pricing System")
st.write("Predict insurance price using Machine Learning")

# Load trained models
model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
kmeans = joblib.load("artifacts/clustering_model.pkl")

st.header("Enter Customer Details")

# User inputs
age = st.slider("Age", 18, 80, 35)
income = st.number_input("Annual Income", 10000, 200000, 50000)
credit_score = st.slider("Credit Score", 300, 850, 650)
claim_history = st.slider("Number of Past Claims", 0, 10, 1)
vehicle_value = st.number_input("Vehicle Value", 5000, 100000, 25000)


model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
kmeans = joblib.load("artifacts/clustering_model.pkl")

explainer = shap.TreeExplainer(model)

# Prediction button
if st.button("Predict Insurance Price"):

    # Create feature array
    features = np.array([[age, income, credit_score, claim_history, vehicle_value]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict risk cluster
    cluster = kmeans.predict(features_scaled)

    # Combine features with cluster
    final_features = np.column_stack((features_scaled, cluster))

    # Model prediction
    prediction = model.predict(final_features)[0]

    # Convert USD → INR
    usd_to_inr = 83
    prediction_inr = prediction * usd_to_inr

    # Show price
    st.subheader("Predicted Insurance Price")
    st.success(f"₹ {prediction_inr:,.2f}  (~ ${prediction:,.2f})")

    # Risk category
    cluster_id = int(cluster[0])

    st.subheader("Risk Category")

    if cluster_id == 0:
        st.success("🟢 Low Risk Customer")
    elif cluster_id == 1:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.error("🔴 High Risk Customer")

    st.subheader("Model Explanation (Why this price?)")

    feature_names = [
    "Age",
    "Income",
    "Credit Score",
    "Claim History",
    "Vehicle Value",
    "Risk Cluster"
]

    shap_values = explainer.shap_values(final_features)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=final_features[0],
        feature_names=feature_names
    )
)

    st.pyplot(fig)