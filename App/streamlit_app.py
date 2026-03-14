import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

# ================= Page Config =================

st.set_page_config(page_title="Smart Insurance Pricing", layout="centered")

st.title("🚗 Smart Insurance Pricing System")
st.caption(
    "AI-powered insurance pricing with risk segmentation and explainable machine learning."
)

# ================= Load Models =================

model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
kmeans = joblib.load("artifacts/clustering_model.pkl")

explainer = shap.TreeExplainer(model)

feature_names = [
    "Age",
    "Income",
    "Credit Score",
    "Claim History",
    "Vehicle Value",
    "Risk Cluster"
]

# ================= Prediction History =================

if "history" not in st.session_state:
    st.session_state.history = []

# ================= Sidebar =================

st.sidebar.header("Model Insights")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="skyblue"
)

ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance")
ax.invert_yaxis()

st.sidebar.pyplot(fig)

# ================= User Inputs =================

st.header("Enter Customer Details")

age = st.slider("Age", 18, 80, 35)

income = st.number_input(
    "Annual Income",
    min_value=150000,
    max_value=2500000,
    value=700000
)

credit_score = st.slider("Credit Score", 300, 850, 650)

claim_history = st.slider(
    "Number of Past Claims",
    0,
    10,
    1
)

vehicle_value = st.number_input(
    "Vehicle Value",
    min_value=200000,
    max_value=3500000,
    value=900000
)

# ================= Prediction =================

if st.button("Predict Insurance Price"):

    features = np.array([[age, income, credit_score, claim_history, vehicle_value]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict cluster for model input
    cluster = kmeans.predict(features_scaled)

    final_features = np.column_stack((features_scaled, cluster))

    # Predict price
    prediction = model.predict(final_features)[0]

    usd_to_inr = 83
    prediction_inr = prediction * usd_to_inr

    st.subheader("Predicted Insurance Price")
    st.success(f"₹ {prediction_inr:,.2f} (~ ${prediction:,.2f})")

    # ================= Risk Logic =================

    risk_score = (
        claim_history * 2 +
        max(0, (700 - credit_score) / 100) +
        (age < 25)
    )

    st.subheader("Risk Category")

    if risk_score < 1.5:
        st.success("🟢 Low Risk Customer")
        risk_label = "Low Risk"
    elif risk_score < 3:
        st.warning("🟡 Medium Risk Customer")
        risk_label = "Medium Risk"
    else:
        st.error("🔴 High Risk Customer")
        risk_label = "High Risk"

    # ================= SHAP Explanation =================

    st.subheader("Model Explanation (Why this price?)")
    st.write("This chart explains how each feature influenced the prediction.")

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

    # ================= Save Prediction =================

    st.session_state.history.append({
        "Age": age,
        "Income": income,
        "Credit Score": credit_score,
        "Claims": claim_history,
        "Vehicle Value": vehicle_value,
        "Predicted Price (₹)": round(prediction_inr, 2),
        "Risk Category": risk_label
    })

# ================= Prediction History =================

st.subheader("Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)

    st.subheader("Price Distribution")

    prices = history_df["Predicted Price (₹)"]

    fig, ax = plt.subplots()

    if len(prices) > 1:
        # Histogram when multiple predictions exist
        ax.hist(
            prices,
            bins=min(10, len(prices)),
            color="skyblue",
            edgecolor="black"
        )
    else:
        # Single prediction → simple bar
        ax.bar(
            ["Prediction"],
            prices,
            color="skyblue"
        )

    ax.set_title("Price Distribution")
    ax.set_xlabel("Predicted Price (₹)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction History",
        csv,
        "prediction_history.csv",
        "text/csv"
    )

else:
    st.info("No predictions yet.")