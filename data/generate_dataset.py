import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 5000

# ---------------------------
# Generate customer features
# ---------------------------

age = np.random.randint(18, 70, n_samples)

income = np.random.normal(700000, 300000, n_samples)
income = np.clip(income, 150000, 2500000)

credit_score = np.random.normal(700, 90, n_samples)
credit_score = np.clip(credit_score, 300, 850)

claim_history = np.random.poisson(1.5, n_samples)
claim_history = np.clip(claim_history, 0, 7)

vehicle_value = np.random.normal(900000, 400000, n_samples)
vehicle_value = np.clip(vehicle_value, 200000, 3500000)

# ---------------------------
# Risk score calculation
# ---------------------------

young_driver_risk = (age < 25).astype(int)
low_income_risk = (income < 400000).astype(int)

risk_score = (
    0.5 * claim_history +
    0.3 * (750 - credit_score) / 450 +
    0.15 * young_driver_risk +
    0.05 * low_income_risk
)

# ---------------------------
# Base insurance price
# ---------------------------

base_price = (
    0.035 * vehicle_value +
    2500 * claim_history +
    0.003 * (800 - credit_score) * vehicle_value / 1000
)

# ---------------------------
# Final price with risk
# ---------------------------

price = (
    base_price +
    base_price * 0.25 * risk_score +
    np.random.normal(0, 8000, n_samples)
)

price = np.clip(price, 10000, None)

# ---------------------------
# Create dataframe
# ---------------------------

df = pd.DataFrame({
    "age": age,
    "income": income.astype(int),
    "credit_score": credit_score.astype(int),
    "claim_history": claim_history,
    "vehicle_value": vehicle_value.astype(int),
    "price": price.astype(int)
})

# ---------------------------
# Save dataset
# ---------------------------

df.to_csv("data/pricing_data.csv", index=False)

print("Dataset generated successfully!")
print("Shape:", df.shape)
print(df.head())