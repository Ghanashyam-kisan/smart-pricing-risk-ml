import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 5000

age = np.random.randint(18, 65, n_samples)
income = np.random.normal(600000, 250000, n_samples).clip(150000, 2000000)
credit_score = np.random.normal(700, 80, n_samples).clip(300, 900)
claim_history = np.random.poisson(1.2, n_samples).clip(0, 6)
vehicle_value = np.random.normal(800000, 300000, n_samples).clip(200000, 3000000)

risk_score = (
    0.4 * claim_history +
    0.3 * (700 - credit_score) / 400 +
    0.2 * (age < 25).astype(int) +
    0.1 * (income < 400000).astype(int)
)

base_price = (
    0.03 * vehicle_value +
    1500 * claim_history +
    0.002 * (800 - credit_score) * vehicle_value / 1000
)

price = (base_price + base_price * 0.15 * risk_score +
         np.random.normal(0, 5000, n_samples)).clip(8000, None)

df = pd.DataFrame({
    "age": age,
    "income": income.astype(int),
    "credit_score": credit_score.astype(int),
    "claim_history": claim_history,
    "vehicle_value": vehicle_value.astype(int),
    "price": price.astype(int)
})

df.to_csv("data/pricing_data.csv", index=False)
print("Dataset generated:", df.shape)

