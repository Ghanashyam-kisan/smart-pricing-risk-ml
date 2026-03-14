import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_and_preprocess(path):

    df = pd.read_csv(path)

    # Remove missing values
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for future predictions
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")

    return X, X_scaled, y
