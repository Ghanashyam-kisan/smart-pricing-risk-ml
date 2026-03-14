import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_preprocess
from src.clustering import perform_clustering
from src.models import train_random_forest, train_xgboost
from src.evaluation import evaluate, get_feature_importance
from src.tuning import tune_xgboost
from src.explainability import explain_model


def main():

    print("\n==============================")
    print("SMART PRICING ML PIPELINE")
    print("==============================\n")

    # 1️⃣ Load and preprocess data
    print("Loading and preprocessing data...")

    X, X_scaled, y = load_and_preprocess("data/pricing_data.csv")

    print("Dataset loaded successfully.")
    print("Feature count:", X.shape[1])
    print("Samples:", X.shape[0])


    # 2️⃣ Risk segmentation using clustering
    print("\nPerforming risk segmentation (KMeans)...")

    kmeans_model, risk_clusters = perform_clustering(X_scaled)


    # 3️⃣ Combine scaled features with risk cluster
    X_final = np.column_stack((X_scaled, risk_clusters))


    # 4️⃣ Train-test split (WITH clustering)
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42
    )


    # 5️⃣ Train baseline models (WITH clustering)
    print("\nTraining models WITH risk clustering...")

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)


    # 6️⃣ Evaluate baseline models
    print("\n===== Models WITH Risk Clustering =====")

    rf_metrics = evaluate(rf_model, X_test, y_test)
    xgb_metrics = evaluate(xgb_model, X_test, y_test)

    print("\nRandom Forest Performance:")
    print(rf_metrics)

    print("\nXGBoost Performance:")
    print(xgb_metrics)


    # 7️⃣ Hyperparameter tuning for XGBoost
    print("\nRunning hyperparameter tuning for XGBoost...")

    tuned_xgb, best_params = tune_xgboost(X_train, y_train)

    tuned_metrics = evaluate(tuned_xgb, X_test, y_test)

    print("\n===== Tuned XGBoost Performance =====")
    print(tuned_metrics)

    print("\nBest Params:")
    print(best_params)


    # 8️⃣ Feature importance (Explainability)
    print("\nGenerating feature importance...")

    feature_names = list(X.columns) + ["risk_cluster"]

    rf_importance = get_feature_importance(rf_model, feature_names)
    xgb_importance = get_feature_importance(xgb_model, feature_names)

    print("\nRandom Forest Feature Importance:")
    print(rf_importance.head())

    print("\nXGBoost Feature Importance:")
    print(xgb_importance.head())

    # SHAP Explainability
    print("\nRunning SHAP explainability...")

    sample_data = X_test[:200]

    explain_model(tuned_xgb, sample_data)


    # 9️⃣ Baseline models (WITHOUT risk clustering)
    print("\nTraining baseline models WITHOUT clustering...")

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    rf_base = train_random_forest(X_train_base, y_train_base)
    xgb_base = train_xgboost(X_train_base, y_train_base)

    print("\n===== Models WITHOUT Risk Clustering =====")

    print("\nBaseline Random Forest:")
    print(evaluate(rf_base, X_test_base, y_test_base))

    print("\nBaseline XGBoost:")
    print(evaluate(xgb_base, X_test_base, y_test_base))


    # 🔟 Save best model (for deployment later)
    print("\nSaving best model...")

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(tuned_xgb, "artifacts/best_model.pkl")
    joblib.dump(kmeans_model, "artifacts/clustering_model.pkl")

    print("Models saved in artifacts/")


    print("\nPipeline execution completed successfully!")


# Required for Windows multiprocessing
if __name__ == "__main__":
    main()