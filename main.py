import numpy as np
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_preprocess
from src.clustering import perform_clustering
from src.models import train_random_forest, train_xgboost
from src.evaluation import evaluate, get_feature_importance
from src.tuning import tune_xgboost


def main():
    # 1️⃣ Load and preprocess data
    X, X_scaled, y = load_and_preprocess("data/pricing_data.csv")

    # 2️⃣ Risk segmentation using clustering
    risk_clusters = perform_clustering(X_scaled)

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
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # 6️⃣ Evaluate baseline models
    print("\n===== Models WITH Risk Clustering =====")
    print("Random Forest Performance:")
    print(evaluate(rf_model, X_test, y_test))

    print("\nXGBoost Performance:")
    print(evaluate(xgb_model, X_test, y_test))

    # 7️⃣ Hyperparameter tuning for XGBoost
    tuned_xgb, best_params = tune_xgboost(X_train, y_train)

    print("\n===== Tuned XGBoost Performance =====")
    print(evaluate(tuned_xgb, X_test, y_test))
    print("Best Params:", best_params)

    # 8️⃣ Feature importance (Explainability)
    feature_names = list(X.columns) + ["risk_cluster"]

    rf_importance = get_feature_importance(rf_model, feature_names)
    xgb_importance = get_feature_importance(xgb_model, feature_names)

    print("\nRandom Forest Feature Importance:")
    print(rf_importance.head())

    print("\nXGBoost Feature Importance:")
    print(xgb_importance.head())

    # 9️⃣ Baseline models (WITHOUT risk clustering)
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    rf_base = train_random_forest(X_train_base, y_train_base)
    xgb_base = train_xgboost(X_train_base, y_train_base)

    print("\n===== Models WITHOUT Risk Clustering =====")
    print("Baseline Random Forest:")
    print(evaluate(rf_base, X_test_base, y_test_base))

    print("\nBaseline XGBoost:")
    print(evaluate(xgb_base, X_test_base, y_test_base))


# 🔥 REQUIRED for Windows + multiprocessing safety
if __name__ == "__main__":
    main()
