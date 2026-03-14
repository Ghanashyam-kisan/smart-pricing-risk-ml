from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


def tune_xgboost(X_train, y_train):

    param_dist = {

        "n_estimators": [200, 300, 400, 500],
        "max_depth": [4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8]
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=20,
        scoring="r2",
        cv=3,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_