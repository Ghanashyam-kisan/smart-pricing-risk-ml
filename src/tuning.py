from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def tune_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1          # 🔥 IMPORTANT
    )

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=1          # 🔥 IMPORTANT (no parallel processes)
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
