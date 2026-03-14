from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# -----------------------------
# Train Random Forest
# -----------------------------
def train_random_forest(X_train, y_train):

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    return rf


# -----------------------------
# Train XGBoost
# -----------------------------
def train_xgboost(X_train, y_train):

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    xgb.fit(X_train, y_train)

    return xgb


# -----------------------------
# Train Linear Regression
# -----------------------------
def train_linear_regression(X_train, y_train):

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    return lr


# -----------------------------
# Train ALL models (optional comparison)
# -----------------------------
def train_models(X_train, y_train):

    models = {}

    models["LinearRegression"] = train_linear_regression(X_train, y_train)
    models["RandomForest"] = train_random_forest(X_train, y_train)
    models["XGBoost"] = train_xgboost(X_train, y_train)

    return models