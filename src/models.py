from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf




def train_xgboost(X_train, y_train):
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb
