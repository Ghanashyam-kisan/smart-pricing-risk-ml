import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

    return results


def get_feature_importance(model, feature_names):

    importance = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    return df.sort_values(by="importance", ascending=False)