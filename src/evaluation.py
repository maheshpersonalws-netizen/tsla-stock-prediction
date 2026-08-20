import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    """
    Computes regression metrics.
    
    Returns:
    - dict: Dictionary containing MAE, MSE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

def compile_model_comparison(y_true, predictions_dict):
    """
    Compares multiple models by compiling their performance metrics.
    
    Parameters:
    - y_true (pd.Series): Actual target values.
    - predictions_dict (dict): Dictionary mapping model name to y_pred.
    
    Returns:
    - pd.DataFrame: Metric comparison DataFrame.
    """
    comparison = []
    for model_name, y_pred in predictions_dict.items():
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["Model"] = model_name
        comparison.append(metrics)
        
    df_comparison = pd.DataFrame(comparison)
    # Reorder columns so Model is first
    cols = ["Model", "MAE", "MSE", "RMSE", "R2"]
    df_comparison = df_comparison[cols]
    
    return df_comparison

def get_best_model_name(df_comparison, criterion="RMSE"):
    """
    Determines the best model based on the metric comparison.
    By default, chooses the model with the minimum RMSE. If criterion is 'R2', chooses maximum R2.
    """
    if df_comparison.empty:
        return None
        
    if criterion == "R2":
        best_row = df_comparison.loc[df_comparison["R2"].idxmax()]
    else:
        best_row = df_comparison.loc[df_comparison["RMSE"].idxmin()]
        
    return best_row["Model"], best_row[criterion]

def get_feature_importances(fitted_model, feature_names):
    """
    Extracts feature importances from a fitted Random Forest model (or GridSearchCV search object).
    
    Returns:
    - pd.Series: Sorted feature importances.
    """
    # Handle if fitted_model is GridSearchCV
    if hasattr(fitted_model, "best_estimator_"):
        estimator = fitted_model.best_estimator_
    else:
        estimator = fitted_model
        
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Model does not support feature importances.")
        
    importances = estimator.feature_importances_
    sorted_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return sorted_importances
