import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def time_based_split(df, split_ratio=0.8, features=None):
    """
    Performs chronological/time-based train-test split (80:20 split by default).
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing features and 'Target'.
    - split_ratio (float): Ratio of training data.
    - features (list): List of feature column names. If None, uses original notebook list.
    
    Returns:
    - X_train, X_test, y_train, y_test: Split feature and target sets.
    """
    if features is None:
        features = ['Close_Lag1', 'Close_Lag2', 'MA7', 'MA21']
        
    train_size = int(len(df) * split_ratio)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    X_train = train[features]
    y_train = train['Target']
    X_test = test[features]
    y_test = test['Target']
    
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Trains a default Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Trains and tunes a Decision Tree Regressor via GridSearchCV."""
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    tree = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(tree, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search

def train_random_forest(X_train, y_train):
    """Trains and tunes a Random Forest Regressor via GridSearchCV."""
    param_rf = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_rf, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search

def save_model(model, filepath):
    """Saves a model using joblib. Creates directories if necessary."""
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filepath):
    """Loads a model from a joblib file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at {filepath}")
    return joblib.load(filepath)

def train_and_save_all(X_train, y_train, model_dir="models/saved_models"):
    """
    Trains all three models (LR, DT, RF) and saves them to the specified directory.
    
    Returns:
        dict: A dictionary of trained models.
    """
    print("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    save_model(lr_model, os.path.join(model_dir, "linear_regression.joblib"))
    
    print("Training Decision Tree Regressor model (hyperparameter tuning)...")
    dt_model = train_decision_tree(X_train, y_train)
    save_model(dt_model, os.path.join(model_dir, "decision_tree.joblib"))
    
    print("Training Random Forest Regressor model (hyperparameter tuning)...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, os.path.join(model_dir, "random_forest.joblib"))
    
    return {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }
