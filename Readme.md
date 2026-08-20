# Time-Based Stock Price Prediction for TSLA

## 📌 Project Overview
This project is an end-to-end Machine Learning web application designed to analyze historical market data and predict Tesla, Inc. (TSLA) stock prices. It transitions an experimental Jupyter Notebook workflow into a modular, production-style python codebase and builds an interactive analytical dashboard using Streamlit.

The implementation strictly preserves the core ML methodology, hyperparameters, split ratio, and feature pipelines of the original academic notebook while introducing interactive elements, visualizations, and robust data caching.

---

## 👨‍💻 Student Information
* **Student Name:** Mahesh Prakash Kumawat
* **Enrollment No.:** 230020107034
* **Guide:** Prof. Aayushee Bhatt
* **Department:** Computer Engineering
* **Institute:** Ahmedabad Institute of Technology

---

## 🎯 Objective
To build, evaluate, and compare three machine learning regression models (Linear Regression, Decision Tree, and Random Forest) using historical stock prices. The project validates model performance using chronological (time-based) splits rather than randomized splits to prevent data leakage and simulate real trading prediction scenarios.

---

## 🚀 Features
1. **Interactive Dashboard:** Dynamic display of key metrics (latest price, average price, range, volume) and historical stock price/volume charts.
2. **Data Explorer:** Visualized data table structure, statistical descriptions, data types, and filtering tools.
3. **Exploratory Data Analysis (EDA):** Interactive closing trends, moving averages, and correlation heatmaps.
4. **Feature Analysis:** Explanation and correlation of engineered lag and technical indicator features.
5. **Model Evaluation & Comparison:** Metrics comparison matrix (MAE, MSE, RMSE, R² Score) and actual vs. predicted charts.
6. **Stock Price Prediction:** Interactive inputs for generating close price estimations based on custom lag and moving average values or by selecting specific historical dates.
7. **Robust Caching:** Dynamic data loader that uses locally cached `TSLA.csv` data when offline or downloads fresh data from Yahoo Finance.

---

## 📊 Machine Learning Methodology

### 1. Dataset
* **Source:** Yahoo Finance (programmatic retrieval via `yfinance` library).
* **Date Range:** `2015-01-01` to `2024-12-31`.
* **Primary Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### 2. Feature Engineering
To capture temporal relationships, the following features are engineered:
* **Lag Features:** 
  * `Close_Lag1`: The stock's closing price 1 trading day prior.
  * `Close_Lag2`: The stock's closing price 2 trading days prior.
* **Technical Indicators:**
  * `MA7`: 7-day rolling average of closing prices.
  * `MA21`: 21-day rolling average of closing prices.

Rows containing null values due to lag/rolling window initializations are dropped.

### 3. Splitting Strategy
* **Split Type:** Chronological (Time-Based) Split.
* **Ratio:** 80% Training Data, 20% Testing Data.
* **Features:** `['Close_Lag1', 'Close_Lag2', 'MA7', 'MA21']`
* **Target:** `Target` (equivalent to current day `Close` price).

### 4. Regression Models & Hyperparameters
1. **Linear Regression:** Default settings from `scikit-learn`.
2. **Decision Tree Regressor:** Hyperparameter tuned via 3-fold Cross-Validation (`GridSearchCV`).
   * Tuned parameters: `max_depth` ([3, 5, 10, None]) and `min_samples_split` ([2, 5, 10]).
   * `random_state` set to `42` for reproducibility.
3. **Random Forest Regressor:** Hyperparameter tuned via 3-fold Cross-Validation (`GridSearchCV`).
   * Tuned parameters: `n_estimators` ([100, 200]), `max_depth` ([5, 10, None]), and `min_samples_split` ([2, 5]).
   * `random_state` set to `42` for reproducibility.

### 5. Evaluation Metrics
Models are compared based on:
* **MAE:** Mean Absolute Error (measures absolute prediction deviations).
* **MSE:** Mean Squared Error (penalizes larger errors).
* **RMSE:** Root Mean Squared Error (interpretable in stock price USD).
* **R² Score:** Coefficient of determination (measures variance explained by the model).

---

## 📁 Project Structure
```text
TSLA-Stock-Prediction/
│
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Academic README and metadata
├── .gitignore                  # Git ignore rules for cached models/data
│
├── data/
│   └── TSLA.csv                # Local cached dataset downloaded from yfinance
│
├── notebooks/
│   └── original_notebook.ipynb # The unmodified original Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data retrieval from yfinance/CSV
│   ├── preprocessing.py        # Flat index normalization
│   ├── feature_engineering.py  # Lag features & technical indicators
│   ├── models.py               # Houses LR, DT, RF definitions & parameter grids
│   ├── evaluation.py           # Computes MAE, MSE, RMSE, R² & Feature Importance
│   └── visualization.py        # Generates interactive Plotly plots
│
├── models/
│   └── saved_models/           # Saved serialized joblib model files
│
└── assets/
    └── project_assets/         # Saved visual assets or screenshots
```

---

## 🛠️ Installation & Setup

1. **Clone the project directory** or navigate into the workspace.
2. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment**:
   * Windows:
     ```bash
     .venv\Scripts\activate
     ```
   * Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

1. **Pre-train the models** (or run the verification script) to compile features and serialize model files:
   ```bash
   python -c "from src.data_loader import load_data; from src.preprocessing import clean_data; from src.feature_engineering import generate_features; from src.models import time_based_split, train_and_save_all; df = generate_features(clean_data(load_data())); X_tr, X_te, y_tr, y_te = time_based_split(df); train_and_save_all(X_tr, y_tr)"
   ```
2. **Launch the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the local link shown in the terminal (typically `http://localhost:8501`).

---

## 📸 Screenshots
*(Screenshots of the finished interface can be placed here in the assets/ folder)*

---

## ⚠️ Limitations & Future Enhancements

### Limitations
1. **Lag Dependency:** The model is an autoregressive-style regressor. It requires historical lag values (`Close_Lag1`, `Close_Lag2`) to predict the current day's price, which means it cannot predict arbitrary future dates multiple weeks in advance without sequential iterative feeding (which accumulates error).
2. **Non-stationarity:** Financial stock markets are highly non-stationary. Basic regression models do not account for external market events, macroeconomic adjustments, or sentiment shifts (such as news reports or Elon Musk's tweets).

### Future Enhancements
1. **Dynamic Rolling Refit:** Re-training the models daily as new stock ticks arrive to capture recent trends.
2. **Deep Learning Integration:** Implementing recurrent architectures such as LSTM (Long Short-Term Memory) or GRU.
3. **Sentiment Analysis:** Fetching news headlines or social media sentiments for TSLA and incorporating sentiment scores as predictive features.