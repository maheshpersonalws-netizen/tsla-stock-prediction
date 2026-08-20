import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import generate_features
from src.models import time_based_split, train_and_save_all, load_model
from src.evaluation import evaluate_predictions, compile_model_comparison, get_best_model_name, get_feature_importances
from src.visualization import (
    plot_historical_prices, plot_moving_averages, plot_volume, 
    plot_correlation_heatmap, plot_model_comparison, plot_actual_vs_predicted,
    plot_feature_importances, TESLA_RED, LIGHT_NAVY
)

# Page Configuration
st.set_page_config(
    page_title="TSLA Stock Predictions Pvt. Ltd.",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected Minimal White & Black CSS Style Rules (Targeted layout selectors)
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* Target main view container and sidebar explicitly */
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        background-image: none !important;
        color: #111827 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    /* Keep the header container visible so that the sidebar toggle button works */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: none !important;
        visibility: visible !important;
    }
    
    /* Hide MainMenu and footer elements */
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    
    /* Minimal Card System (Clean Black and White Borders) */
    .premium-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        margin-bottom: 20px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* Interactive Card hovering */
    .premium-card:hover {
        transform: translateY(-2px) !important;
        border-color: #111827 !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06) !important;
    }

    /* Inner card typography */
    .metric-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        color: #4B5563 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #111827 !important;
        margin-top: 8px;
        letter-spacing: -0.02em;
    }
    
    .metric-value.red-glow {
        color: #DC2626 !important;
    }

    .metric-value.green-glow {
        color: #059669 !important;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #6B7280 !important;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Header layout typography */
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #111827 !important;
        letter-spacing: -0.03em;
        margin-bottom: 0px;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #4B5563 !important;
        margin-top: 5px;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Tabs custom override styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 16px !important;
        color: #4B5563 !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #111827 !important;
        color: #111827 !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
DATA_PATH = "data/TSLA.csv"
MODEL_DIR = "models/saved_models"

# Caching Data Loading
@st.cache_data
def get_processed_data():
    raw_df = load_data(csv_path=DATA_PATH)
    clean_df = clean_data(raw_df)
    features_df = generate_features(clean_df)
    return clean_df, features_df

# Helper function to check and load models
def load_trained_models(X_train, y_train):
    models = {}
    model_files = {
        "Linear Regression": "linear_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib"
    }
    
    all_exists = True
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            all_exists = False
            break
            
    if all_exists:
        try:
            for name, filename in model_files.items():
                models[name] = load_model(os.path.join(MODEL_DIR, filename))
            return models
        except Exception as e:
            st.error(f"Error loading saved models: {e}")
            return None
    return None

st.sidebar.markdown(f"<h3 style='text-align: center; color: {TESLA_RED}; font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.25rem; margin-bottom: 0px;'>TSLA Quantum Capital Analytics</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 0px;'>Quantitative Intelligence Portal</p>", unsafe_allow_html=True)

branding_name = "TSLA Quantum Capital Analytics"

# Navigation
st.sidebar.markdown("### 📋 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "📊 Dashboard",
        "🔍 Data Explorer",
        "📈 Exploratory Analysis",
        "⚙️ Feature Engineering",
        "🎯 Model Performance",
        "🔮 Price Prediction",
        "🎓 About Project"
    ]
)

# Load data with handling for missing values
try:
    clean_df, features_df = get_processed_data()
    X_train, X_test, y_train, y_test = time_based_split(features_df)
except Exception as e:
    st.error(f"Error initializing data: {e}")
    st.stop()

# Check and Load Models
models_dict = load_trained_models(X_train, y_train)

# Dynamic Header
st.markdown(f"<h1 class='main-title'>{branding_name}</h1>", unsafe_allow_html=True)

# PAGE 1: DASHBOARD
if page == "📊 Dashboard":
    st.markdown("<p class='sub-title'>Next-Gen Predictive Analytics & Quantitative Forecasting System</p>", unsafe_allow_html=True)
    
    # Calculate Key Metrics Dynamically
    latest_rec = clean_df.iloc[-1]
    prev_rec = clean_df.iloc[-2] if len(clean_df) > 1 else latest_rec
    
    latest_close = latest_rec['Close']
    prev_close = prev_rec['Close']
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100
    
    total_records = len(clean_df)
    min_price = clean_df['Close'].min()
    max_price = clean_df['Close'].max()
    avg_price = clean_df['Close'].mean()
    
    start_date = clean_df['Date'].min().strftime('%Y-%m-%d')
    end_date = clean_df['Date'].max().strftime('%Y-%m-%d')
    
    # Layout Metrics in Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="premium-card tesla-brand">
            <div class="metric-title">Latest Close Price</div>
            <div class="metric-value {'green-glow' if price_change >= 0 else 'red-glow'}">${latest_close:.2f}</div>
            <div class="metric-sub">
                {'▲' if price_change >= 0 else '▼'} {abs(price_change):.2f} ({pct_change:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">Price Range (Min - Max)</div>
            <div class="metric-value">${min_price:.2f} - ${max_price:.2f}</div>
            <div class="metric-sub">Historical Extremes</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">Average Stock Price</div>
            <div class="metric-value">${avg_price:.2f}</div>
            <div class="metric-sub">Mean Close Value</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-sub">Trading Days ({start_date} to {end_date})</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Visualizations
    st.subheader("📈 TSLA Historical Price Chart")
    fig_price = plot_historical_prices(clean_df)
    st.plotly_chart(fig_price, use_container_width=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📊 Trading Volume Trends")
        fig_vol = plot_volume(clean_df)
        st.plotly_chart(fig_vol, use_container_width=True)
    with col_right:
        st.subheader("🛠️ Technical Moving Averages")
        fig_ma = plot_moving_averages(clean_df)
        st.plotly_chart(fig_ma, use_container_width=True)

# PAGE 2: DATA EXPLORER
elif page == "🔍 Data Explorer":
    st.markdown("<p class='sub-title'>Interactive Data Audit & Metric Summaries</p>", unsafe_allow_html=True)
    
    # Layout tabs
    tab_view, tab_summary = st.tabs(["Raw Data Preview", "Data Structure Summary"])
    
    with tab_view:
        st.subheader("TSLA Dataset Filter View")
        
        # Filters in columns
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            min_date = clean_df['Date'].min().to_pydatetime()
            max_date = clean_df['Date'].max().to_pydatetime()
            date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            
        with f_col2:
            min_vol, max_vol = int(clean_df['Volume'].min()), int(clean_df['Volume'].max())
            vol_range = st.slider("Volume Range", min_value=min_vol, max_value=max_vol, value=(min_vol, max_vol))
            
        # Apply filters
        filtered_df = clean_df.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['Date'] >= pd.to_datetime(date_range[0])) & 
                (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
            ]
        filtered_df = filtered_df[
            (filtered_df['Volume'] >= vol_range[0]) & 
            (filtered_df['Volume'] <= vol_range[1])
        ]
        
        st.write(f"Showing **{len(filtered_df)}** records matching filter criteria.")
        st.dataframe(filtered_df.style.format({
            "Open": "{:.2f}",
            "High": "{:.2f}",
            "Low": "{:.2f}",
            "Close": "{:.2f}",
            "Volume": "{:,}"
        }), use_container_width=True)
        
    with tab_summary:
        st.subheader("Dataset Info")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Shape:**")
            st.write(clean_df.shape)
            
            st.markdown("**Column Data Types:**")
            dtypes_df = pd.DataFrame(clean_df.dtypes, columns=['Data Type']).astype(str)
            st.dataframe(dtypes_df, use_container_width=True)
            
        with col_s2:
            st.markdown("**Missing Value Summary:**")
            null_df = pd.DataFrame(clean_df.isnull().sum(), columns=['Null Count'])
            st.dataframe(null_df, use_container_width=True)
            
        st.subheader("Statistical Summary")
        st.dataframe(clean_df.describe().style.format({
            "Open": "{:.2f}",
            "High": "{:.2f}",
            "Low": "{:.2f}",
            "Close": "{:.2f}",
            "Volume": "{:,.0f}"
        }), use_container_width=True)

# PAGE 3: EXPLORATORY ANALYSIS (EDA)
elif page == "📈 Exploratory Analysis":
    st.markdown("<p class='sub-title'>Market Dynamics & Trend Visualization</p>", unsafe_allow_html=True)
    
    st.subheader("📉 Price Trend Overlay")
    # Custom interactive plot with multiple trace toggles
    fig_overlay = go.Figure()
    fig_overlay.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['High'], name="High", line=dict(color='#10B981', width=1)))
    fig_overlay.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['Low'], name="Low", line=dict(color='#EF4444', width=1)))
    fig_overlay.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['Close'], name="Close", line=dict(color=LIGHT_NAVY, width=2)))
    fig_overlay.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_overlay, use_container_width=True)
    
    st.subheader("Distribution Analysis")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        # Close price distribution
        fig_dist = px.histogram(
            clean_df, x="Close", nbins=50, 
            title="TSLA Close Price Distribution",
            color_discrete_sequence=[LIGHT_NAVY]
        )
        fig_dist.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_d2:
        # Volume distribution
        fig_vdist = px.histogram(
            clean_df, x="Volume", nbins=50, 
            title="Trading Volume Distribution",
            color_discrete_sequence=['#64748B']
        )
        fig_vdist.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_vdist, use_container_width=True)

# PAGE 4: FEATURE ENGINEERING
elif page == "⚙️ Feature Engineering":
    st.markdown("<p class='sub-title'>Predictive Inputs & Technical Indicators</p>", unsafe_allow_html=True)
    
    # Feature Description list
    st.markdown("""
    ### 🔬 Features Implemented:
    * **Close_Lag1**: The closing price shifted by 1 day (yesterday's close price). Captures immediate momentum.
    * **Close_Lag2**: The closing price shifted by 2 days.
    * **MA7**: 7-day simple moving average of close prices. Represents short-term trend line.
    * **MA21**: 21-day simple moving average of close prices. Represents medium-term trend line.
    """)
    
    # Feature correlation heatmap
    features_list = ['Close', 'Close_Lag1', 'Close_Lag2', 'MA7', 'MA21']
    st.subheader("🔥 Feature Correlation Matrix")
    st.markdown("Correlation values between engineered inputs and the target close price. Values close to 1 indicates extremely high positive correlation.")
    fig_corr = plot_correlation_heatmap(features_df, features_list)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Features preview
    st.subheader("📋 Engineered Dataset Preview")
    show_cols = ['Date', 'Close', 'Close_Lag1', 'Close_Lag2', 'MA7', 'MA21', 'Target']
    st.dataframe(features_df[show_cols].head(15).style.format({
        "Close": "{:.2f}",
        "Close_Lag1": "{:.2f}",
        "Close_Lag2": "{:.2f}",
        "MA7": "{:.2f}",
        "MA21": "{:.2f}",
        "Target": "{:.2f}"
    }), use_container_width=True)

# PAGE 5: MODEL PERFORMANCE
elif page == "🎯 Model Performance":
    st.markdown("<p class='sub-title'>Quantitative Benchmarks & Evaluation Reports</p>", unsafe_allow_html=True)
    
    # Check if models are trained. If not, trigger training
    if models_dict is None:
        st.warning("⚠️ Serialized models are not yet trained and saved. Please trigger training below.")
        if st.button("🚀 Train & Save Models"):
            with st.spinner("Training models and saving files (this might take a few seconds)..."):
                try:
                    models_dict = train_and_save_all(X_train, y_train, model_dir=MODEL_DIR)
                    st.success("All models trained and saved to models/saved_models/!")
                    # Force page reload to load models correctly
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training: {e}")
        st.stop()
        
    st.markdown("Model evaluation metrics calculated dynamically on the test partition (20% chronological split).")
    
    # Generate Predictions for the test set
    predictions = {}
    for name, model in models_dict.items():
        predictions[name] = model.predict(X_test)
        
    # Compile comparison DataFrame
    df_comparison = compile_model_comparison(y_test, predictions)
    
    # Highlight Best Model
    best_name, best_val = get_best_model_name(df_comparison, criterion="RMSE")
    st.markdown(f"""
    <div class="premium-card success-brand" style="margin-top: 15px;">
        <div class="metric-title">🏆 Top Performing Predictor</div>
        <div class="metric-value green-glow" style="font-size: 1.8rem;">{best_name}</div>
        <div class="metric-sub">RMSE: {best_val:.4f} (Lower RMSE denotes superior fitting)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics display
    st.subheader("📊 Metric Comparison Table")
    st.dataframe(df_comparison.style.format({
        "MAE": "{:.2f}",
        "MSE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R2": "{:.4f}"
    }), use_container_width=True)
    
    # Toggle performance charts
    metric_choice = st.selectbox("Select Evaluation Metric for Comparison Chart:", ["RMSE", "MAE", "MSE", "R2"])
    fig_comp = plot_model_comparison(df_comparison, metric=metric_choice)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Actual vs Predicted Plots
    st.subheader("📈 Actual vs. Predicted Visualizations")
    model_choice = st.selectbox("Select Model to Visualize Actual vs. Predicted:", list(models_dict.keys()))
    
    test_dates = features_df['Date'].iloc[len(X_train):]
    fig_avp = plot_actual_vs_predicted(test_dates, y_test, predictions[model_choice], model_choice)
    st.plotly_chart(fig_avp, use_container_width=True)
    
    # Feature Importance (only for Random Forest)
    st.subheader("💡 Feature Importance Analysis")
    st.markdown("Contribution of engineered features to the Random Forest model's predictions.")
    try:
        sorted_fi = get_feature_importances(models_dict["Random Forest"], X_train.columns)
        fig_fi = plot_feature_importances(sorted_fi)
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load feature importances: {e}")

# PAGE 6: PRICE PREDICTION
elif page == "🔮 Price Prediction":
    st.markdown("<p class='sub-title'>Real-Time Predictive Interface & Inference Engine</p>", unsafe_allow_html=True)
    
    if models_dict is None:
        st.error("⚠️ Saved models not found. Please train models on the 'Model Performance' page first.")
        st.stop()
        
    st.markdown("Use trained models to estimate the closing stock price. Select a model and customize features or load a specific historical date.")
    
    # Interactive selection controls
    pred_mode = st.radio("Input Method:", ["📅 Load Price Data from Historical Date", "🎛️ Manually Enter Feature Sliders"])
    
    # Default parameters based on selection
    if pred_mode == "📅 Load Price Data from Historical Date":
        # Let user choose a date from the dataset
        st.subheader("Select Historical Date")
        dates_list = features_df['Date'].dt.date.tolist()
        # default to latest date
        selected_date = st.selectbox("Select Trading Date:", dates_list, index=len(dates_list)-1)
        
        # Extract features for selected date
        row = features_df[features_df['Date'].dt.date == selected_date].iloc[0]
        lag1_val = float(row['Close_Lag1'])
        lag2_val = float(row['Close_Lag2'])
        ma7_val = float(row['MA7'])
        ma21_val = float(row['MA21'])
        actual_val = float(row['Target'])
        
        st.info(f"Loaded values for **{selected_date}**:  \n"
                f"**Close_Lag1 (Yesterday Close):** ${lag1_val:.2f} | "
                f"**Close_Lag2 (2 Days Ago Close):** ${lag2_val:.2f}  \n"
                f"**MA7 (Short-term Trend):** ${ma7_val:.2f} | "
                f"**MA21 (Medium-term Trend):** ${ma21_val:.2f}  \n"
                f"**Actual Closing Price on this day:** **${actual_val:.2f}**")
    else:
        st.subheader("Configure Input Feature Sliders")
        # Define ranges based on actual data bounds
        min_p = float(features_df['Close'].min())
        max_p = float(features_df['Close'].max())
        mid_p = float(features_df['Close'].median())
        
        lag1_val = st.slider("Close_Lag1 (Yesterday Close Price)", min_value=min_p*0.5, max_value=max_p*1.2, value=mid_p)
        lag2_val = st.slider("Close_Lag2 (2 Days Ago Close Price)", min_value=min_p*0.5, max_value=max_p*1.2, value=mid_p)
        ma7_val = st.slider("MA7 (7-day Average price)", min_value=min_p*0.5, max_value=max_p*1.2, value=mid_p)
        ma21_val = st.slider("MA21 (21-day Average price)", min_value=min_p*0.5, max_value=max_p*1.2, value=mid_p)
        actual_val = None
        
    model_choice = st.selectbox("Select Model for Prediction:", list(models_dict.keys()))
    
    if st.button("Generate Prediction", type="primary"):
        input_data = pd.DataFrame([{
            'Close_Lag1': lag1_val,
            'Close_Lag2': lag2_val,
            'MA7': ma7_val,
            'MA21': ma21_val
        }])
        
        # Predict price
        model = models_dict[model_choice]
        predicted_price = float(model.predict(input_data)[0])
        
        # Display output card
        st.markdown(f"""
        <div class="premium-card success-brand" style="margin-top: 20px;">
            <div class="metric-title">Predicted Stock Price ({model_choice})</div>
            <div class="metric-value green-glow">${predicted_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display difference if we have actual value
        if actual_val is not None:
            error = predicted_price - actual_val
            pct_err = (error / actual_val) * 100
            err_color = "#10B981" if abs(pct_err) < 2 else "#F59E0B" if abs(pct_err) < 5 else "#EF4444"
            st.markdown(f"**Actual Closing Price:** ${actual_val:.2f}")
            st.markdown(f"**Prediction Error:** <span style='color:{err_color}; font-weight:bold;'>${error:.2f} ({pct_err:.2f}%)</span>", unsafe_allow_html=True)
            
            # Simple bar plot actual vs predicted comparison
            fig_bar = go.Figure(data=[
                go.Bar(name='Actual Price', x=['TSLA Close Price'], y=[actual_val], marker_color=LIGHT_NAVY),
                go.Bar(name='Predicted Price', x=['TSLA Close Price'], y=[predicted_price], marker_color=TESLA_RED)
            ])
            fig_bar.update_layout(
                yaxis_title="USD",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                barmode='group',
                width=400, height=350
            )
            st.plotly_chart(fig_bar, use_container_width=False)

# PAGE 7: ABOUT PROJECT
elif page == "🎓 About Project":
    st.markdown("<p class='sub-title'>Academic Internship Framework & Student Info</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Project Overview
    This web application was developed as a productization extension of a **Jupyter Notebook-based Semester 7 Summer Internship project** focusing on Time-Based Stock Price Prediction.
    
    The underlying algorithms compare Linear Regression with cross-validated Decision Tree and Random Forest Regressors to predict the closing price of Tesla (TSLA) stock.
    
    ### 🎓 Internship Information
    * **Student Name:** Mahesh Prakash Kumawat
    * **Enrollment No.:** 230020107034
    * **Guide:** Prof. Aayushee Bhatt
    * **Department:** Computer Engineering
    * **Institute:** Ahmedabad Institute of Technology
    
    ---
    
    ### 🛠️ ML Pipeline & Technical Implementation
    * **Chronological Split:** Stock data contains chronological auto-correlations. Splitting features chronologically (80% training set on older historical dates, 20% test set on newer dates) prevents 'data leakage' where future prices inadvertently predict past prices.
    * **Auto-Regressive Indicators:** Features are constructed purely from historical lag records:
      $$\\text{Predicted Price}_t = f(\\text{Close}_{t-1}, \\text{Close}_{t-2}, \\text{Moving Average 7}_t, \\text{Moving Average 21}_t)$$
    * **Cross-Validation:** 3-fold cross-validation (`GridSearchCV`) ensures Decision Tree and Random Forest models are optimized for generalizability on unseen data rather than overfitted on training dates.
    
    ### ⚠️ Limitations & Warning
    * **Lag Limitation:** Autoregressive stock models require yesterday's actual stock price to predict today's price. Consequently, this model is **not designed to predict arbitrary dates weeks or months in advance** directly, as any recursive prediction error aggregates exponentially.
    * **Market Volatility:** Financial stock markets are subject to exogenous shocks (earnings calls, regulations, CEO tweets, macroeconomics) which cannot be predicted solely by past technical price history.
    * **No Financial Advice:** This project is strictly for academic/educational demonstration purposes. It does not constitute financial or investment advice.
    """)

# Sidebar Footer
st.sidebar.markdown(f"""
<div class="credit-text">
    Developed by Mahesh Kumawat<br>
    Ahmedabad Institute of Technology<br>
    CE Department • Internship 2026
</div>
""", unsafe_allow_html=True)
