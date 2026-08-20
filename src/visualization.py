import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Minimal Light Theme Colors
PRIMARY_BLACK = "#111827"
SECONDARY_GRAY = "#4B5563"
LIGHT_GRAY = "#9CA3AF"
BORDER_GRAY = "#E5E7EB"
TESLA_RED = "#DC2626"
BLUE_ACCENT = "#2563EB"
LIGHT_NAVY = PRIMARY_BLACK


def get_minimal_layout(title, xaxis_title, yaxis_title):
    """Utility to return a minimal light layout for plotly figures."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="Space Grotesk, sans-serif", size=16, color=PRIMARY_BLACK)
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=SECONDARY_GRAY, size=12)),
            tickfont=dict(color=SECONDARY_GRAY, size=11),
            gridcolor=BORDER_GRAY,
            linecolor=PRIMARY_BLACK,
            zerolinecolor=BORDER_GRAY
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=SECONDARY_GRAY, size=12)),
            tickfont=dict(color=SECONDARY_GRAY, size=11),
            gridcolor=BORDER_GRAY,
            linecolor=PRIMARY_BLACK,
            zerolinecolor=BORDER_GRAY
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

def plot_historical_prices(df):
    """Plots historical close price of TSLA (Minimal Light)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color=PRIMARY_BLACK, width=2)
    ))
    fig.update_layout(**get_minimal_layout("TSLA Historical Close Price", "Date", "Stock Price (USD)"))
    return fig

def plot_moving_averages(df):
    """Plots close price along with moving averages MA7 and MA21 (Minimal Light)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color=PRIMARY_BLACK, width=1.5, dash='solid')
    ))
    if 'MA7' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA7'],
            mode='lines',
            name='MA 7-Day',
            line=dict(color=TESLA_RED, width=1.5, dash='dash')
        ))
    if 'MA21' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA21'],
            mode='lines',
            name='MA 21-Day',
            line=dict(color=BLUE_ACCENT, width=1.5, dash='dot')
        ))
    fig.update_layout(**get_minimal_layout("TSLA Price & Technical Indicators", "Date", "Stock Price (USD)"))
    return fig

def plot_volume(df):
    """Plots trading volumes over time (Minimal Light)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume',
        marker=dict(color=LIGHT_GRAY)
    ))
    layout = get_minimal_layout("TSLA Historical Trading Volume", "Date", "Volume")
    # Columns do not need a line unified hover
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig

def plot_correlation_heatmap(df, features):
    """Plots correlation heatmap of numerical features (Minimal Light)."""
    corr_matrix = df[features].corr()
    
    fig = px.imshow(
        corr_matrix,
        x=features,
        y=features,
        color_continuous_scale="Greys",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap"
    )
    
    layout = get_minimal_layout("Feature Correlation Heatmap", "", "")
    layout.pop("xaxis")
    layout.pop("yaxis")
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(tickfont=dict(color=SECONDARY_GRAY)),
        yaxis=dict(tickfont=dict(color=SECONDARY_GRAY))
    )
    return fig

def plot_model_comparison(df_comparison, metric="RMSE"):
    """Plots a bar chart comparing performance metrics across models (Minimal Light)."""
    df_plot = df_comparison.copy()
    
    fig = go.Figure()
    # Grayscale bar coloring
    colors = [PRIMARY_BLACK, SECONDARY_GRAY, LIGHT_GRAY]
    
    fig.add_trace(go.Bar(
        x=df_plot["Model"],
        y=df_plot[metric],
        marker_color=colors[:len(df_plot)],
        text=df_plot[metric],
        textposition='auto',
    ))
    
    better_text = "Lower is better" if metric in ["MAE", "MSE", "RMSE"] else "Higher is better"
    layout = get_minimal_layout(f"Model Comparison: {metric} ({better_text})", "Model", metric)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig

def plot_actual_vs_predicted(dates, y_true, y_pred, model_name):
    """Plots actual values versus predicted values (Minimal Light)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y_true.values if hasattr(y_true, "values") else y_true,
        mode='lines',
        name='Actual Close Price',
        line=dict(color=PRIMARY_BLACK, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred,
        mode='lines',
        name=f'Predicted ({model_name})',
        line=dict(color=TESLA_RED, width=1.5, dash='dash')
    ))
    fig.update_layout(**get_minimal_layout(f"Actual vs. Predicted Price ({model_name})", "Date", "Price (USD)"))
    return fig

def plot_feature_importances(feature_importances):
    """Plots a horizontal bar chart of feature importances (Minimal Light)."""
    df_fi = pd.DataFrame({
        'Feature': feature_importances.index,
        'Importance': feature_importances.values
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_fi['Feature'],
        x=df_fi['Importance'],
        orientation='h',
        marker=dict(color=PRIMARY_BLACK)
    ))
    
    layout = get_minimal_layout("Random Forest Feature Importances", "Importance", "Feature")
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig
