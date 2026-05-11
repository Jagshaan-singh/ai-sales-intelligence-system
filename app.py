"""
============================================================
  Sales Forecasting — Streamlit Web App
  Run with: streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for a polished look ───────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1976D2;
        margin-bottom: 0;
    }
    .sub-header { color: #666; font-size: 1rem; margin-top: 0; }
    .metric-card {
        background: #F0F7FF;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #BBDEFB;
    }
    .metric-label { font-size: 0.8rem; color: #666; margin-bottom: 4px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #1565C0; }
    .stButton > button {
        background-color: #1976D2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: set plot style ───────────────────────────────────
def set_style():
    plt.rcParams.update({
        'figure.facecolor' : 'white',
        'axes.facecolor'   : '#F8F9FA',
        'axes.edgecolor'   : '#DDDDDD',
        'axes.grid'        : True,
        'grid.color'       : '#EEEEEE',
        'font.family'      : 'DejaVu Sans',
        'axes.titlesize'   : 12,
        'axes.titleweight' : 'bold',
    })


# ── Helper: load & clean data ────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
        # Accept either 'Date,Sales' or first two columns
        df.columns = ['Date', 'Sales']
    else:
        # Built-in demo dataset
        dates = pd.date_range(start='2021-01-01', periods=36, freq='MS')
        sales = [
            1000, 1100, 1050, 1200, 1300, 1250,
            1400, 1500, 1350, 1600, 1700, 1900,
            1800, 1950, 1850, 2000, 2100, 2050,
            2200, 2350, 2300, 2400, 2500, 2700,
            2600, 2700, 2650, 2800, 2900, 2850,
            3000, 3100, 3050, 3200, 3300, 3500
        ]
        df = pd.DataFrame({'Date': dates, 'Sales': sales})

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Sales'] = df['Sales'].interpolate(method='linear')
    return df


# ── Helper: train model ──────────────────────────────────────
@st.cache_resource
def train_prophet(df, changepoint_scale, seasonality_scale):
    prophet_df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_scale,
        seasonality_prior_scale=seasonality_scale,
        interval_width=0.95
    )
    model.fit(prophet_df)
    return model, prophet_df


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Chart_line_icon.svg/200px-Chart_line_icon.svg.png",
             width=60)
    st.markdown("## ⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload CSV (Date, Sales)",
        type=['csv'],
        help="Two columns: Date (YYYY-MM or YYYY-MM-DD) and Sales (numbers)"
    )

    st.markdown("---")
    forecast_days = st.slider("Forecast period (days)", 30, 365, 90, step=30)

    st.markdown("**Model tuning**")
    changepoint_scale = st.slider(
        "Trend flexibility", 0.01, 0.5, 0.05, step=0.01,
        help="Higher = model follows data more closely (risk of overfitting)"
    )
    seasonality_scale = st.slider(
        "Seasonality strength", 1, 20, 10, step=1,
        help="Higher = stronger seasonal patterns captured"
    )

    st.markdown("---")
    show_raw  = st.checkbox("Show raw data table", False)
    show_eval = st.checkbox("Show model evaluation", True)
    show_comp = st.checkbox("Show trend components", True)


# ── HEADER ───────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Sales Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Facebook Prophet · Time Series Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# ── LOAD DATA ────────────────────────────────────────────────
df = load_data(uploaded_file)
if uploaded_file is None:
    st.info("💡 Using built-in demo data. Upload your own CSV via the sidebar.")

# ── METRIC CARDS ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 Data Points", len(df))
with col2:
    st.metric("📅 Date Range",
              f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
with col3:
    st.metric("📊 Avg Monthly Sales", f"{df['Sales'].mean():,.0f}")
with col4:
    growth = ((df['Sales'].iloc[-1] - df['Sales'].iloc[0]) / df['Sales'].iloc[0]) * 100
    st.metric("📈 Total Growth", f"{growth:+.1f}%")

if show_raw:
    st.dataframe(df.style.format({'Sales': '{:,.0f}'}), use_container_width=True)

st.markdown("---")

# ── TRAIN MODEL ─────────────────────────────────────────────
with st.spinner("🤖 Training Prophet model... (this takes a few seconds)"):
    model, prophet_df = train_prophet(df, changepoint_scale, seasonality_scale)
    future   = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    future_only = forecast[forecast['ds'] > prophet_df['ds'].max()]

st.success(f"✅ Model trained! Forecasting {forecast_days} days ahead.")

# ── FORECAST PLOT ────────────────────────────────────────────
st.subheader("🔮 Sales Forecast")
set_style()
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(prophet_df['ds'], prophet_df['y'],
        color='#1976D2', linewidth=2.5, label='Historical Sales', zorder=3)
ax.plot(future_only['ds'], future_only['yhat'],
        color='#E91E63', linewidth=2.5, linestyle='--', label=f'{forecast_days}-Day Forecast')
ax.fill_between(future_only['ds'],
                future_only['yhat_lower'],
                future_only['yhat_upper'],
                alpha=0.2, color='#E91E63', label='95% Confidence Interval')
ax.axvline(x=prophet_df['ds'].max(), color='gray',
           linestyle=':', linewidth=1.5, label='Forecast Start')
ax.set_xlabel('Date'); ax.set_ylabel('Sales')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
st.pyplot(fig)

# ── FORECAST NUMBERS ─────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.metric("Avg Forecasted Sales", f"{future_only['yhat'].mean():,.0f}")
with col2:
    st.metric("Peak Forecasted Sales", f"{future_only['yhat'].max():,.0f}")

# ── DOWNLOAD ─────────────────────────────────────────────────
csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
csv_data.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
csv_data['Date'] = csv_data['Date'].dt.strftime('%Y-%m-%d')
csv_str = csv_data.tail(forecast_days + 5).to_csv(index=False)
st.download_button(
    "⬇️ Download Forecast CSV",
    data=csv_str,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

# ── MODEL EVALUATION ─────────────────────────────────────────
if show_eval:
    st.markdown("---")
    st.subheader("📊 Model Evaluation")
    st.caption("We hold out the last 20% of data and check how well the model predicts it.")

    split_idx   = int(len(prophet_df) * 0.8)
    train_data  = prophet_df.iloc[:split_idx]
    test_data   = prophet_df.iloc[split_idx:]
    eval_model  = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                          daily_seasonality=False)
    eval_model.fit(train_data)
    test_fc     = eval_model.predict(test_data[['ds']])
    actual      = test_data['y'].values
    predicted   = test_fc['yhat'].values

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"{mae:,.1f}", help="Mean Absolute Error — average units off")
    col2.metric("RMSE", f"{rmse:,.1f}", help="Root Mean Squared Error — penalizes big errors")
    col3.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute % Error — % off on average")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(test_data['ds'], actual,    color='#1976D2', linewidth=2, marker='o', ms=4, label='Actual')
    ax2.plot(test_data['ds'], predicted, color='#E91E63', linewidth=2,
             linestyle='--', marker='s', ms=4, label='Predicted')
    ax2.set_title(f'Actual vs Predicted  |  MAPE = {mape:.1f}%')
    ax2.set_xlabel('Date'); ax2.set_ylabel('Sales')
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig2)

# ── COMPONENTS ───────────────────────────────────────────────
if show_comp:
    st.markdown("---")
    st.subheader("🔍 Trend & Seasonality Components")
    st.caption("Prophet breaks the forecast into: overall trend + yearly seasonality pattern.")
    fig3 = model.plot_components(forecast)
    st.pyplot(fig3)

st.markdown("---")
st.caption("Built with Prophet · Streamlit · Matplotlib · Pandas")
