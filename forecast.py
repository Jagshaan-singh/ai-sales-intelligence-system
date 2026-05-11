"""
============================================================
  Sales Forecasting with Time Series Analysis
  Using Facebook Prophet + ARIMA + Linear Regression
============================================================
  Author  : Your Name
  Project : Monthly Sales Forecast (Next 90 Days)
  Dataset : data/sales_data.csv  (Date | Sales)
============================================================
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA # type: ignore

# ────────────────────────────────────────────
# 0. SETUP — Create output folder & plot style
# ────────────────────────────────────────────
os.makedirs('output', exist_ok=True)

def set_plot_style():
    """Apply a clean, professional look to all plots."""
    plt.rcParams.update({
        'figure.facecolor' : 'white',
        'axes.facecolor'   : '#F8F9FA',
        'axes.edgecolor'   : '#DDDDDD',
        'axes.grid'        : True,
        'grid.color'       : '#EEEEEE',
        'grid.linestyle'   : '-',
        'font.family'      : 'DejaVu Sans',
        'axes.titlesize'   : 13,
        'axes.titleweight' : 'bold',
        'axes.labelsize'   : 11,
        'xtick.labelsize'  : 9,
        'ytick.labelsize'  : 9,
    })

set_plot_style()
print("=" * 55)
print("   SALES FORECASTING — Time Series Analysis")
print("=" * 55)


# ────────────────────────────────────────────
# 1. DATA PREPROCESSING
# ────────────────────────────────────────────
print("\n[1] Loading and preprocessing data...")

# Load CSV file
df = pd.read_csv('data/sales_data.csv')

# Convert Date column from text → datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Sort chronologically (oldest first)
df = df.sort_values('Date').reset_index(drop=True)

# Handle missing values by linearly interpolating between known values
df['Sales'] = df['Sales'].interpolate(method='linear')

print(f"    Rows loaded    : {len(df)}")
print(f"    Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"    Missing values : {df.isnull().sum().sum()}")
print(f"    Sales range    : {df['Sales'].min():,.0f} – {df['Sales'].max():,.0f}")


# ────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ────────────────────────────────────────────
print("\n[2] Running EDA and saving plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 11))
fig.suptitle('Sales EDA Dashboard', fontsize=15, fontweight='bold', y=1.01)

# Plot A — Raw sales over time
axes[0].plot(df['Date'], df['Sales'], color='#1976D2', linewidth=2.5)
axes[0].fill_between(df['Date'], df['Sales'], alpha=0.1, color='#1976D2')
axes[0].set_title('Monthly Sales Over Time')
axes[0].set_ylabel('Sales')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')

# Plot B — Rolling average (smooths noise to reveal trend)
df['Rolling_3M'] = df['Sales'].rolling(window=3).mean()
axes[1].plot(df['Date'], df['Sales'], color='gray', alpha=0.4, linewidth=1.5, label='Actual')
axes[1].plot(df['Date'], df['Rolling_3M'], color='#E91E63', linewidth=2.5, label='3-Month Rolling Avg')
axes[1].set_title('Sales Trend (3-Month Rolling Average)')
axes[1].set_ylabel('Sales')
axes[1].legend()
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')

# Plot C — Average by month (reveals seasonality)
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Sales'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
bars = axes[2].bar(monthly_avg.index, monthly_avg.values,
                   color='#4CAF50', edgecolor='white', linewidth=0.8)
axes[2].set_title('Average Sales by Month — Seasonality View')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Average Sales')
axes[2].set_xticks(range(1, 13))
axes[2].set_xticklabels(month_names)
# Label bars
for bar, val in zip(bars, monthly_avg.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('output/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved → output/eda_plots.png")


# ────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ────────────────────────────────────────────
print("\n[3] Engineering features...")

def add_time_features(df):
    """Extract time-based features for ML models."""
    df = df.copy()
    df['Year']          = df['Date'].dt.year
    df['Month']         = df['Date'].dt.month
    df['Quarter']       = df['Date'].dt.quarter
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
    df['Is_Q4']         = df['Month'].isin([10, 11, 12]).astype(int)
    df['Lag_1']         = df['Sales'].shift(1)   # Sales 1 month ago
    df['Lag_3']         = df['Sales'].shift(3)   # Sales 3 months ago
    df.dropna(inplace=True)
    return df

df_feat = add_time_features(df)
print(f"    Features created: Year, Month, Quarter, DaysSinceStart, Is_Q4, Lag_1, Lag_3")
print(f"    Rows after lag   : {len(df_feat)}")


# ────────────────────────────────────────────
# 4. MODEL BUILDING
# ────────────────────────────────────────────
print("\n[4] Training models...")

# ── 4A: Prophet ──────────────────────────────
print("    [Prophet] Training...")
prophet_df = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    interval_width=0.95        # 95% confidence interval
)
model.fit(prophet_df)
print("    [Prophet] Done.")

# ── 4B: Linear Regression ────────────────────
print("    [Linear Regression] Training...")
features = ['Year', 'Month', 'Quarter', 'DaysSinceStart', 'Lag_1', 'Lag_3']
X = df_feat[features]
y_target = df_feat['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, shuffle=False)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_r2 = lr_model.score(X_test, y_test)
print(f"    [Linear Regression] R² = {lr_r2:.3f}")

# ── 4C: ARIMA ────────────────────────────────
print("    [ARIMA] Training...")
arima_model = ARIMA(df['Sales'], order=(2, 1, 2))
arima_result = arima_model.fit()
print("    [ARIMA] Done.")


# ────────────────────────────────────────────
# 5. FORECASTING — Next 90 Days
# ────────────────────────────────────────────
print("\n[5] Generating 90-day forecast...")

future   = model.make_future_dataframe(periods=90, freq='D')
forecast = model.predict(future)

# Separate future predictions from historical fit
future_only = forecast[forecast['ds'] > prophet_df['ds'].max()]

# Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
    'output/forecast.csv', index=False
)

print(f"    Forecast period: {future_only['ds'].min().date()} → {future_only['ds'].max().date()}")
print(f"    Avg predicted sales: {future_only['yhat'].mean():,.0f}")
print("    Saved → output/forecast.csv")

# Plot forecast
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(prophet_df['ds'], prophet_df['y'],
        color='#1976D2', linewidth=2.5, label='Historical Sales', zorder=3)
ax.plot(future_only['ds'], future_only['yhat'],
        color='#E91E63', linewidth=2.5, linestyle='--', label='90-Day Forecast')
ax.fill_between(future_only['ds'],
                future_only['yhat_lower'],
                future_only['yhat_upper'],
                alpha=0.2, color='#E91E63', label='95% Confidence Interval')
ax.axvline(x=prophet_df['ds'].max(), color='gray',
           linestyle=':', linewidth=1.5, label='Forecast Start')
ax.set_title('Sales Forecast — Next 90 Days', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('output/forecast_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved → output/forecast_plot.png")


# ────────────────────────────────────────────
# 6. MODEL EVALUATION
# ────────────────────────────────────────────
print("\n[6] Evaluating model accuracy...")

# Train on 80%, evaluate on last 20%
split_idx  = int(len(prophet_df) * 0.8)
train_data = prophet_df.iloc[:split_idx]
test_data  = prophet_df.iloc[split_idx:]

eval_model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                     daily_seasonality=False)
eval_model.fit(train_data)
test_forecast = eval_model.predict(test_data[['ds']])

actual    = test_data['y'].values
predicted = test_forecast['yhat'].values

mae  = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"\n    ┌─────────────────────────────────┐")
print(f"    │  MAE  : {mae:>10,.1f} units        │")
print(f"    │  RMSE : {rmse:>10,.1f} units        │")
print(f"    │  MAPE : {mape:>10.1f} %            │")
print(f"    └─────────────────────────────────┘")

# Quality assessment
if mape < 5:
    quality = "Excellent (< 5%)"
elif mape < 10:
    quality = "Good (5–10%)"
elif mape < 20:
    quality = "Acceptable (10–20%)"
else:
    quality = "Needs improvement (> 20%)"
print(f"    Forecast quality: {quality}")

# Evaluation plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(test_data['ds'], actual, color='#1976D2', linewidth=2, marker='o', ms=4, label='Actual')
axes[0].plot(test_data['ds'], predicted, color='#E91E63', linewidth=2,
             linestyle='--', marker='s', ms=4, label='Predicted')
axes[0].set_title(f'Actual vs Predicted | MAE={mae:,.0f} | MAPE={mape:.1f}%')
axes[0].set_xlabel('Date'); axes[0].set_ylabel('Sales')
axes[0].legend()
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')

errors = actual - predicted
axes[1].hist(errors, bins=10, color='#9C27B0', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero error')
axes[1].axvline(errors.mean(), color='#E91E63', linestyle='-', linewidth=1.5,
                label=f'Mean error = {errors.mean():,.0f}')
axes[1].set_title('Prediction Error Distribution')
axes[1].set_xlabel('Error (Actual − Predicted)'); axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.savefig('output/evaluation_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved → output/evaluation_plot.png")


# ────────────────────────────────────────────
# 7. PROPHET COMPONENT PLOTS (Bonus)
# ────────────────────────────────────────────
print("\n[7] Generating component plots...")

fig_comp = model.plot_components(forecast)
fig_comp.suptitle('Forecast Components: Trend & Seasonality', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/components.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved → output/components.png")


# ────────────────────────────────────────────
# DONE
# ────────────────────────────────────────────
print("\n" + "=" * 55)
print("   ALL DONE! Check the 'output/' folder for results.")
print("   Files created:")
print("     output/eda_plots.png      — EDA charts")
print("     output/forecast_plot.png  — 90-day forecast")
print("     output/forecast.csv       — raw forecast numbers")
print("     output/evaluation_plot.png — model accuracy")
print("     output/components.png     — trend & seasonality")
print("=" * 55)
