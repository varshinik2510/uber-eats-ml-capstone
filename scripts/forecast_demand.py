import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "data"
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_demand_forecasting():
    print("[1/5] Loading orders dataset and preparing hourly time-series...")
    df_orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    
    time_col = [c for c in ['order_timestamp', 'timestamp', 'created_at'] if c in df_orders.columns][0]
    df_orders['order_timestamp'] = pd.to_datetime(df_orders[time_col])

    # 1. Aggregate into hourly order volume
    df_hourly = df_orders.set_index('order_timestamp').resample('h').size().reset_index()
    df_hourly.columns = ['timestamp', 'order_volume']

    hourly_path = os.path.join(DATA_DIR, "hourly_demand.csv")
    df_hourly.to_csv(hourly_path, index=False)
    print(f" -> Hourly demand series saved to '{hourly_path}' ({len(df_hourly)} hours).")

    # 2. Augmented Dickey-Fuller (ADF) Stationarity Test
    print("\n[2/5] Running Augmented Dickey-Fuller (ADF) Stationarity Test...")
    series = df_hourly['order_volume'].values
    adf_result = adfuller(series)
    adf_stat, p_value = adf_result[0], adf_result[1]
    
    print(f" -> ADF Statistic: {adf_stat:.4f}")
    print(f" -> p-value:       {p_value:.4e}")
    if p_value < 0.05:
        print(" -> Verdict: Series is STATIONARY (p < 0.05). Suitable for ARMA/ARIMA modeling.")
    else:
        print(" -> Verdict: Series is NON-STATIONARY. First-order differencing (d=1) required.")

    # 3. Time-Series Seasonal Decomposition (24-hour cycle)
    print("\n[3/5] Performing Time-Series Seasonal Decomposition (Period=24h)...")
    decomp_series = df_hourly.set_index('timestamp')['order_volume']
    decomposition = seasonal_decompose(decomp_series, model='additive', period=24)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    decomposition.observed.plot(ax=axes[0], color='#2E86AB', title='Observed Hourly Demand')
    decomposition.trend.plot(ax=axes[1], color='#E74C3C', title='Trend Component')
    decomposition.seasonal.plot(ax=axes[2], color='#27AE60', title='Seasonal Pattern (24-Hour Cycle)')
    decomposition.resid.plot(ax=axes[3], color='#8E44AD', title='Residual / Noise')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    decomp_path = os.path.join(FIGURES_DIR, "demand_time_series_decomposition.png")
    plt.savefig(decomp_path, dpi=300)
    plt.close()
    print(f" -> Decomposition figure saved to '{decomp_path}'.")

    # 4. Chronological Train-Test Split (Hold out last 48 hours for validation)
    print("\n[4/5] Fitting ARIMA Forecasting Model with Chronological Validation...")
    test_horizon = min(48, int(len(df_hourly) * 0.15))
    train_series = series[:-test_horizon]
    test_series = series[-test_horizon:]

    # Fit ARIMA (p=2, d=1, q=2)
    arima_model = ARIMA(train_series, order=(2, 1, 2))
    fitted_model = arima_model.fit()

    predictions = fitted_model.forecast(steps=test_horizon)
    predictions = np.clip(predictions, 0, None)  # Volume cannot be negative

    mae = mean_absolute_error(test_series, predictions)
    rmse = np.sqrt(mean_squared_error(test_series, predictions))
    mape = np.mean(np.abs((test_series - predictions) / np.maximum(test_series, 1))) * 100

    print(f" -> Out-of-Sample MAE:  {mae:.2f} orders/hour")
    print(f" -> Out-of-Sample RMSE: {rmse:.2f} orders/hour")
    print(f" -> Out-of-Sample MAPE: {mape:.2f}%")

    # Save metrics
    metrics_df = pd.DataFrame([{
        'model': 'ARIMA(2,1,2)',
        'adf_p_value': round(p_value, 5),
        'test_horizon_hours': test_horizon,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape_percent': round(mape, 2)
    }])
    metrics_df.to_csv(os.path.join(DATA_DIR, "demand_forecasting_metrics.csv"), index=False)

    # 5. Forecast Plot
    print("\n[5/5] Generating Hourly Demand Forecast visualization...")
    plt.figure(figsize=(12, 5))
    time_indices = df_hourly['timestamp'].values
    
    # Plot recent history (last 120 hours) and forecast
    plot_window = min(len(df_hourly), 120)
    plt.plot(time_indices[-plot_window:], series[-plot_window:], label='Actual Hourly Demand', color='#2C3E50', lw=2)
    plt.plot(time_indices[-test_horizon:], predictions, label='ARIMA Out-of-Sample Forecast', color='#E67E22', lw=2.5, linestyle='--')
    plt.axvline(time_indices[-test_horizon], color='red', linestyle=':', label='Forecast Start Horizon')

    plt.title(f"Uber Eats Hourly Order Demand: Actual vs. ARIMA Forecast (Horizon: {test_horizon}h)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Hourly Order Volume", fontsize=11)
    plt.legend(loc='upper left')
    plt.tight_layout()

    forecast_path = os.path.join(FIGURES_DIR, "demand_forecast_evaluation.png")
    plt.savefig(forecast_path, dpi=300)
    plt.close()
    print(f" -> Forecast plot saved to '{forecast_path}'.")
    print("\n[Done] Day 5 Demand Forecasting module completed successfully.")

if __name__ == "__main__":
    run_demand_forecasting()