import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory to project root
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure dependencies
try:
    import cvxpy, arch
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy", "arch"])

# Imports
from Copula import SkewedTCopulaModel
from portfolio_strategy.backtester import run_backtest
from clean_df_paper import df_out_sample_set_weekly, df_out_sample_set_daily

# === Load DCC matrices ===
with open("dcc_correlation_results_all_weekly.pkl", "rb") as f:
    dcc_data = pickle.load(f)

# === Load ARMA forecasts ===
with open("arma_results.pkl", "rb") as f:
    arma_forecasts = pickle.load(f)

# === Convert ARMA daily forecasts to weekly returns ===
returns_dict = {}
for ticker, content in arma_forecasts.items():
    pred = content['predictions']
    full_idx = df_out_sample_set_daily[ticker].index[-len(pred):]
    daily_series = pd.Series(pred, index=full_idx)

    daily_series.index = pd.to_datetime(daily_series.index)
    daily_series = daily_series.sort_index()

    # Align to weekly Monday dates
    weekly_series = daily_series.resample('W-MON').last()
    weekly_series = weekly_series[weekly_series.index.isin(df_out_sample_set_weekly.index)]

    print(f"\n{ticker} — ARMA daily: {daily_series.index.min()} to {daily_series.index.max()}")
    print(f"{ticker} — Resampled weekly: {weekly_series.index.min()} to {weekly_series.index.max()}")
    print(f"{ticker} — Matching weekly dates count: {len(weekly_series)}")

    clean_name = ticker.replace(" US Equity", "")
    returns_dict[clean_name] = weekly_series

forecast_df = pd.DataFrame(returns_dict)

# === Sanity check ===
print("=== Forecast Weekly Returns Index Preview ===")
print(forecast_df.index[:5])
print("=== DCC Weekly Dates Preview ===")
print(list(dcc_data['weekly_dates'])[:5])

# === Load Copula without __init__ ===
copula = SkewedTCopulaModel.__new__(SkewedTCopulaModel)
copula.weekly_matrices = dcc_data['matrices']
copula.weekly_dates = dcc_data['weekly_dates']

# Inject dummy copula parameters for simulation
dummy_tickers = forecast_df.columns.tolist()
copula.copula_params = {
    date: {
        'df': 5,
        'gamma': {ticker: 1.0 for ticker in dummy_tickers}
    }
    for date in copula.weekly_dates
}

print("\n=== Copula Params Example ===")
first_date = next(iter(copula.weekly_dates))
print(f"Date: {first_date}")
print("Params:", copula.copula_params.get(first_date, "NOT FOUND"))

# === Run backtest ===
print("\n=== Running Panel B strategy: ARMA-DCC-Skewed t Copula (Long Only) ===")
perf_series = run_backtest(
    weekly_returns=forecast_df,
    copula_model=copula,
    alpha=0.95,
    allow_short=False
)

# === Inspect result ===
print("\n=== perf_series Preview ===")
print(perf_series.head())
print("Type:", type(perf_series))
print("Is empty:", perf_series.empty)
print("Dtype:", perf_series.dtype)

# === Save & Plot ===
if isinstance(perf_series, pd.Series) and not perf_series.empty and pd.api.types.is_numeric_dtype(perf_series):
    perf_series.cumsum().plot(
        title="Panel B: ARMA-DCC-Skewed t Copula (Long Only)",
        figsize=(10, 5)
    )
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("arma_dcc_skt_cumulative_return.png")
    plt.show()

    perf_series.to_csv("arma_dcc_skt_returns.csv")
else:
    print("⚠️ No valid portfolio returns to plot. Check perf_series content.")


import numpy as np

# === Performance Metrics ===
returns = perf_series.dropna()
mean_return = returns.mean()
std_return = returns.std()
downside_std = returns[returns < 0].std()
cumulative = (1 + returns).cumprod()

# CVaR at 95%
alpha = 0.95
cvar_95 = -returns[returns <= returns.quantile(1 - alpha)].mean()

# Max drawdown
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("\n=== Panel B: ARMA-DCC-SKT Performance ===")
print(f"Realized return (%):       {mean_return * 52 * 100:.3f}")
print(f"Return / CVaR:             {mean_return / cvar_95:.4f}")
print(f"Sortino ratio:             {mean_return / downside_std:.4f}")
print(f"Max drawdown (%):          {max_drawdown * 100:.3f}")
