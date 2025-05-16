# run_opti_arma_adcc_skt.py

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

# === Imports
from Copula import SkewedTCopulaModel
from portfolio_strategy.backtester import run_backtest
from clean_df_paper import df_out_sample_set_weekly, df_out_sample_set_daily

# === Load ADCC matrices ===
with open("adcc_results_all_weekly.pkl", "rb") as f:
    adcc_data = pickle.load(f)

# === Load ARMA forecasts ===
with open("arma_results.pkl", "rb") as f:
    arma_forecasts = pickle.load(f)

# === Convert ARMA forecasts to weekly returns ===
returns_dict = {}
for ticker, content in arma_forecasts.items():
    pred = content['predictions']
    full_idx = df_out_sample_set_daily[ticker].index[-len(pred):]
    daily_series = pd.Series(pred, index=pd.to_datetime(full_idx)).sort_index()

    weekly_series = daily_series.resample('W-MON').ffill()
    weekly_series = weekly_series[weekly_series.index.isin(df_out_sample_set_weekly.index)]

    clean_name = ticker.replace(" US Equity", "")
    returns_dict[clean_name] = weekly_series

    print(f"\n{ticker} — ARMA daily: {daily_series.index.min()} to {daily_series.index.max()}")
    print(f"{ticker} — Resampled weekly: {weekly_series.index.min()} to {weekly_series.index.max()}")
    print(f"{ticker} — Matching weekly dates count: {len(weekly_series)}")

forecast_df = pd.DataFrame(returns_dict)

print("\n=== Forecast Weekly Returns Index Preview ===")
print(forecast_df.index[:5])
print("=== ADCC Weekly Dates Preview ===")
print(adcc_data["weekly_dates"][:5])

# === Load Copula (without calling __init__)
copula = SkewedTCopulaModel.__new__(SkewedTCopulaModel)
copula.weekly_matrices = adcc_data['matrices']
copula.weekly_dates = adcc_data['weekly_dates']

# === Load estimated copula parameters
with open("copula_params_adcc.pkl", "rb") as f:
    copula_data = pickle.load(f)
    copula.copula_params = copula_data.get("copula_params", copula_data)

# === Determine first valid date for which we have copula params
valid_dates = sorted(copula.copula_params.keys())
start_date = valid_dates[0]
forecast_df = forecast_df[forecast_df.index >= start_date]
print("\nFirst valid copula param date:", start_date)
print("Trimmed forecast_df start:", forecast_df.index.min())

# === Sync dates: only use those present in all required structures
synced_dates = []
for d in copula.weekly_dates:
    reason = []
    if d not in forecast_df.index:
        reason.append("missing in forecast_df")
    if d not in copula.copula_params:
        reason.append("missing in copula_params")
    if d not in copula.weekly_matrices:
        reason.append("missing in weekly_matrices")

    if not reason:
        synced_dates.append(d)
        print(f"{d}: ✅ synced")
    else:
        print(f"{d}: ❌ skipped ({', '.join(reason)})")

copula.weekly_dates = synced_dates
print("\n=== Final Synced Dates ===")
print(copula.weekly_dates[:5])
print(f"Total synced dates: {len(copula.weekly_dates)}")

# === Run backtest
print("\n=== Running Panel B strategy: ARMA-ADCC-Skewed t Copula (Long Only) ===")
perf_series = run_backtest(
    weekly_returns=forecast_df,
    copula_model=copula,
    start_date=start_date,
    alpha=0.95,
    allow_short=False
)

# === Inspect results
print("\n=== perf_series Preview ===")
print(perf_series.head())
print("Type:", type(perf_series))
print("Is empty:", perf_series.empty)
print("Dtype:", perf_series.dtype)

# === Save & plot
if isinstance(perf_series, pd.Series) and not perf_series.empty and pd.api.types.is_numeric_dtype(perf_series):
    perf_series.cumsum().plot(
        title="Panel B: ARMA-ADCC-Skewed t Copula (Long Only)",
        figsize=(10, 5)
    )
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("arma_adcc_skt_cumulative_return.png")
    plt.show()

    perf_series.to_csv("arma_adcc_skt_returns.csv")

    # Extra stats
    realized_return = perf_series.sum() * 100
    downside = perf_series[perf_series < 0]
    sortino = perf_series.mean() / downside.std() if not downside.empty else 0
    cvar = perf_series[perf_series <= perf_series.quantile(0.05)].mean()
    cvar_ratio = perf_series.mean() / abs(cvar) if cvar else float('inf')
    mdd = (perf_series.cumsum().cummax() - perf_series.cumsum()).max()

    print("\n=== Panel B: ARMA-ADCC-SKT Performance ===")
    print(f"Realized return (%):     {realized_return:.3f}")
    print(f"Return / CVaR:           {cvar_ratio:.4f}")
    print(f"Sortino ratio:           {sortino:.4f}")
    print(f"Max drawdown (%):        {mdd * 100:.3f}")
else:
    print("⚠️ No valid portfolio returns to plot. Check perf_series content.")
