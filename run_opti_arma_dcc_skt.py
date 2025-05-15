import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory to project root (adjust as needed)
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

# === Load ARMA return forecasts ===
with open("arma_results.pkl", "rb") as f:
    arma_forecasts = pickle.load(f)

# === Reformat ARMA predictions to weekly returns ===
returns_dict = {}
for ticker, content in arma_forecasts.items():
    pred = content['predictions']
    full_idx = df_out_sample_set_daily.index[-len(pred):]
    daily_series = pd.Series(pred, index=full_idx)

    # Align to weekly frequency (last daily prediction before each rebalance)
    weekly_series = daily_series.resample('W-FRI').last()
    weekly_series = weekly_series[weekly_series.index.isin(df_out_sample_set_weekly.index)]

    clean_name = ticker.replace(" US Equity", "")
    returns_dict[clean_name] = weekly_series

forecast_df = pd.DataFrame(returns_dict)

# === Load Copula (without running init) and inject DCC matrices ===
copula = SkewedTCopulaModel.__new__(SkewedTCopulaModel)
copula.weekly_matrices = dcc_data['matrices']
copula.weekly_dates = dcc_data['weekly_dates']

# Option 1: Inject dummy copula_params to satisfy generator logic
copula.copula_params = {
    date: {'nu': 5, 'gamma': [0, 0, 0]} for date in copula.weekly_dates
}


print("=== Copula Params Example ===")
first_date = next(iter(copula.weekly_dates))
print(f"Date: {first_date}")
print("Params:", copula.copula_params.get(first_date, "NOT FOUND"))


# === Run backtest ===
print("\n=== Running Panel B strategy: ARMA-DCC-Skewed t Copula (Long Only) ===")
perf_series = run_backtest(
    weekly_returns=forecast_df,
    copula_model=copula,
    alpha=0.95,
    allow_short=False  # STRICTLY Panel B logic
)

print("\n=== perf_series Preview ===")
print(perf_series.head())
print("Type:", type(perf_series))
print("Is empty:", perf_series.empty)
print("Dtype:", perf_series.dtype)


# === Save & Plot Results ===
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
else:
    print("⚠️ No valid portfolio returns to plot. Check perf_series content.")
