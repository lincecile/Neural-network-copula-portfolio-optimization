import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from arch.__future__ import reindexing

from clean_df_paper import df_total_set_weekly

spx_returns = df_total_set_weekly["SPY US Equity"] * 100

window = 52  # one year (52 weeks) of data

rolling_params = []
forecast_dates = []
forecast_variances = []

for start in range(0, len(spx_returns) - window + 1):
    window_data = spx_returns.iloc[start : start + window]
    
    garch_model = arch_model(window_data, p=1, q=1,
                             mean='constant', vol='GARCH', dist='normal')
    
    gm_result = garch_model.fit(disp='off')
    
    rolling_params.append(gm_result.params)
    
    print(f"Window: {window_data.index[0]} to {window_data.index[-1]}")
    print(gm_result.params)
    print("-" * 40)
    
    # Forecast one week ahead variance
    forecast = gm_result.forecast(horizon=1)
    # Extract the variance forecast for the next period from the last row
    f_variance = forecast.variance.iloc[-1, 0]
    # Compute forecast date (assuming weekly frequency)
    forecast_date = window_data.index[-1] + pd.DateOffset(weeks=1)
    
    forecast_variances.append(f_variance)
    forecast_dates.append(forecast_date)

# Compute realized variance for the forecast dates if available (using squared returns)
realized_variances = []
for date in forecast_dates:
    if date in spx_returns.index:
        realized_variance = spx_returns.loc[date] ** 2
        realized_variances.append(realized_variance)
    else:
        realized_variances.append(float('nan'))

# Create a plot comparing forecast variance with realized variance
plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, forecast_variances, marker='o', linestyle='-', label='Forecast Variance')
plt.plot(forecast_dates, realized_variances, marker='x', linestyle='--', label='Realized Variance')
plt.xlabel('Forecast Date')
plt.ylabel('Variance')
plt.title('Forecast vs Realized Variance from Rolling GARCH(1,1) Fits')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()