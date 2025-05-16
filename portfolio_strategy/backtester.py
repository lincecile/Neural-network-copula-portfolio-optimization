# portfolio_strategy/backtester.py

import numpy as np
import pandas as pd
from portfolio_strategy.optimizer_cvar import solve_cvar_lp
from portfolio_strategy.scenario_generator import generate_return_scenarios

def run_backtest(weekly_returns, copula_model, start_date=None, end_date=None, alpha=0.95, allow_short=False):
    portfolio_returns = []
    dates = []

    for date in copula_model.weekly_dates:
        if start_date and date < start_date:
            print(f"{date}: ❌ Skipped (before start_date)")
            continue
        if end_date and date > end_date:
            print(f"{date}: ❌ Skipped (after end_date)")
            continue
        if date not in copula_model.copula_params:
            print(f"{date}: ❌ Skipped (missing copula parameters)")
            continue
        if date not in weekly_returns.index:
            print(f"{date}: ❌ Skipped (missing forecast return)")
            continue

        try:
            scenarios = generate_return_scenarios(copula_model, date, n_samples=1000)
            weights = solve_cvar_lp(scenarios, alpha=alpha, allow_short=allow_short)

            returns_at_date = weekly_returns.loc[date].values
            realized_return = returns_at_date @ weights

            # Estimate CVaR
            simulated_returns = generate_return_scenarios(copula_model, date, n_samples=1000)
            simulated_portfolio_returns = simulated_returns @ weights
            losses = -simulated_portfolio_returns
            sorted_losses = np.sort(losses)
            cvar_cut = int((1 - alpha) * len(sorted_losses))
            cvar_est = sorted_losses[cvar_cut:].mean()

            portfolio_returns.append(realized_return)
            dates.append(date)

            print(f"{date}: ✅ return = {realized_return:.4f} | CVaR = {cvar_est:.4f} | weights = {np.round(weights, 4)}")
        except Exception as e:
            print(f"{date}: ❌ Error during optimization or simulation: {e}")
            continue

    return pd.Series(portfolio_returns, index=dates, name="CVaR_Portfolio")
