# portfolio_strategy/backtester.py

import pandas as pd
from portfolio_strategy.optimizer_cvar import solve_cvar_lp
from portfolio_strategy.scenario_generator import generate_return_scenarios

def run_backtest(weekly_returns, copula_model, start_date=None, end_date=None, alpha=0.95, allow_short=False):
    """
    Run weekly CVaR-based portfolio optimization backtest.
    
    Args:
        weekly_returns (pd.DataFrame): Actual weekly returns (used for performance).
        copula_model (SkewedTCopulaModel): Trained copula with correlation matrices and parameters.
        start_date (optional): Start of backtest.
        end_date (optional): End of backtest.
        alpha (float): CVaR confidence level.
        allow_short (bool): Whether short-selling is allowed.
    
    Returns:
        pd.Series: Portfolio returns over time.
    """
    portfolio_returns = []
    dates = []

    for date in copula_model.weekly_dates:
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue

        try:
            scenarios = generate_return_scenarios(copula_model, date, n_samples=1000)
            weights = solve_cvar_lp(scenarios, alpha=alpha, allow_short=allow_short)

            # Get realized return
            if date in weekly_returns.index:
                realized_return = weekly_returns.loc[date].values @ weights
                portfolio_returns.append(realized_return)
                dates.append(date)
                print(f"{date}: return = {realized_return:.4f}")
        except Exception as e:
            print(f"{date}: Error during optimization or data. Skipping. ({e})")
            continue

    return pd.Series(portfolio_returns, index=dates, name="CVaR_Portfolio")
