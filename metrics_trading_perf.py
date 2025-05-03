import numpy as np

def evaluate_strategy(y_true, y_pred, freq=252):
    """
    Évalue les performances d'une stratégie de trading basée sur les prédictions.
    y_true : rendements réels (array-like)
    y_pred : rendements prédits (array-like)
    freq : nombre de périodes par an (252 pour journalier)
    """

    # Générer des positions (1 = long, -1 = short)
    positions = np.sign(y_pred)

    # Rendements de la stratégie : position * rendement réel
    strategy_returns = positions * y_true

    # Annualized return
    avg_daily_return = np.mean(strategy_returns)
    annualized_return = (1 + avg_daily_return)**freq - 1

    # Sharpe ratio (sans taux sans risque ici)
    sharpe_ratio = np.sqrt(freq) * avg_daily_return / np.std(strategy_returns, ddof=1)

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    return annualized_return, sharpe_ratio, max_drawdown
