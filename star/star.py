import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Logistic transition function
def logistic_transition(z, gamma, c):
    return 1 / (1 + np.exp(-gamma * (z - c)))

def create_lagged_matrix(y, lags):
    # Create a DataFrame of lagged values
    lagged = pd.concat([y.shift(i) for i in range(1, lags + 1)], axis=1)
    lagged.columns = [f"lag_{i}" for i in range(1, lags + 1)]
    # Drop rows with NaN values due to shifting
    lagged = lagged.dropna()
    return lagged

def star_model(y, lags=1, transition_variable=None, d=1):
    """
    Estimate LSTAR model using Nonlinear Least Squares.

    Parameters:
    y: pandas Series
        Time series data.
    lags: int
        Number of autoregressive lags.
    transition_variable: pandas Series or None
        Transition variable z_t (default: y lagged by d)
    d: int
        Delay parameter if transition_variable is None

    Returns:
    result: dict
        Dictionary with estimated parameters, fitted values, and diagnostics.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    T = len(y)
    X = create_lagged_matrix(y, lags)
    y_dep = y.shift(-lags).dropna().iloc[:len(X)]

    # Default transition variable = lagged y by d
    if transition_variable is None:
        z = y.shift(d).dropna().iloc[lags:]
    else:
        z = pd.Series(transition_variable).dropna().iloc[lags:]

    # Align everything
    common_index = X.index.intersection(y_dep.index).intersection(z.index)
    X = X.loc[common_index]
    y_dep = y_dep.loc[common_index]
    z = z.loc[common_index]

    # Initial parameters guess (phi + beta + gamma + c)
    phi0 = np.zeros(lags + 1)
    beta0 = np.zeros(lags + 1)
    gamma0 = 1.0
    c0 = np.median(z)
    init_params = np.hstack([phi0, beta0, gamma0, c0])

    def model(params):
        phi = params[0:lags + 1]
        beta = params[lags + 1:2 * lags + 2]
        gamma = params[-2]
        c = params[-1]

        G = logistic_transition(z.values, gamma, c)
        X_const = np.column_stack([np.ones(len(X))] + [X.iloc[:, i] for i in range(X.shape[1])])

        linear_part = X_const @ phi
        nonlinear_part = X_const @ beta

        return linear_part + G * nonlinear_part

    def objective(params):
        fitted = model(params)
        return np.sum((y_dep.values - fitted) ** 2)

    opt_result = minimize(objective, init_params, method='L-BFGS-B')

    est_params = opt_result.x
    fitted_vals = model(est_params)

    # Diagnostics
    residuals = y_dep - pd.Series(fitted_vals, index=common_index)
    n = len(y_dep)
    k = len(est_params)
    rss = np.sum(residuals ** 2)
    tss = np.sum((y_dep - np.mean(y_dep)) ** 2)
    sigma2 = rss / (n - k)

    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    r2 = 1 - rss / tss

    result = {
        'params': est_params,
        'phi': est_params[0:lags + 1],
        'beta': est_params[lags + 1:2 * lags + 2],
        'gamma': est_params[-2],
        'c': est_params[-1],
        'fitted': pd.Series(fitted_vals, index=common_index),
        'residuals': residuals,
        'aic': aic,
        'bic': bic,
        'r_squared': r2,
        'sigma2': sigma2,
        'success': opt_result.success,
        'message': opt_result.message
    }

    return result