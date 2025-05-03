import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit  # Logistic sigmoid function

def star_model(series: pd.Series, order: int, threshold: float, gamma: float):
    """
    Create a STAR model (LSTAR variant) from the given Series.
    
    The model is defined as:
        y_t = α + ∑(φ_i * y_{t-i}) + [β + ∑(θ_i * y_{t-i})] * G(z_t) + ε_t
    where G(z_t) is a logistic transition function:
        G(z_t) = 1 / (1 + exp(-gamma*(z_t - threshold)))
    Here we use the first lag as the threshold variable.
    
    Parameters:
        series (pd.Series): Time-series data.
        order (int): The autoregressive order (number of lags).
        threshold (float): Threshold value for the transition function.
        gamma (float): Smoothness parameter for the logistic transition.
    
    Returns:
        RegressionResults: The fitted STAR model.
    """
    # Prepare the data frame with lagged values
    data = pd.DataFrame({'y': series})
    for i in range(1, order + 1):
        data[f'lag_{i}'] = series.shift(i)
    
    # Use the first lag as the threshold variable
    data['G'] = expit(gamma * (series.shift(1) - threshold))
    
    # Build the design matrix with both linear and transition components
    X = pd.DataFrame()
    X['const'] = 1
    for i in range(1, order + 1):
        X[f'lag_{i}'] = data[f'lag_{i}']
        X[f'lag_{i}_G'] = data[f'lag_{i}'] * data['G']
    
    # Drop rows with NaN values (introduced by shifting)
    valid_idx = data.dropna().index
    X = X.loc[valid_idx]
    y = data.loc[valid_idx, 'y']
    
    # Fit the model using ordinary least squares
    model = sm.OLS(y, X)
    fitted_model = model.fit()
    
    return fitted_model