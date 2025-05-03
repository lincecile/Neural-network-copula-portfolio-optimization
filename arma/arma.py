import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arma_model(series: pd.Series, order: tuple[int, int]):
    """
    Create an ARMA model from the given Series.

    Parameters:
        series (pd.Series): Time-series data.
        order (tuple): ARMA order as (p, q).

    Returns:
        ARIMAResults: The fitted ARMA model.
    """
    model = ARIMA(series, order=(order[0], 0, order[1]))
    fitted_model = model.fit()
    return fitted_model
