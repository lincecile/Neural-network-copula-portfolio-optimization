import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from clean_df_paper import df_total_set_daily, df_training_set_daily, df_test_set_daily, df_out_sample_set_daily
from metrics_forecast_error import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, theil_u_statistic

# ----- Paramètres -----

## SPY US Equity, DIA US Equity, QQQ US Equity 
rnn_config = {
    "tickers": ["SPY US Equity", "DIA US Equity", "QQQ US Equity"],
    "lags": [[1, 2, 3, 5, 7, 8, 9, 10, 12], [1, 3, 4, 6, 7, 8, 9, 10], [1, 4, 5, 6, 7, 9, 10, 12]],
    "learning_algorithm": ["sgd", "sgd", "sgd"],
    "learning_rate": [0.003, 0.005, 0.002],
    "momentum": [0.005, 0.006, 0.005],
    "iteration_steps": [40000, 35000, 40000],
    "init_weights": ["random", "random", "random"],
    "hidden_nodes": [6, 7, 10]
}

