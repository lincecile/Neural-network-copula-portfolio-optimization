import numpy as np

# Fonction MAE
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Fonction MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    # Assurez-vous que y_true ne contient pas de zéros pour éviter la division par zéro
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Fonction RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Fonction THEIL-U
def theil_u_statistic(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rmse_pred = np.sqrt(np.mean(y_pred ** 2))
    rmse_true = np.sqrt(np.mean(y_true ** 2))
    return rmse / (rmse_pred + rmse_true)