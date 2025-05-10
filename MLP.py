import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from clean_df_paper import df_training_set_daily, df_test_set_daily, df_out_sample_set_daily
from metrics_forecast_error import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, theil_u_statistic
from test_statistic import diebold_mariano_test, pesaran_timmermann_test

# ----- Paramètres -----

## SPY US Equity, DIA US Equity, QQQ US Equity 
mlp_config = {
    "tickers": ["SPY US Equity", "DIA US Equity", "QQQ US Equity"],
    "lags": [[1, 3, 5, 6, 8, 9, 12], [2, 4, 5, 7, 9, 10, 11], [1, 2, 3, 5, 6, 8, 10, 11, 12]],
    "learning_algorithm": ["sgd", "sgd", "sgd"],
    "learning_rate": [0.003, 0.002, 0.003],
    "momentum": [0.004, 0.005, 0.005],
    "iteration_steps": [30000, 45000, 30000],
    "init_weights": ["random", "random", "random"],
    "hidden_nodes": [6, 9, 8]
}

# ----- Fonction pour créer des lags -----
def create_lag_features(df, target_col, lags=[1, 2, 3]):
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)
    df_lagged['target'] = df_lagged[target_col].shift(-1)  # prédire le return à t+1
    return df_lagged.dropna()

#define the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        # Initialisation normale
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        x = self.output_layer(x)
        return x


# ----- Évaluation -----
def evaluate_performance(y_true, y_pred, dataset_name="Dataset"):
    # Ensure both arrays are 1-dimensional
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    theil = theil_u_statistic(y_true, y_pred)
    
    print(f"📊 {dataset_name}:")
    print(f"MAE         : {mae:.4f}")
    print(f"MAPE        : {mape:.4f}")
    print(f"RMSE        : {rmse:.4f}")
    print(f"MSE         : {mse:.4f}")
    print(f"R2 Score    : {r2:.4f}")
    print(f"Theil's U  : {theil:.4f}")
    print("=" * 40)


def main(ticker, lags_list, learning_algorithm, learning_rate, momentum, iteration_steps, init_weights, hidden_nodes):

    # ----- Data -----
    df_training_set = df_training_set_daily[ticker].to_frame()
    df_test_set = df_test_set_daily[ticker].to_frame()
    df_out_sample_set = df_out_sample_set_daily[ticker].to_frame()

    # ----- Création des features -----
    df_train_lagged = create_lag_features(df_training_set, target_col=df_training_set.columns[0], lags=lags_list)
    df_test_lagged = create_lag_features(df_test_set, target_col=df_training_set.columns[0], lags=lags_list)
    df_out_lagged = create_lag_features(df_out_sample_set, target_col=df_training_set.columns[0], lags=lags_list)

    # ----- Séparation des features/target -----
    X_train = df_train_lagged.drop(columns=['target',df_train_lagged.columns[0]])
    y_train = df_train_lagged['target']

    X_test = df_test_lagged.drop(columns=['target',df_test_lagged.columns[0]])
    y_test = df_test_lagged['target']

    X_out = df_out_lagged.drop(columns=['target',df_out_lagged.columns[0]])
    y_out = df_out_lagged['target']

    # ----- Standardisation -----
    scaler = StandardScaler()
    X_train_scaled =  scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_out_scaled = scaler.transform(X_out)

    #Convert to torch tensors
    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
    X_out_scaled = torch.tensor(X_out_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    y_out_tensor = torch.tensor(y_out.values, dtype=torch.float32).view(-1, 1)

    # Instantiate the model with 7 input nodes
    model = MLP(input_size=X_train_scaled.shape[1], hidden_size=hidden_nodes)

    # Define loss function and optimizer using Gradient Descent (SGD)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Training loop with 30000 iterations
    num_iterations = iteration_steps
    for step in range(num_iterations):
        optimizer.zero_grad()
        predictions = model(X_train_scaled)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Make predictions on test and out-of-sample sets
    y_pred_test = model(X_test_scaled).detach().numpy()
    y_pred_out = model(X_out_scaled).detach().numpy()

    evaluate_performance(y_test, y_pred_test, "Test Set")
    evaluate_performance(y_out, y_pred_out, "Out-of-Sample")
    
    return y_test, y_pred_test, y_out, y_pred_out

# ----- Boucle sur les tickers -----

result_dict = {}
result_dict_df = {}

for i in range(len(mlp_config["tickers"])):
    ticker = mlp_config["tickers"][i]
    lags_list = mlp_config["lags"][i]
    learning_algorithm = mlp_config["learning_algorithm"][i]
    learning_rate = mlp_config["learning_rate"][i]
    momentum = mlp_config["momentum"][i]
    iteration_steps = mlp_config["iteration_steps"][i]
    init_weights = mlp_config["init_weights"][i]
    hidden_nodes = mlp_config["hidden_nodes"][i]

    print(f"------------------ Ticker: {ticker} ------------------")

    y_test, y_pred_test, y_out, y_pred_out = main(ticker, lags_list, learning_algorithm, learning_rate, momentum, iteration_steps, init_weights, hidden_nodes)

    result_dict[ticker] = y_pred_out
    # print(f"y_pred_out: {y_pred_out}")
    # exit()
# exit()
# # ----- Test Statistique -----
# # Prédiction naïve = lag 1 (car on prédit t+1 à partir de t)
# y_pred_naive_out = df_out_lagged['lag_1'].values

# dm_stat = diebold_mariano_test(y_out.values, y_pred_out, y_pred_naive_out)
# print(f"📉 DM Test vs Naive Out-of-Sample: Statistic = {dm_stat:.3f}")

# pt_stat, pt_pval = pesaran_timmermann_test(y_out, np.array(y_pred_out).flatten())
# print(f"\n🧭 PT Test Out-of-Sample: Statistic = {pt_stat:.3f}, p-value = {pt_pval:.3f}")

# y_pred_out: [[ 5.3424825e-04]
#  [ 5.2006292e-04]
#  [ 7.8006391e-04]
#  [ 9.8844210e-04]
#  [-1.3810538e-03]
#  [-5.3428103e-05]
#  [-1.6644389e-03]
#  [-3.2597394e-03]
#  [ 4.3460820e-03]