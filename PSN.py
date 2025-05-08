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
psn_config = {
    "tickers": ["SPY US Equity", "DIA US Equity", "QQQ US Equity"],
    "lags": [[1, 4, 5, 6, 7, 9, 10, 11, 12], [1, 2, 5, 6, 8, 9, 10], [2, 4, 5, 6, 7, 8, 9, 10, 11]],
    "learning_algorithm": ["sgd", "sgd", "sgd"],
    # "learning_rate": [0.4, 0.3, 0.3],
    "learning_rate": [0.001, 0.001, 0.001],
    "momentum": [0.5, 0.5, 0.4],
    "iteration_steps": [40000, 40000, 25000],
    "init_weights": ["random", "random", "random"],
    "hidden_nodes": [5, 6, 8]
}


# ----- Fonction pour créer des lags -----
def create_lag_features(df, target_col, lags=[1, 2, 3]):
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)
    df_lagged['target'] = df_lagged[target_col].shift(-1)  # prédire le return à t+1
    return df_lagged.dropna()

#define the model
class PSNModel(nn.Module):
    def __init__(self, input_size, hidden_nodes=6):
        super(PSNModel, self).__init__()
        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        
        # Sigma Layer: agrégation linéaire (similaire à une couche linéaire)
        self.sigma_layer = nn.Linear(input_size, hidden_nodes)
        
        # Psi Layer: non-linéarité (on peut aussi utiliser tanh ou une activation floue)
        self.psi_activation = nn.ReLU()
        
        # Output layer
        self.output_layer = nn.Linear(hidden_nodes, 1)
        
        self.init_weights()

    def init_weights(self):
        # Initialisation personnalisée (optionnelle)
        nn.init.normal_(self.sigma_layer.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0)
        if self.sigma_layer.bias is not None:
            nn.init.constant_(self.sigma_layer.bias, 0)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        sigma_out = self.sigma_layer(x)           # Σ layer: linéaire
        psi_out = self.psi_activation(sigma_out)  # Ψ layer: non-linéaire
        output = self.output_layer(psi_out)       # Sortie finale
        return output


# ----- Évaluation -----
def evaluate_performance(y_true, y_pred, dataset_name="Dataset"):
    # Ensure both arrays are 1-dimensional
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # print(f"y_true: {y_true}")
    # print(f"y_pred: {y_pred}")
    
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
    # print(np.isnan(X_train_scaled.numpy()).sum(), np.isinf(X_train_scaled.numpy()).sum())


    #Convert to torch tensors
    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
    X_out_scaled = torch.tensor(X_out_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    y_out_tensor = torch.tensor(y_out.values, dtype=torch.float32).view(-1, 1)
    # print(np.isnan(X_train_scaled.numpy()).sum(), np.isinf(X_train_scaled.numpy()).sum())



    # Instantiate the model with 7 input nodes
    model = PSNModel(input_size=X_train_scaled.shape[1], hidden_nodes=hidden_nodes)


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

for i in range(len(psn_config["tickers"])):
    ticker = psn_config["tickers"][i]
    lags_list = psn_config["lags"][i]
    learning_algorithm = psn_config["learning_algorithm"][i]
    learning_rate = psn_config["learning_rate"][i]
    momentum = psn_config["momentum"][i]
    iteration_steps = psn_config["iteration_steps"][i]
    init_weights = psn_config["init_weights"][i]
    hidden_nodes = psn_config["hidden_nodes"][i]

    print(f"------------------ Ticker: {ticker} ------------------")
    
    y_test, y_pred_test, y_out, y_pred_out = main(ticker, lags_list, learning_algorithm, learning_rate, momentum, iteration_steps, init_weights, hidden_nodes)

    result_dict[ticker] = y_pred_out

# exit()
# # ----- Test Statistique -----
# # Prédiction naïve = lag 1 (car on prédit t+1 à partir de t)
# y_pred_naive_out = df_out_lagged['lag_1'].values

# dm_stat = diebold_mariano_test(y_out.values, y_pred_out, y_pred_naive_out)
# print(f"📉 DM Test vs Naive Out-of-Sample: Statistic = {dm_stat:.3f}")

# pt_stat, pt_pval = pesaran_timmermann_test(y_out, np.array(y_pred_out).flatten())
# print(f"\n🧭 PT Test Out-of-Sample: Statistic = {pt_stat:.3f}, p-value = {pt_pval:.3f}")