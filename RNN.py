import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from clean_df_paper import df_total_set, df_training_set, df_test_set, df_out_sample_set
from metrics_forecast_error import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, theil_u_statistic
from test_statistic import diebold_mariano_test, pesaran_timmermann_test


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

# ----- Fonction pour créer des lags -----
def create_lag_features(df, target_col='return', n_lags=5):
    df_lagged = df.copy()
    for lag in [1,2,3,5,7,8,9,10,12]:
        df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)
    df_lagged['target'] = df_lagged[target_col].shift(-1)  # prédire le return à t+1
    return df_lagged.dropna()

# ----- Paramètres -----
n_lags = 3
df_training_set = df_training_set[df_training_set.columns[0]].to_frame()
df_test_set = df_test_set[df_test_set.columns[0]].to_frame()
df_out_sample_set = df_out_sample_set[df_out_sample_set.columns[0]].to_frame()

# ----- Création des features -----
df_train_lagged = create_lag_features(df_training_set, target_col=df_training_set.columns[0], n_lags=n_lags)
df_test_lagged = create_lag_features(df_test_set, target_col=df_test_set.columns[0], n_lags=n_lags)
df_out_lagged = create_lag_features(df_out_sample_set, target_col=df_out_sample_set.columns[0], n_lags=n_lags)

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

# Convert to torch tensors and add the sequence length dimension (seq_length=1)
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
X_out_scaled = torch.tensor(X_out_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
y_out_tensor = torch.tensor(y_out.values, dtype=torch.float32).view(-1, 1)

#define the model
class RNNModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=6, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        # Define a basic RNN; batch_first=True expects input Tensor shape (batch, seq, feature)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        # Initialize RNN weights with N(0,1) and biases with zeros
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # Initialize fully-connected layer weights and bias
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        # Initialize hidden state with zeros: shape (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        # Using the output from the last time step (assumes sequence length >= 1)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model with 7 input nodes
model = RNNModel()

# Define loss function and optimizer using Gradient Descent (SGD)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.005)

# Training loop with 40000 iterations
num_iterations = 40000
for step in range(num_iterations):
    optimizer.zero_grad()
    predictions = model(X_train_scaled)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions on test and out-of-sample sets
y_pred_test = model(X_test_scaled).detach().numpy()
y_pred_out = model(X_out_scaled).detach().numpy()

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

evaluate_performance(y_test, y_pred_test, "Test Set")
evaluate_performance(y_out, y_pred_out, "Out-of-Sample")

# ----- Test Statistique -----
# Prédiction naïve = lag 1 (car on prédit t+1 à partir de t)
y_pred_naive_out = df_out_lagged['lag_1'].values

dm_stat = diebold_mariano_test(y_out.values, y_pred_out, y_pred_naive_out)
print(f"📉 DM Test vs Naive Out-of-Sample: Statistic = {dm_stat:.3f}")

pt_stat, pt_pval = pesaran_timmermann_test(y_out, np.array(y_pred_out).flatten())
print(f"\n🧭 PT Test Out-of-Sample: Statistic = {pt_stat:.3f}, p-value = {pt_pval:.3f}")