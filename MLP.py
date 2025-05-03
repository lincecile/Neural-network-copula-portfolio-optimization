import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from clean_df_paper import df_total_set, df_training_set, df_test_set, df_out_sample_set
from metrics_forecast_error import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, theil_u_statistic

# ----- Fonction pour créer des lags -----
def create_lag_features(df, target_col='return', n_lags=5):
    df_lagged = df.copy()
    for lag in range(1, n_lags + 1):
        df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)
    df_lagged['target'] = df_lagged[target_col].shift(-1)  # prédire le return à t+1
    return df_lagged.dropna()

# ----- Paramètres -----
n_lags = 20
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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_out_scaled = scaler.transform(X_out)

# ----- Définition et entraînement du MLP -----
mlp = MLPRegressor(hidden_layer_sizes=(10,),
                   activation='relu',
                   solver='adam',
                   max_iter=500,
                   random_state=42)

mlp.fit(X_train_scaled, y_train)

# ----- Prédictions -----
y_pred_test = mlp.predict(X_test_scaled)
y_pred_out = mlp.predict(X_out_scaled)

# ----- Évaluation -----
print("📊 Test Set:")
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_test))
print("RMSE:", root_mean_squared_error(y_test, y_pred_test))
print("theil_u_statistic:", theil_u_statistic(y_test, y_pred_test))


print("\n📊 Out-of-sample:")
print("MAE:", mean_absolute_error(y_out, y_pred_out))
print("MAPE:", mean_absolute_percentage_error(y_out, y_pred_out))
print("RMSE:", root_mean_squared_error(y_out, y_pred_out))
print("theil_u_statistic:", theil_u_statistic(y_out, y_pred_out))
