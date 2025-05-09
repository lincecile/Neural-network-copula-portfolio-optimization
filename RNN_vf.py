# Packages pour le modèle RNN
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from clean_df_paper import df_test_set_daily, df_training_set_daily, df_out_sample_set_daily 

# Préparation des séquences
def prepare_data(series, seq_length):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i + seq_length])
        y.append(scaled[i + seq_length])
    return np.array(X), np.array(y), scaler

# Modèle RNN simple
def train_rnn_model(X_train, y_train, X_val, y_val, seq_length):
    model = Sequential()
    model.add(SimpleRNN(32, activation='tanh', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=8, callbacks=[es], verbose=0)
    return model

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true != 0
    if np.any(mask):
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        return np.nan  # ou 0

# THEIL-U
def theil_u(y_true, y_pred):
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return num / denom

seq_length = 10
assets = ['SPY US Equity', 'DIA US Equity', 'QQQ US Equity']
results = {}

for asset in assets:
    print(f"\n🔧 Entraînement modèle pour : {asset}")

    train = df_training_set_daily[asset]
    test = df_test_set_daily[asset]
    out = df_out_sample_set_daily[asset]

    X_train, y_train, scaler_train = prepare_data(train, seq_length)
    X_test, y_test, _ = prepare_data(test, seq_length)
    X_out, y_out, scaler_out = prepare_data(out, seq_length)

    model = train_rnn_model(X_train, y_train, X_test, y_test, seq_length)

    y_pred_out_scaled = model.predict(X_out)
    y_pred_out = scaler_out.inverse_transform(y_pred_out_scaled)
    y_true_out = scaler_out.inverse_transform(y_out.reshape(-1, 1))

    mae = mean_absolute_error(y_true_out, y_pred_out)
    rmse = np.sqrt(mean_squared_error(y_true_out, y_pred_out))

    results[asset] = {
        "MAE": mae,
        "RMSE": rmse,
        "model": model,
        "scaler": scaler_out,
        "X_out": X_out,
        "y_out": y_out
    }

    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")

summary = []

for asset in assets:
    print(f"\n📊 Résultats pour : {asset}")

    model = results[asset]['model']
    scaler = results[asset]['scaler']
    X_out = results[asset]['X_out']
    y_out = results[asset]['y_out']

    y_pred_scaled = model.predict(X_out)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_out.reshape(-1, 1))

    # Visualisation
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Réel', linewidth=1.5)
    plt.plot(y_pred, label='Prédiction', linestyle='--')
    plt.title(f"{asset} - Prédictions vs Réalité (Out-of-sample)")
    plt.xlabel("Jours")
    plt.ylabel("Rendements")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Statistiques
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    theilu = theil_u(y_true, y_pred)

    summary.append({
        "ETF": asset.replace(" US Equity", ""),
        "MAE": mae,
        "MAPE (%)": mape,
        "RMSE": rmse,
        "THEIL-U": theilu
    })

# Résumé dans un tableau
summary_df = pd.DataFrame(summary).set_index("ETF")
from IPython.display import display
display(summary_df)
