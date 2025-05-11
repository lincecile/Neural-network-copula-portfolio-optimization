# forecasts/MLP.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecasts.forecaster import NnForecaster
from forecasts.nn_model_dataclass import NnModel
from ticker_dataclass import Ticker


class MlpForecaster(NnForecaster):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 df_out: pd.DataFrame, ticker: Ticker, hardcoded: bool = True):
        super().__init__(df_train, df_test, df_out, ticker, NnModel.mlp, hardcoded)
        self.build_model()

    def build_model(self):
        self.fc1 = nn.Linear(self.input_nodes, self.hidden_nodes)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_nodes, self.output_node)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

    def evaluate_model(self):
        self.eval()
        with torch.no_grad():
            preds = self.forward(self.x_test_tensor.float()).numpy()

        y_true_np = self.y_scaler.inverse_transform(self.y_test_tensor.numpy())
        preds_original = self.y_scaler.inverse_transform(preds)

        mae = mean_absolute_error(y_true_np, preds_original)
        rmse = np.sqrt(mean_squared_error(y_true_np, preds_original))
        mask = y_true_np.flatten() != 0
        mape = np.mean(np.abs((y_true_np.flatten()[mask] - preds_original.flatten()[mask]) / y_true_np.flatten()[mask])) * 100
        theilu = np.sqrt(np.mean((y_true_np - preds_original) ** 2)) / (
            np.sqrt(np.mean(y_true_np ** 2)) + np.sqrt(np.mean(preds_original ** 2))
        )
        return mae, mape, rmse, theilu, preds_original


        
