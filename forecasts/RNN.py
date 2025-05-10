# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecaster import NnForecaster, create_lag_features
from ticker_dataclass import Ticker
from .nn_model_dataclass import NnModel

# %% RNN Forecaster Class
class RnnForecaster(NnForecaster):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 df_out: pd.DataFrame, ticker: Ticker, hardcoded: bool = True):
        super().__init__(df_train, df_test, df_out, ticker, NnModel.rnn, hardcoded)
        self.build_model()

    def build_model(self):
        input_size = self.input_nodes
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=self.hidden_nodes,
                          batch_first=True,
                          nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_nodes, self.output_node)
        self.init_weights()

        self.optimizer = optim.SGD(
            list(self.rnn.parameters()) + list(self.fc.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self.loss_fn = nn.MSELoss()

    def init_weights(self):
        for param in self.rnn.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0.0, std=1.0)
            else:
                nn.init.constant_(param, 0.0)
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(1))  # Add sequence dimension
        return self.fc(out[:, -1, :])

    def train_model(self):
        for _ in range(self.iteration_steps):
            self.rnn.train()
            self.fc.train()
            self.optimizer.zero_grad()
            preds = self.forward(self.x_train_tensor.float())
            loss = self.loss_fn(preds, self.y_train_tensor.float())
            loss.backward()
            self.optimizer.step()

    def evaluate(self, x_tensor, y_true_np):
        self.rnn.eval()
        self.fc.eval()
        with torch.no_grad():
            preds = self.forward(x_tensor.float()).numpy()
        mae = mean_absolute_error(y_true_np, preds)
        rmse = np.sqrt(mean_squared_error(y_true_np, preds))
        mask = y_true_np.flatten() != 0
        mape = np.mean(np.abs((y_true_np.flatten()[mask] - preds.flatten()[mask]) / y_true_np.flatten()[mask])) * 100
        theilu = np.sqrt(np.mean((y_true_np - preds) ** 2)) / (
            np.sqrt(np.mean(y_true_np ** 2)) + np.sqrt(np.mean(preds ** 2))
        )
        return mae, mape, rmse, theilu, preds

    def __optimize_param(self):
        pass  # Optional
