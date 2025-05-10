# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecasts.forecaster import NnForecaster, create_lag_features
from ticker_dataclass import Ticker
from forecasts.nn_model_dataclass import NnModel
from clean_df_paper import df_training_set_daily, df_test_set_daily, df_out_sample_set_daily

# %% MLP Forecaster Class
class MLP(NnForecaster):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 df_out: pd.DataFrame, ticker: str, hardcoded: bool = True):
        super().__init__(df_train, df_test, df_out, ticker, NnModel.mlp, hardcoded)
        self.build_model()

    def build_model(self):
        self.input_layer = nn.Linear(self.input_nodes, self.hidden_nodes)
        self.output_layer = nn.Linear(self.hidden_nodes, self.output_node)
        self.init_weights()
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            list(self.input_layer.parameters()) + list(self.output_layer.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self.loss_fn = nn.MSELoss()

    def init_weights(self):
        # Normal distribution initialization as per the parameters
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        x = self.output_layer(x)
        return x

    def train_model(self):
        for _ in range(self.iteration_steps):
            self.input_layer.train()
            self.output_layer.train()
            self.optimizer.zero_grad()
            preds = self.forward(self.x_train_tensor.float())
            loss = self.loss_fn(preds, self.y_train_tensor.float())
            loss.backward()
            self.optimizer.step()

    def evaluate(self, x_tensor, y_true_np):
        self.input_layer.eval()
        self.output_layer.eval()
        with torch.no_grad():
            preds = self.forward(x_tensor.float()).numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_np, preds)
        rmse = np.sqrt(mean_squared_error(y_true_np, preds))
        
        # Calculate MAPE, handling zero values
        mask = y_true_np.flatten() != 0
        mape = np.mean(np.abs((y_true_np.flatten()[mask] - preds.flatten()[mask]) / y_true_np.flatten()[mask])) * 100
        
        # Calculate Theil's U statistic
        theilu = np.sqrt(np.mean((y_true_np - preds) ** 2)) / (
            np.sqrt(np.mean(y_true_np ** 2)) + np.sqrt(np.mean(preds ** 2))
        )
        
        return mae, mape, rmse, theilu, preds

    def __optimize_param(self):
        """
        Optional hyperparameter optimization
        """
        pass  # Implement hyperparameter optimization if needed

