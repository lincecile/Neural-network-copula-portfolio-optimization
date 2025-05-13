# forecasts/MLP.py

import torch
import torch.nn as nn
import torch.optim as optim
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
        
        # Setup optimizer (added for consistency with other implementations)
        self.optimizer = optim.SGD(
            list(self.fc1.parameters()) + list(self.fc2.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self.loss_fn = nn.MSELoss()

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

    def train_model(self):
        """
        Train the model on the training set using the parameters specified in the constructor.
        
        This method implements a standard training loop for neural networks:
        1. Set the model to training mode
        2. Initialize optimizer with specified parameters from the base class
        3. Loop through the specified number of iterations:
            a. Perform forward pass
            b. Calculate loss
            c. Perform backpropagation
            d. Update weights
            e. Optionally print progress
            
        Returns:
            List of loss values throughout training for monitoring purposes.
        """
        # Set model to training mode
        self.train()
        
        # Initialize optimizer according to learning algorithm
        if self.learning_algorithm == "Gradient descent":
            # Use the optimizer created during initialization if it exists
            if not hasattr(self, 'optimizer'):
                self.optimizer = optim.SGD(
                    self.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum
                )
        else:
            raise ValueError(f"Learning algorithm {self.learning_algorithm} not implemented")
            
        # Define loss function if not already defined
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = nn.MSELoss()
        
        # Make sure input tensors are float32 for better numerical stability
        x_train = self.x_train_tensor.float()
        y_train = self.y_train_tensor.float()
        
        # Track losses for monitoring
        losses = []
        
        # Training loop
        for iteration in range(self.iteration_steps):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self(x_train)
            
            # Compute loss
            loss = self.loss_fn(y_pred, y_train)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Store loss
            losses.append(loss.item())
            
            # Print progress every 1000 iterations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}/{self.iteration_steps}, Loss: {loss.item():.6f}")
        
        print(f"Training completed. Final loss: {losses[-1]:.6f}")
        return losses

    def evaluate_model(self):
        """
        Evaluate the model on the test set.
        
        Returns:
            tuple: (mae, mape, rmse, theilu, predictions)
                - mae: Mean Absolute Error
                - mape: Mean Absolute Percentage Error (using Symmetric MAPE)
                - rmse: Root Mean Squared Error
                - theilu: Theil's U statistic
                - predictions: Model predictions on test data
        """
        # Set model to evaluation mode
        self.eval()
        
        # Make sure input tensors are float32
        x_test = self.x_test_tensor.float()
        
        # Get predictions
        with torch.no_grad():
            y_pred = self(x_test)
        
        # Convert to numpy for metric calculation
        y_pred_np = y_pred.numpy()
        y_true_np = self.y_test_tensor.numpy()
        
        # Inverse transform predictions and actual values
        y_pred_original = self.y_scaler.inverse_transform(y_pred_np)
        y_true_original = self.y_scaler.inverse_transform(y_true_np)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        
        # Calculate Symmetric MAPE (sMAPE) - better for financial time series with values near zero
        # Updated to be consistent with other implementations
        smape = np.mean(200.0 * np.abs(y_pred_original - y_true_original) / (np.abs(y_pred_original) + np.abs(y_true_original) + 1e-8))
        mape = smape  # Using sMAPE instead of traditional MAPE
        
        # Calculate Theil's U
        numerator = np.sqrt(np.mean((y_pred_original - y_true_original) ** 2))
        denominator = np.sqrt(np.mean(y_true_original ** 2)) + np.sqrt(np.mean(y_pred_original ** 2))
        theilu = numerator / denominator if denominator != 0 else np.inf
        
        print(f"Test metrics: MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}, Theil's U: {theilu:.4f}")
        
        return mae, mape, rmse, theilu, y_pred_original
        
    def predict(self, x=None):
        """
        Predict the output for the given input or out-of-sample set.
        
        Args:
            x (torch.Tensor, optional): Input data. If None, uses the out-of-sample data.
            
        Returns:
            numpy.ndarray: Predictions in original scale
        """
        # Set model to evaluation mode
        self.eval()
        
        # Use out-of-sample data if no input is provided
        if x is None:
            x = self.x_out_tensor.float()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            y_pred = self(x)
        
        # Inverse transform predictions to original scale
        y_pred_np = y_pred.numpy()
        y_pred_original = self.y_scaler.inverse_transform(y_pred_np)
        
        return y_pred_original