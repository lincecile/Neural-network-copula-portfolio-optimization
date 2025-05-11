# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecasts.forecaster import NnForecaster
from ticker_dataclass import Ticker
from forecasts.nn_model_dataclass import NnModel
from clean_df_paper import df_training_set_daily, df_test_set_daily, df_out_sample_set_daily
import os
import pickle

# %% RNN Forecaster Class
class RNNForecaster(NnForecaster):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, 
                 df_out: pd.DataFrame, ticker: str, hardcoded: bool = True):
        # Call parent constructor
        super().__init__(df_train=df_train, 
                        df_test=df_test, 
                        df_out=df_out, 
                        ticker=ticker, 
                        model=NnModel.rnn, 
                        hardcoded=hardcoded)
        
        # Update input_nodes to match actual input size
        self.input_nodes = self.x_train_tensor.size(1)
        
        self.build_model()
    
    def build_model(self):
        """Build the RNN model"""
        # Print input size for debugging
        
        # RNN with custom parameters
        self.rnn = nn.RNN(
            input_size=self.input_nodes,
            hidden_size=self.hidden_nodes,
            num_layers=1,
            batch_first=True,
            nonlinearity='relu'
        )
        
        self.output_layer = nn.Linear(self.hidden_nodes, self.output_node)
        self.init_weights()
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            list(self.rnn.parameters()) + list(self.output_layer.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self.loss_fn = nn.MSELoss()
    
    def init_weights(self):
        """Initialize weights with N(0, 1) and biases with 0"""
        # RNN weights - according to the N(0,1) parameter
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Output layer weights
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x):
        """Forward pass for one-step ahead prediction"""
        batch_size = x.size(0)
        
        # Debug: verify dimensionality
        
        # Ensure correct dimensions
        if x.size(-1) != self.input_nodes:
            if x.size(-1) > self.input_nodes:
                x = x[..., :self.input_nodes]
            else:
                raise ValueError(f"Input has too few features: got {x.size(-1)}, need {self.input_nodes}")
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # Reshape for RNN
        x_reshaped = x.view(batch_size, 1, self.input_nodes)
        
        # Initialize hidden state
        h_0 = torch.zeros(1, batch_size, self.hidden_nodes, dtype=torch.float32)
          # Forward pass
        out, _ = self.rnn(x_reshaped, h_0)
        output = out[:, -1, :]
        prediction = self.output_layer(output)

        return prediction
        
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
        self.train_mode = True
        
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
                - mape: Mean Absolute Percentage Error
                - rmse: Root Mean Squared Error
                - theilu: Theil's U statistic
                - predictions: Model predictions on test data
        """
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
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
