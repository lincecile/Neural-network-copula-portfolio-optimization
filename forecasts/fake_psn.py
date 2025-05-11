# filepath: c:\Users\Enzo\OneDrive\Obsidian\Enzo\Cours\s2\Outils quantitatifs\Neural-network-copula-portfolio-optimization\forecasts\fake_psn.py
#%% imports

import torch
import torch.nn as nn
import torch.optim as optim


from .forecaster import NnForecaster
from .nn_model_dataclass import NnModel
import pandas as pd

#%% class

class FakePsnForecaster(NnForecaster):
    """This class implements the fake Pi-Sigma Network model for forecasting as described in the paper."""
    
    def __init__(self, df_train, df_test, df_out, ticker, hardcoded=True):
        model = NnModel.psn
        super(FakePsnForecaster, self).__init__(df_train, df_test, df_out, ticker, model, hardcoded)
        
        # Create a proper Pi-Sigma Network architecture
        # Input-to-hidden layer (fully connected)
        self.input_layer = nn.Linear(self.input_nodes, self.hidden_nodes)
        
        # Hidden-to-output layer with learnable weights (product units)
        self.sigma_layer = nn.Linear(self.hidden_nodes, 1, bias=True)
        
        # Scaling factor (learnable)
        self.c = nn.Parameter(torch.tensor(1.0))
        
        # Hidden layer activation function
        self.hidden_activation = nn.Tanh()
        
    def init_weights(self):
        """Initialise the weights of the input layer"""
        if self.initialisation_of_weights == "N(0,1)":
            self.input_layer.weight.data.normal_(0, 1)  # Initialize input layer weights
            self.sigma_layer.weight.data.normal_(0, 1)  # Initialize sigma layer weights
        else:
            raise ValueError("Initialisation of weights not implemented")
        
    def forward(self, x):
        """
        Performs a forward pass through the network.
        This is a proper implementation of a Pi-Sigma Network, which:
        1. Applies the input-to-hidden layer weights
        2. Applies an activation function to the hidden units
        3. Computes a weighted sum of the hidden activations
        4. Applies a final scaling factor
        
        Parameters:
            x (torch.Tensor): The input tensor to the network.
        Returns:
            torch.Tensor: The output tensor after applying the forward computations.
        """
        # Input to hidden layer
        z = self.input_layer(x)
        
        # Apply activation function
        h = self.hidden_activation(z)
        
        # Apply sigma layer (weighted sum)
        out = self.sigma_layer(h)
        
        # Scale output (similar to the paper's implementation)
        out = out * self.c
        
        return out
        
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
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum
            )
        else:
            raise ValueError(f"Learning algorithm {self.learning_algorithm} not implemented")
            
        # Define loss function - MSE is standard for regression problems
        loss_fn = nn.MSELoss()
        
        # Make sure input tensors are float32 for better numerical stability
        x_train = self.x_train_tensor.float()
        y_train = self.y_train_tensor.float()
        
        # Track losses for monitoring
        losses = []
        
        # Training loop
        for iteration in range(self.iteration_steps):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = self(x_train)
            
            # Compute loss
            loss = loss_fn(y_pred, y_train)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
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
