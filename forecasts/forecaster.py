#%% imports

from abc import ABC, abstractmethod
import json
import typing as tp
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from ticker_dataclass import Ticker

from .nn_model_dataclass import NnModel

#%% functions

def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """
    Create lagged features for the target column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target column.
    lags (list): List of lag periods to create features for.
    
    Returns:
    pd.DataFrame: DataFrame with lagged features added.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Drop rows with NaN values created by shifting
    df.dropna(inplace=True)
    
    # Rename the target column to 'target'
    df.rename(columns={target_col: 'target'}, inplace=True)
    
    return df

#%% class

class NnForecaster(ABC, torch.nn.Module):
    """
    Abstract base class for forecasting models usin neural networks.
    """
    def __init__ (self, df_train : pd.DataFrame, df_test : pd.DataFrame, 
                  df_out : pd.DataFrame, ticker : Ticker, model : NnModel, 
                  hardcoded : bool = True):
        
        super(NnForecaster, self).__init__()
        
        self.ticker = ticker
        self.model = model
        self.hardcoded = hardcoded
        self.to_be_trained = False

        self.df_train = df_train
        self.df_test = df_test
        self.df_out = df_out
        
        param = self.__get_parameters()
        self.learning_algorithm = param["Learning algorithm"]
        self.learning_rate = param["Learning rate"]
        self.momentum = param["Momentum"]
        self.iteration_steps = param["Iteration steps"]
        self.initialisation_of_weights = param["Initialisation of weights"]
        self.input_nodes = param["Input nodes"]
        self.hidden_nodes = param["Hidden nodes"]
        self.output_node = param["Output node"]
        self.input_lags = param["Input lags"]
        
        # Train
        train_feat = create_lag_features(df_train, target_col=df_train.columns[0], lags=self.input_lags)
        self.x_train = train_feat.drop(columns=['target'])
        self.y_train = train_feat['target']

        # Test
        test_feat = create_lag_features(df_test, target_col=df_test.columns[0], lags=self.input_lags)
        self.x_test = test_feat.drop(columns=['target'])
        self.y_test = test_feat['target']

        # Out
        out_feat = create_lag_features(df_out, target_col=df_out.columns[0], lags=self.input_lags)
        self.x_out = out_feat.drop(columns=['target'])
        self.y_out = out_feat['target']

        #standardize
        self.scaler = StandardScaler()

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.x_out = self.scaler.transform(self.x_out)
        
        # Standardize target variables
        self.y_scaler = StandardScaler()

        self.y_train = self.y_scaler.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test = self.y_scaler.transform(self.y_test.values.reshape(-1, 1))
        self.y_out = self.y_scaler.transform(self.y_out.values.reshape(-1, 1))

        self.x_train_tensor = torch.tensor(self.x_train, dtype=torch.float64)
        self.x_test_tensor = torch.tensor(self.x_test, dtype=torch.float64)
        self.x_out_tensor = torch.tensor(self.x_out, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float64).view(-1, 1)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float64).view(-1, 1)
        self.y_out_tensor = torch.tensor(self.y_out, dtype=torch.float64).view(-1, 1)
        
        if self.to_be_trained:
            self.__optimize_param()
            pass
        
    def __get_parameters(self) -> tp.Dict[str, tp.Any]:
        """
        Get the parameters for the model.
        """
        if self.hardcoded:
            # Load parameters from a hardcoded file
            param = json.load(open("forecasts/param.json"))[self.ticker][self.model]

        else:
            # Starting guess for gradient descent (using your MLP hardcoded values)
            param = {
                "Learning algorithm": "Gradient descent",
                "Learning rate": 0.003,
                "Momentum": 0.004,
                "Iteration steps": 30000,
                "Initialisation of weights": "N(0,1)",
                "Input nodes": 7,
                "Hidden nodes": 6,                "Output node": 1,
                "Input lags": [1, 3, 5, 7, 8, 9, 12]
            }
            self.to_be_trained = True
                
        return param
        
    @abstractmethod
    def init_weights(self):
        # Initialize weights with N(0, 1) and biases with 0
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def train_model(self):
        """
        Train the model on the training set using the parameters specified in the constructor.
        
        This method implements a standard training loop for neural networks:
        1. Initialize optimizer with specified parameters from the base class
        2. Loop through the specified number of iterations:
            a. Perform forward pass
            b. Calculate loss
            c. Perform backpropagation
            d. Update weights
            e. Optionally print progress
            
        Returns:
            List of loss values throughout training for monitoring purposes.
        """
        
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
    
    @abstractmethod
    def evaluate_model(self):
        """
        Evaluate the model on the test set.
        Should return mae, mape, rmse, theilu, preds
        """
        pass
    
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