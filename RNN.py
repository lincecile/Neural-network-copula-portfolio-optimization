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
        
        # Print which ETF we're predicting and which ones we're using as features
        print(f"\n{'='*50}")
        print(f"Model Configuration for {ticker.value}")
        print(f"{'='*50}")
        
        # Identify which ETFs are used as features
        etf_features = [col for col in df_train.columns if 'Equity' in col]
        lag_features = [col for col in df_train.columns if 'lag' in col]
        
        print("\nFeatures being used:")
        print(f"ETFs as input features: {etf_features}")  # Ces ETFs servent de variables explicatives
        print(f"Lagged values of {ticker.value}: {lag_features}")  # Les valeurs passées de l'ETF à prédire
        
        # Update input_nodes to match actual input size
        self.input_nodes = self.x_train_tensor.size(1)
        print(f"\nTotal input features: {self.input_nodes}")
        
        self.build_model()
    
    def build_model(self):
        """Build the RNN model"""
        # Print input size for debugging
        print(f"Input features size: {self.input_nodes}")
        
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
        print(f"Input shape: {x.shape}, Expected input nodes: {self.input_nodes}")
        
        # Ensure correct dimensions
        if x.size(-1) != self.input_nodes:
            print(f"Warning: Input size mismatch. Got {x.size(-1)}, expected {self.input_nodes}")
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
        """Train the RNN model"""
        # Ensure tensors have matching dimensions
        min_size = min(self.x_train_tensor.shape[0], self.y_train_tensor.shape[0])
        x_train = self.x_train_tensor[:min_size]
        y_train = self.y_train_tensor[:min_size]
        
        
        for epoch in range(self.iteration_steps):
            self.rnn.train()
            self.output_layer.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            preds = self.forward(x_train.float())
            loss = self.loss_fn(preds, y_train.float())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Print progress every 5000 steps
            if (epoch + 1) % 5000 == 0:
                print(f"Epoch [{epoch + 1}/{self.iteration_steps}], Loss: {loss.item():.4f}")
    
    def evaluate(self, x_tensor, y_true_np):
        """Evaluate the RNN model performance using one-step-ahead prediction"""
        self.rnn.eval()
        self.output_layer.eval()
        
        # Convert y_true_np to numpy array if it's a pandas Series
        if isinstance(y_true_np, pd.Series):
            y_true_np = y_true_np.to_numpy()
        
        # Ensure tensors have matching dimensions
        min_size = min(x_tensor.shape[0], len(y_true_np))
        x_tensor = x_tensor[:min_size]
        y_true_np = y_true_np[:min_size]
        
        with torch.no_grad():
            preds = self.forward(x_tensor.float()).numpy().flatten()  # Flatten predictions
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_np, preds)
        rmse = np.sqrt(mean_squared_error(y_true_np, preds))
        
        # Calculate MAPE, handling zero values
        mask = y_true_np != 0
        mape = np.mean(np.abs((y_true_np[mask] - preds[mask]) / y_true_np[mask])) * 100
        
        # Calculate Theil's U statistic
        theilu = np.sqrt(np.mean((y_true_np - preds) ** 2)) / (
            np.sqrt(np.mean(y_true_np ** 2)) + np.sqrt(np.mean(preds ** 2))
        )
        
        return mae, mape, rmse, theilu, preds

def main():
    """Process SPY, DIA, and QQQ ETFs sequentially"""
    
    # Dictionary to store results
    all_results = {}
    
    # Process each ETF
    etfs = [Ticker.spy, Ticker.dia, Ticker.qqq]
    
    for etf in etfs:
        print(f"\nProcessing {etf.value}...")
        
        # Create RNN model
        rnn_model = RNNForecaster(
            df_train=df_training_set_daily,
            df_test=df_test_set_daily,
            df_out=df_out_sample_set_daily,
            ticker=etf,
            hardcoded=True
        )
        
        # Train model
        rnn_model.train_model()
        
        # Store results for this ETF
        all_results[etf.value] = {}
        
        # Test set evaluation
        mae, mape, rmse, theilu, test_preds = rnn_model.evaluate(
            rnn_model.x_test_tensor,
            rnn_model.y_test
        )
        all_results[etf.value]['test'] = {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Theil U': theilu,
            'Predictions': test_preds
        }
        
        # Out-of-sample evaluation
        mae, mape, rmse, theilu, out_preds = rnn_model.evaluate(
            rnn_model.x_out_tensor,
            rnn_model.y_out
        )
        all_results[etf.value]['out_sample'] = {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Theil U': theilu,
            'Predictions': out_preds
        }
    
    # Display combined results
    print("\n" + "="*50)
    print("COMBINED RESULTS FOR ALL ETFs")
    print("="*50)
    
    for etf in etfs:
        print(f"\nResults for {etf.value}:")
        print("-"*30)
        
        print("Test Set Results:")
        results = all_results[etf.value]['test']
        print(f"MAE: {results['MAE']:.4f}")
        print(f"MAPE: {results['MAPE']:.2f}%")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"Theil's U: {results['Theil U']:.4f}")
        
        print("\nOut-of-Sample Results:")
        results = all_results[etf.value]['out_sample']
        print(f"MAE: {results['MAE']:.4f}")
        print(f"MAPE: {results['MAPE']:.2f}%")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"Theil's U: {results['Theil U']:.4f}")
    
    return all_results

if __name__ == "__main__":
    results = main()