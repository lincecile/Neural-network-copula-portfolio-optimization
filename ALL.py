import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import t, norm
from arch import arch_model
from datetime import timedelta

# Import your existing modules - these would need to be in the same directory
from DCC import DCCModel
from MLP import MLP, create_lag_features
from Copula import SkewedTCopula



class IntegratedModel:
    def __init__(self, tickers, mlp_configs, training_data_daily, test_data_daily, out_sample_data_daily):
        """
        Integrated model for ETF return prediction
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        mlp_configs : dict
            Configuration for MLP models
        training_data_daily, test_data_daily, out_sample_data_daily : DataFrame
            Daily return data for training, testing, and out-of-sample evaluation
        """
        self.tickers = tickers
        self.mlp_configs = mlp_configs
        self.training_data_daily = training_data_daily
        self.test_data_daily = test_data_daily
        self.out_sample_data_daily = out_sample_data_daily
        
        # Models and results storage
        self.mlp_models = {}
        self.daily_predictions = {}
        self.weekly_predictions = {}
        self.dcc_model = None
        self.skewed_t_copula = None
        self.dynamic_correlations = None
        
    def _convert_daily_to_weekly(self, daily_returns):
        # Resample to weekly frequency
        weekly_returns = (1+daily_returns.resample('W')).cumprod() - 1
        return weekly_returns
    
    def _train_mlp_model(self, ticker, df_training_set_daily, lags, learning_rate, momentum, iteration_steps, hidden_nodes):
        # Create lagged features
        df_ticker = df_training_set_daily[ticker].to_frame()
        df_lagged = create_lag_features(df_ticker, target_col=ticker, lags=lags)
        
        # Split features and target
        X = df_lagged.drop(columns=['target', ticker])
        y = df_lagged['target']
        X_cols = X.columns.tolist()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        # Initialize and train MLP model
        model = MLP(input_size=X_scaled.shape[1], hidden_nodes=hidden_nodes)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
        # Training loop
        for step in range(iteration_steps):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model, scaler, X_cols
    
    def _predict_with_mlp(self, model, scaler, test_data_daily, ticker, feature_cols, lags):

        df_ticker = test_data_daily[ticker].to_frame()
        df_lagged = create_lag_features(df_ticker, target_col=ticker, lags=lags)
        
        # Remove rows with NaN (due to lagging)
        df_lagged = df_lagged.dropna()
        
        # Extract features
        X = df_lagged[feature_cols]
        
        # Scale features
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Generate predictions
        with torch.no_grad():
            y_pred = model(X_tensor).numpy().flatten()
        
        return pd.Series(y_pred, index=df_lagged.index)
    
    def train_models(self):
        """Train MLP models for all tickers"""
        for i, ticker in enumerate(self.tickers):
            print(f"\n--- Training MLP model for {ticker} ---")
            
            # Extract configuration for this ticker
            # Train model
            model, scaler, feature_cols = self._train_mlp_model(
                ticker, 
                self.training_data_daily, 
                self.mlp_configs["lags"][i],
                self.mlp_configs["learning_rate"][i],
                self.mlp_configs["momentum"][i],
                self.mlp_configs["iteration_steps"][i],
                self.mlp_configs["hidden_nodes"][i]
            )
            
            # Store model
            self.mlp_models[ticker] = {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'lags': self.mlp_configs["lags"][i]
            }
            
            # Generate predictions
            test_pred = self._predict_with_mlp(
                model, scaler, self.test_data_daily, ticker, feature_cols, self.mlp_configs["lags"][i]
            )
            out_pred = self._predict_with_mlp(
                model, scaler, self.out_sample_data_daily, ticker, feature_cols, self.mlp_configs["lags"][i]
            )
            
            # Store daily predictions
            self.daily_predictions[ticker] = pd.concat([test_pred, out_pred])
            
        # Combine all predictions into a single DataFrame
        self.daily_predictions_df = pd.DataFrame({
            ticker: self.daily_predictions[ticker] for ticker in self.tickers
        })
        print(self.daily_predictions_df)
        # Convert daily predictions to weekly
        self.weekly_predictions_df = self._convert_daily_to_weekly(self.daily_predictions_df)
        
        print("\n--- MLP models trained and predictions generated ---")
        return self.weekly_predictions_df
    
    def train_dcc_model(self):
        """Train DCC model on weekly predictions"""
        print("\n--- Training DCC model ---")
        
        # Initialize and fit DCC model
        self.dcc_model = DCCModel(self.weekly_predictions_df.dropna())
        self.dcc_model.fit_univariate_garch(p=1, q=1)
        self.dcc_model.fit_dcc()
        
        # Extract dynamic correlations
        self.dynamic_correlations = self.dcc_model.get_dynamic_correlations()
        
        print(f"DCC model fitted with parameters: a = {self.dcc_model.dcc_params[0]:.4f}, b = {self.dcc_model.dcc_params[1]:.4f}")
        return self.dynamic_correlations
    
    def fit_skewed_t_copula(self, window_size=52):
        """
        Fit skewed-t copula using rolling window approach
        
        Parameters:
        -----------
        window_size : int
            Rolling window size in weeks (52 = 1 year)
        """
        print("\n--- Fitting skewed-t copula ---")
        
        # Get standardized residuals from DCC model
        std_residuals = self.dcc_model.std_residuals
        
        # Number of tickers
        n_tickers = len(self.tickers)
        
        # Initialize empty DataFrame for time-varying parameters
        dates = std_residuals.index[window_size-1:]
        n_periods = len(dates)
        
        df_params = pd.DataFrame(index=dates, columns=['df'] + [f'skew_{ticker}' for ticker in self.tickers])
        correlation_matrices = {}
        
        # Use rolling window approach
        for t in range(n_periods):
            # Extract window of residuals
            window_residuals = std_residuals.iloc[t:t+window_size].values
            
            # Transform to uniform margins using empirical CDF
            U = np.zeros_like(window_residuals)
            for j in range(n_tickers):
                sorted_residuals = np.sort(window_residuals[:, j])
                ecdf = np.arange(1, len(sorted_residuals) + 1) / (len(sorted_residuals) + 1)
                U[:, j] = np.interp(window_residuals[:, j], sorted_residuals, ecdf)
            
            # Extract correlation matrix for this period from DCC model
            R_t = self.dcc_model.R_t[t + window_size - 1]
            
            # Fit skewed-t copula
            copula = SkewedTCopula(n_dim=n_tickers)
            copula.fit(U, R=R_t)
            
            # Store parameters
            df_params.loc[dates[t], 'df'] = copula.df
            for j, ticker in enumerate(self.tickers):
                df_params.loc[dates[t], f'skew_{ticker}'] = copula.skew_params[j]
            
            # Store correlation matrix
            correlation_matrices[dates[t]] = R_t
            
            # Store most recent copula model
            if t == n_periods - 1:
                self.skewed_t_copula = copula
        
        self.copula_params = df_params
        self.correlation_matrices = correlation_matrices
        
        print(f"Skewed-t copula fitted with rolling window of {window_size} weeks")
        return df_params
    
    def generate_scenario_forecasts(self, date, n_scenarios=1000):
        """
        Generate scenario forecasts for a specific date
        
        Parameters:
        -----------
        date : datetime
            Date for which to generate scenarios
        n_scenarios : int
            Number of scenarios to generate
        
        Returns:
        --------
        scenarios : DataFrame
            Scenario returns for each ticker
        """
        if date not in self.copula_params.index:
            raise ValueError(f"No copula parameters available for date {date}")
        
        # Get copula parameters for this date
        df = self.copula_params.loc[date, 'df']
        skew_params = [self.copula_params.loc[date, f'skew_{ticker}'] for ticker in self.tickers]
        R = self.correlation_matrices[date]
        
        # Create copula with these parameters
        copula = SkewedTCopula(n_dim=len(self.tickers))
        copula.df = df
        copula.skew_params = skew_params
        copula.corr_matrix = R
        
        # Generate samples from copula
        U = copula.sample(n_samples=n_scenarios, R=R)
        
        # Transform to returns using empirical inverse CDF
        # Get window of weekly returns
        weekly_returns = self.weekly_predictions_df
        end_idx = weekly_returns.index.get_loc(date)
        start_idx = max(0, end_idx - 52)  # Use up to 1 year of data
        window_returns = weekly_returns.iloc[start_idx:end_idx+1]
        
        # Transform uniform samples to returns
        scenarios = np.zeros_like(U)
        for j, ticker in enumerate(self.tickers):
            returns = window_returns[ticker].dropna().values
            sorted_returns = np.sort(returns)
            scenarios[:, j] = np.interp(
                U[:, j], 
                np.linspace(0, 1, len(sorted_returns)), 
                sorted_returns
            )
        
        return pd.DataFrame(scenarios, columns=self.tickers)
    
    def run_full_model(self, window_size=52):
        """
        Run the complete integrated model
        
        Parameters:
        -----------
        window_size : int
            Rolling window size in weeks
        """
        # Step 1: Train MLP models and generate predictions
        self.train_mlp_models()
        
        print("\n--- MLP models trained and predictions generated ---")

        # Step 2: Train DCC model on weekly predictions
        self.train_dcc_model()

        print("\n--- DCC model trained ---")
        
        # Step 3: Fit skewed-t copula with rolling window
        self.fit_skewed_t_copula(window_size=window_size)
        
        return {
            'weekly_predictions': self.weekly_predictions_df,
            'dynamic_correlations': self.dynamic_correlations,
            'copula_params': self.copula_params
        }
    
    def plot_results(self, figsize=(15, 15)):
        """Plot key results from the model"""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Weekly Return Predictions
        for ticker in self.tickers:
            axes[0].plot(self.weekly_predictions_df.index, 
                         self.weekly_predictions_df[ticker], 
                         label=ticker)
        
        axes[0].set_title('Weekly Return Predictions')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Dynamic Correlations
        for col in self.dynamic_correlations.columns:
            axes[1].plot(self.dynamic_correlations.index, 
                         self.dynamic_correlations[col], 
                         label=col)
        
        axes[1].set_title('Dynamic Conditional Correlations')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Correlation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Copula Parameters
        df_line = axes[2].plot(self.copula_params.index, 
                               self.copula_params['df'], 
                               label='Degrees of Freedom', 
                               color='black', 
                               linewidth=2)
        
        # Create a second y-axis for skewness parameters
        ax2 = axes[2].twinx()
        skew_lines = []
        for ticker in self.tickers:
            line = ax2.plot(self.copula_params.index, 
                            self.copula_params[f'skew_{ticker}'], 
                            label=f'Skewness {ticker}', 
                            linestyle='--')
            skew_lines.append(line[0])
        
        axes[2].set_title('Skewed-t Copula Parameters')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Degrees of Freedom')
        ax2.set_ylabel('Skewness')
        
        # Combine legends from both y-axes
        lines = df_line + skew_lines
        labels = [line.get_label() for line in lines]
        axes[2].legend(lines, labels)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example usage:
if __name__ == "__main__":
    # This is where you'd import your data
    from clean_df_paper import df_training_set_daily, df_test_set_daily, df_out_sample_set_daily
    
    # MLP configuration
    mlp_config = {
        "tickers": ["SPY US Equity", "DIA US Equity", "QQQ US Equity"],
        "lags": [[1, 3, 5, 6, 8, 9, 12], [2, 4, 5, 7, 9, 10, 11], [1, 2, 3, 5, 6, 8, 10, 11, 12]],
        "learning_rate": [0.003, 0.002, 0.003],
        "momentum": [0.004, 0.005, 0.005],
        "iteration_steps": [30000, 45000, 30000],
        "hidden_nodes": [6, 9, 8]
    }
    
    # Create integrated model
    model = IntegratedModel(
        tickers=mlp_config["tickers"],
        mlp_configs=mlp_config,
        training_data_daily=df_training_set_daily,
        test_data_daily=df_test_set_daily,
        out_sample_data_daily=df_out_sample_set_daily
    )
    
    # Run the full model
    results = model.run_full_model(window_size=52)  # 52 weeks = 1 year
    
    # Generate scenario forecasts for the last date
    last_date = model.copula_params.index[-1]
    scenarios = model.generate_scenario_forecasts(last_date, n_scenarios=1000)
    
    # Plot correlation between first two assets
    plt.figure(figsize=(8, 8))
    plt.scatter(scenarios.iloc[:, 0], scenarios.iloc[:, 1], alpha=0.5)
    plt.title(f'Scenario Forecasts: {model.tickers[0]} vs {model.tickers[1]}')
    plt.xlabel(model.tickers[0])
    plt.ylabel(model.tickers[1])
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot results
    model.plot_results()
    plt.show()