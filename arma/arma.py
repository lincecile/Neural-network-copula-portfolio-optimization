import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
import warnings
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Import data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_df_all import df_test_set_daily, df_training_set_daily, df_out_sample_set_daily

class ARMAForecastModel:
    def __init__(self):
        # Configuration as per paper
        self.tickers = ["SPY US Equity", "DIA US Equity", "QQQ US Equity"]
        self.orders = [(8, 8), (10, 10), (7, 7)]  # Paper's ARMA orders
        self.predictions = {}
    
    def fit_arma_model(self, series, order):
        """Fit ARMA model une seule fois"""
        model = ARIMA(series, order=(order[0], 0, order[1]))
        fitted_model = model.fit()
        return fitted_model
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all metrics as per paper"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE calculation
        non_zero_mask = np.abs(y_true) > 1e-10
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        # Theil's U calculation
        rmse_pred = np.sqrt(np.mean(y_pred**2))
        rmse_true = np.sqrt(np.mean(y_true**2))
        theil_u = rmse / (rmse_pred + rmse_true) if (rmse_pred + rmse_true) > 0 else np.nan
        
        return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'Theil_U': theil_u}
    
    def one_step_ahead_forecast_like_rnn(self, model, start_data, test_data):
        """One-step-ahead comme RNN : modèle entraîné une fois, prédictions séquentielles"""
        predictions = []
        current_series = start_data.copy()
        
        for i in range(len(test_data)):
            # Get forecast and extract the value properly
            forecast = model.forecast(steps=1)
            pred = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
            predictions.append(pred)
            
            # Add actual value to history with proper index
            new_point = pd.Series([test_data.iloc[i]], index=[test_data.index[i]])
            current_series = pd.concat([current_series, new_point])
            
            # Update model with new data
            model = model.apply(current_series)
            
            # Progress reporting
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(test_data)}")
        
        return np.array(predictions)
    
    def fit_all_models(self):
        """Fit models for all tickers, like RNN approach"""
        for i, ticker in enumerate(self.tickers):
            order = self.orders[i]
            print(f"\n=== {ticker} ===")
            print(f"Using ARMA{order}")
            
            # Get data
            train = df_training_set_daily[ticker]
            test = df_test_set_daily[ticker]
            out_sample = df_out_sample_set_daily[ticker]
            
            # Entraîne UNE FOIS sur les données train+test
            combined_train = pd.concat([train, test])
            print("Training model once...")
            model = self.fit_arma_model(combined_train, order)
            
            # Out-of-sample predictions (comme ton RNN)
            print("Making out-of-sample predictions...")
            y_pred_out = self.one_step_ahead_forecast_like_rnn(model, combined_train, out_sample)
            y_true_out = out_sample.values
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true_out, y_pred_out)
            
            # Store results
            self.predictions[ticker] = {
                'predictions': y_pred_out,
                'actuals': y_true_out,
                'metrics': metrics
            }
            
            # Print results
            print("Out-of-sample metrics:")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  Theil's U: {metrics['Theil_U']:.4f}")
            
            # Show first predictions
            print("First 5 predictions:")
            for j in range(min(5, len(y_pred_out))):
                print(f"  Day {j+1}: Pred={y_pred_out[j]:.6f}, Actual={y_true_out[j]:.6f}")
    
    def save_results(self):
        """Save results for DCC"""
        # Save pickle
        with open('arma_results.pkl', 'wb') as f:
            pickle.dump(self.predictions, f)
        
        # Save CSV
        returns_df = {}
        for ticker in self.tickers:
            if ticker in self.predictions:
                pred = self.predictions[ticker]['predictions']
                index = df_out_sample_set_daily[ticker].index[-len(pred):]
                returns_df[ticker] = pd.Series(pred, index=index)
        
        df = pd.DataFrame(returns_df)
        #df.to_csv('arma_returns_out_sample.csv')
        
        # Save metrics
        metrics_df = pd.DataFrame({
            ticker: data['metrics'] 
            for ticker, data in self.predictions.items()
        }).T
        #metrics_df.to_csv('arma_metrics.csv')
        


if __name__ == "__main__":
    # Run ARMA forecasting
    arma_model = ARMAForecastModel()
    arma_model.fit_all_models()
    arma_model.save_results()
