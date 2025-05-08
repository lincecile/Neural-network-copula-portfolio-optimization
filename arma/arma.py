import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
import warnings
import time
from datetime import timedelta

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add parent directory to path to import clean_df_paper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_df_paper import df_test_set_daily, df_training_set_daily

def arma_model(series, order):
    """
    Create an ARMA model from the given Series
    """
    model = ARIMA(series, order=(order[0], 0, order[1]))
    fitted_model = model.fit()
    return fitted_model

def calculate_mape_corrected(y_true, y_pred):
    """
    Calculate MAPE in a way more consistent with the paper.
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate absolute percentage errors
    non_zero_indices = np.abs(y_true) > 1e-10
    if not np.any(non_zero_indices):
        return float('inf')
    
    # Filter out zero or near-zero values
    filtered_y_true = y_true[non_zero_indices]
    filtered_y_pred = y_pred[non_zero_indices]
    
    # Calculate percentage errors
    percentage_errors = np.abs((filtered_y_true - filtered_y_pred) / filtered_y_true) * 100
    
    # Return mean
    return np.mean(percentage_errors)

def calculate_theils_u(y_true, y_pred):
    """
    Calculate Theil's U as per the paper
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    
    # Calculate denominators for Theil's U
    rmse_pred = np.sqrt(np.mean(np.square(y_pred)))
    rmse_true = np.sqrt(np.mean(np.square(y_true)))
    
    # Calculate Theil's U
    denominator = rmse_pred + rmse_true
    if denominator > 0:
        return rmse / denominator
    else:
        return np.nan

def one_step_ahead_forecast(train_data, test_data, order, update_frequency=5):
    """
    Perform one-step-ahead forecasting using an ARMA model.
    
    Parameters:
    -----------
    train_data : pandas.Series
        The training data series
    test_data : pandas.Series
        The test data series
    order : tuple
        The order of the ARMA model (p, q)
    update_frequency : int
        How often to re-estimate the model (e.g., 5 means re-estimate every 5 days)
        
    Returns:
    --------
    numpy.ndarray
        The one-step-ahead predictions
    """
    predictions = []
    test_len = len(test_data)
    
    # Initialize first model with training data
    history = train_data.copy()
    model = arma_model(history, order=order)
    
    # Set up progress display
    print(f"    Starting one-step-ahead forecasting for {test_len} steps...")
    start_time = time.time()
    
    for i in range(test_len):
        # Predict one step ahead
        forecast = model.forecast(steps=1)
        next_pred = forecast.iloc[0]
        predictions.append(next_pred)
        
        # Update history with actual value
        history_idx = train_data.index.tolist() + test_data.index.tolist()[:i+1]
        history_values = train_data.values.tolist() + test_data.values.tolist()[:i+1]
        history = pd.Series(history_values, index=history_idx)
        
        # Re-estimate model every update_frequency steps or on last step
        if (i + 1) % update_frequency == 0 or i == test_len - 1:
            model = arma_model(history, order=order)
        
        # Show progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time_per_step = elapsed / (i + 1)
            remaining_steps = test_len - (i + 1)
            est_remaining_time = remaining_steps * avg_time_per_step
            print(f"    Progress: {i+1}/{test_len} steps ({(i+1)/test_len*100:.1f}%) - "
                  f"Elapsed: {timedelta(seconds=int(elapsed))}, "
                  f"Est. remaining: {timedelta(seconds=int(est_remaining_time))}")
    
    # Show final timing
    total_time = time.time() - start_time
    print(f"    Completed in {timedelta(seconds=int(total_time))}")
    
    return np.array(predictions)

def main():
    """
    Main function to test ARMA models on different ETFs
    """
    np.random.seed(42)
    etfs = df_training_set_daily.columns
    
    # Use copies of the dataframes
    df_train = df_training_set_daily.copy()
    df_test = df_test_set_daily.copy()

    # Define orders for each ETF as per the paper
    etf_orders = {
        'SPY US Equity': (8, 8),
        'DIA US Equity': (10, 10),
        'QQQ US Equity': (7, 7)
    }
    
    # Display basic dataset info
    print("Training data shape:", df_train.shape)
    print("Test data shape:", df_test.shape)
    
    # One-step ahead forecasting with model updates every 5 days
    print("\nOne-Step-Ahead ARMA Forecasting Results:")
    print("=" * 60)
    
    for etf in etfs:
        print(f"\nETF: {etf}")
        order = etf_orders.get(etf, (8, 8))
        print(f"  Using order: {order}")
        
        try:
            # Perform one-step-ahead forecasting with model update every 5 days
            predictions = one_step_ahead_forecast(
                df_train[etf], 
                df_test[etf], 
                order=order,
                update_frequency=5  # Re-estimate model every 5 days for speed
            )
            
            # Calculate metrics
            y_true = df_test[etf].values
            y_pred = predictions
            
            mae = mean_absolute_error(y_true, y_pred)
            mape = calculate_mape_corrected(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            theils_u = calculate_theils_u(y_true, y_pred)
            
            print(f"    MAE: {mae:.4f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Theil's U: {theils_u:.4f}")
            
            # Print first few predictions vs actual
            print("\n    First 5 predictions vs actual:")
            for i in range(min(5, len(y_true))):
                print(f"    Day {i+1}: Pred = {y_pred[i]:.6f}, Actual = {y_true[i]:.6f}")
                
        except Exception as e:
            print(f"    Error fitting model: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # For comparison: standard forecasting method
    print("\n\nStandard ARMA Forecasting Results (for comparison):")
    print("=" * 60)
    
    for etf in etfs:
        print(f"\nETF: {etf}")
        order = etf_orders.get(etf, (8, 8))
        print(f"  Using order: {order}")
        
        try:
            # Fit model on entire training set once
            model = arma_model(df_train[etf], order=order)
            
            # Make forecasts for all test period at once
            predictions = model.forecast(steps=len(df_test))
            
            # Calculate metrics
            y_true = df_test[etf].values
            y_pred = predictions.values
                
            mae = mean_absolute_error(y_true, y_pred)
            mape = calculate_mape_corrected(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            theils_u = calculate_theils_u(y_true, y_pred)
            
            print(f"    MAE: {mae:.4f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Theil's U: {theils_u:.4f}")
            
        except Exception as e:
            print(f"    Error fitting model: {str(e)}")
            
if __name__ == "__main__":
    main()