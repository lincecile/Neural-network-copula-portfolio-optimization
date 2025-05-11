
"""
Simplified test script for the neural network forecasting models
This script demonstrates the basic functionality of creating and using a forecaster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticker_dataclass import Ticker
from clean_df_paper import df_training_set_daily, df_test_set_daily, df_out_sample_set_daily
from forecasts.rnn import RNNForecaster
from forecasts.fake_psn import FakePsnForecaster

def test_rnn_forecaster():
    """Test the RNN forecasting model"""
    print("\n" + "="*50)
    print("TESTING RNN FORECASTER")
    print("="*50)
      # Select a ticker for testing
    ticker = Ticker.dia
    print(f"Testing with ticker: {ticker}")
    
    # Get the training, test, and out-of-sample dataframes for the ticker
    train_df = df_training_set_daily[ticker].to_frame()
    test_df = df_test_set_daily[ticker].to_frame()
    out_df = df_out_sample_set_daily[ticker].to_frame()
    
    # Create the RNN forecaster
    print("Creating RNN forecaster...")
    rnn = RNNForecaster(
        df_train=train_df, 
        df_test=test_df, 
        df_out=out_df, 
        ticker=ticker,
        hardcoded=True
    )
    
    # Train the model
    print("Training RNN forecaster...")
    losses = rnn.train_model()
      # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('RNN Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rnn_training_loss.png')
    plt.close()
    
    # Evaluate the model
    print("Evaluating RNN forecaster on test data...")
    mae, mape, rmse, theilu, preds = rnn.evaluate_model()
    
    # Make predictions on out-of-sample data
    print("Making predictions with RNN forecaster on out-of-sample data...")
    out_preds = rnn.predict()
    
    print(f"Out-of-sample predictions shape: {out_preds.shape}")
    print(f"First few predictions: {out_preds[:5].flatten()}")
    
    return rnn, losses, out_preds

def test_fakepsn_forecaster():
    """Test the Fake PSN forecasting model"""
    print("\n" + "="*50)
    print("TESTING FAKE PSN FORECASTER")
    print("="*50)
      # Select a ticker for testing
    ticker = Ticker.spy
    print(f"Testing with ticker: {ticker}")
    
    # Get the training, test, and out-of-sample dataframes for the ticker
    train_df = df_training_set_daily[ticker].to_frame()
    test_df = df_test_set_daily[ticker].to_frame()
    out_df = df_out_sample_set_daily[ticker].to_frame()
    
    # Create the Fake PSN forecaster
    print("Creating Fake PSN forecaster...")
    psn = FakePsnForecaster(
        df_train=train_df, 
        df_test=test_df, 
        df_out=out_df, 
        ticker=ticker,
        hardcoded=True
    )
    
    # Initialize weights
    psn.init_weights()
    
    # Train the model
    print("Training Fake PSN forecaster...")
    losses = psn.train_model()
      # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Fake PSN Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fake_psn_training_loss.png')
    plt.close()
    
    # Evaluate the model
    print("Evaluating Fake PSN forecaster on test data...")
    mae, mape, rmse, theilu, preds = psn.evaluate_model()
    
    # Make predictions on out-of-sample data
    print("Making predictions with Fake PSN forecaster on out-of-sample data...")
    out_preds = psn.predict()
    
    print(f"Out-of-sample predictions shape: {out_preds.shape}")
    print(f"First few predictions: {out_preds[:5].flatten()}")
    
    return psn, losses, out_preds

def compare_models(rnn_preds, psn_preds, ticker):
    """Compare predictions from both models"""
    print("\n" + "="*50)
    print("COMPARING MODELS")
    print("="*50)
    
    # Get actual values
    out_df = df_out_sample_set_daily[ticker].to_frame()
      # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(out_df.index[-len(rnn_preds):], out_df.values[-len(rnn_preds):], label='Actual', color='black')
    plt.plot(out_df.index[-len(rnn_preds):], rnn_preds, label='RNN Predictions', linestyle='--', color='blue')
    plt.plot(out_df.index[-len(psn_preds):], psn_preds, label='Fake PSN Predictions', linestyle='--', color='red')
    plt.title(f'Model Comparison for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'model_comparison_{ticker}.png')
    plt.close()

if __name__ == "__main__":
    # Test RNN forecaster
    rnn, rnn_losses, rnn_preds = test_rnn_forecaster()
    
    # Test Fake PSN forecaster
    psn, psn_losses, psn_preds = test_fakepsn_forecaster()
      # Compare models
    try:
        compare_models(rnn_preds, psn_preds, Ticker.dia)
    except Exception as e:
        print(f"Error in model comparison: {e}")
    
    print("\nTests completed!")