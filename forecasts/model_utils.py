"""
Utility functions for saving and loading neural network models
"""

import os
import pickle
import pandas as pd

def save_model(model, base_dir='./models'):
    """
    Save a trained NnForecaster model to a pickle file.
    
    Args:
        model: An instance of NnForecaster (or subclass)
        base_dir (str): Base directory to save the model
    
    Returns:
        str: Path to the saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a model dictionary with all necessary components
    model_dict = {
        'model': model,
        'x_scaler': model.scaler,
        'y_scaler': model.y_scaler,
        'ticker': model.ticker,
        'model_type': model.model,
        'train_timestamp': pd.Timestamp.now()
    }
    
    # Generate filename based on model type and ticker
    filename = f'{base_dir}/{model.model}_{model.ticker.name}.pkl'
    
    # Save the model dictionary
    with open(filename, 'wb') as f:
        pickle.dump(model_dict, f)
        
    print(f"Model saved to {filename}")
    return filename

def load_model(model_type, ticker, base_dir='./models'):
    """
    Load a previously trained model from a pickle file.
    
    Args:
        model_type (str): Type of model to load (MLP, RNN, PSN)
        ticker: Ticker enum for which the model was trained
        base_dir (str): Base directory where models are stored
        
    Returns:
        NnForecaster: Loaded model instance
    """
    # Generate filename based on model type and ticker
    filename = f'{base_dir}/{model_type}_{ticker.name}.pkl'
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
        
    # Load the model dictionary
    with open(filename, 'rb') as f:
        model_dict = pickle.load(f)
        
    print(f"Model loaded from {filename} (trained on {model_dict['train_timestamp']})")
    
    return model_dict['model']
