#%% imports

from abc import ABC, abstractmethod
import json
import typing as tp
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

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
        
        self.train = df_train
        self.test = df_test
        self.out = df_out
        
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
        
        #train
        self.x_train = create_lag_features(df_train, 
                                           target_col=df_train.columns[0], 
                                           lags=self.input_lags).drop(columns=['target'])
        self.y_train = create_lag_features(df_train,
                                           target_col=df_train.columns[0], 
                                           lags=self.input_lags)['target']
        
        #test
        self.x_test = create_lag_features(df_test,
                                          target_col=df_test.columns[0], 
                                          lags=self.input_lags).drop(columns=['target'])
        self.y_test = create_lag_features(df_test,
                                           target_col=df_test.columns[0], 
                                           lags=self.input_lags)['target']
        
        #out
        self.x_out = create_lag_features(df_out,
                                         target_col=df_out.columns[0], 
                                         lags=self.input_lags).drop(columns=['target'])
        self.y_out = create_lag_features(df_out,
                                          target_col=df_out.columns[0], 
                                          lags=self.input_lags)['target']

        #standardize
        scaler = StandardScaler()
        
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.x_out = scaler.transform(self.x_out)
        
        self.x_train_tensor = torch.tensor(self.x_train, dtype=torch.float64)
        self.x_test_tensor = torch.tensor(self.x_test, dtype=torch.float64)
        self.x_out_tensor = torch.tensor(self.x_out, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float64).view(-1, 1)
        
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
                "Hidden nodes": 6,
                "Output node": 1,
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