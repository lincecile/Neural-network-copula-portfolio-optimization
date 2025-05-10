#%% imports

from abc import ABC, abstractmethod
import typing as tp
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

#%% class

class NnForecaster(ABC):
    """
    Abstract base class for forecasting models.
    """
    def __init__(self):
        self.x = None
        self.layer1 = None
        self.layer2 = None
        
        scaler = StandardScaler()
        
        self.x_train = scaler.fit_transform(self._seperate_x()[0])
        self.x_test = scaler.fit_transform(self._seperate_x()[1])
        self.x_out = scaler.fit_transform(self._seperate_x()[2])
        
        self.x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        self.x_test_tensor = torch.tensor(self.x_test, dtype=torch.float32)
        self.x_out_tensor = torch.tensor(self.x_out, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self._seperate_x()[3], dtype=torch.float32).view(-1, 1)

    @abstractmethod
    def _seperate_x(self) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def init_weights(self):
        # Initialize weights with N(0, 1) and biases with 0
        pass
    
    @abstractmethod
    def forward(self, x):
        pass