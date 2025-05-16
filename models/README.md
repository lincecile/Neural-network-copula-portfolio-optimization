# Neural Network Models Directory

This directory contains pickled trained neural network models used for time series forecasting.

## Model Types
- MLP (Multilayer Perceptron)
- RNN (Recurrent Neural Network)
- PSN (Pi-Sigma Network)

## File Naming Convention
Files are named using the convention: `{model_type}_{ticker_name}.pkl`

Examples:
- `MLP_spy.pkl` - MLP model trained for SPY ETF
- `RNN_dia.pkl` - RNN model trained for DIA ETF
- `PSN_qqq.pkl` - PSN model trained for QQQ ETF

## Using the Models
Models are automatically saved after training. To load and use these models:

```python
from forecasts.model_utils import load_model
from ticker_dataclass import Ticker
from forecasts.nn_model_dataclass import NnModel

# Load a model
model = load_model(NnModel.rnn, Ticker.spy)

# Use the model for prediction
predictions = model.predict()
```

For a complete example, see `forecasts/load_model_example.py`
