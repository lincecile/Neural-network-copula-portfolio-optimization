#%% imports

from enum import Enum

#%% class

class Ticker(str, Enum):
    spy = "SPY US Equity"
    dia = "DIA US Equity"
    qqq = "QQQ US Equity"
    # Add more tickers as needed