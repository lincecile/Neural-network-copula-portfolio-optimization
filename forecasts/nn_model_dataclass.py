#%% imports

from dataclasses import dataclass

#%% class

@dataclass
class NnModel:
    mlp = 'MLP'
    rnn = 'RNN'
    psn = 'PSN'
    # add more models as needed