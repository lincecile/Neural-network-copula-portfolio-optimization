from clean_df_paper import df_out_sample_set_daily
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from RNN import result_dict
rnn_pred_out = result_dict
from MLP import result_dict
mlp_pred_out = result_dict
from PSN import result_dict
psn_pred_out = result_dict


df = df_out_sample_set_daily.copy()
columns = list(df.columns)

pred_dict = {'rnn' : rnn_pred_out,'mlp' :  mlp_pred_out, "psn" : psn_pred_out}

pred_dict_weekly = {}
for model, dico in pred_dict.items():
    for k, df in dico.items():
        weekly_returns = df.resample('W').sum()
        pred_dict_weekly[model + ' ' + k] = weekly_returns
