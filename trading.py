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

result_df = pd.DataFrame()

for n,i in pred_dict.items():
    for c in columns:
        series = df[c]
        series = series[-len(i[c]):]
        for s in range(len(series)):
            if i[c][s] > 0:
                series[s] = series[s]
            else:
                series[s] = -series[s]
        result_df[n + '_' + c] = series
        
result_df_cumprod = (1 + result_df).cumprod()