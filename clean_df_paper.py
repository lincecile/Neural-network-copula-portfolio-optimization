import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

df = pd.read_csv(r"data/px_last.csv", index_col=0, parse_dates=True)
df = df[['SPY US Equity','DIA US Equity', 'QQQ US Equity']]

start_date_total_set = "2011-01-03"
end_date_total_set = "2015-04-13"

# Jours de bourse aux États-Unis (NYSE)
nyse = mcal.get_calendar("NYSE")

# Obtenir tous les jours d'ouverture du marché entre les deux dates
trading_days = nyse.schedule(start_date=start_date_total_set, end_date=end_date_total_set)
trading_days_index = trading_days.index

# df utilisé dans le papier
df_total_set = df.loc[df.index.intersection(trading_days_index)]

df_total_set = df_total_set.resample('W').last()
df_total_set = np.log(df_total_set / df_total_set.shift(1)).dropna()

###############################################################################
########################## Création des set ###################################
###############################################################################

# in sample = 70%
# 2 ans de données pour l'échantillon d'entraînement
start_date_training_set = start_date_total_set
end_date_training_set = "2012-12-31"

# 1 an de données pour l'échantillon de test
start_date_test_set = "2013-01-02"
end_date_test_set = "2013-12-31"

# out of sample = 30%
# 1.25 an de données pour l'échantillon hors échantillon
start_date_out_sample_set = "2014-01-02"
end_date_out_sample_set = end_date_total_set

# Création des ensembles de données
df_in_sample_set = df_total_set.loc[start_date_training_set:end_date_test_set]
df_training_set = df_total_set.loc[start_date_training_set:end_date_training_set]
df_test_set = df_total_set.loc[start_date_test_set:end_date_test_set]
df_out_sample_set = df_total_set.loc[start_date_out_sample_set:end_date_out_sample_set]

