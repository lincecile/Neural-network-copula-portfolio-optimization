import pandas as pd
import pandas_market_calendars as mcal

df = pd.read_csv(r"data/px_last.csv", index_col=0, parse_dates=True)
df = df[['SPY US Equity','DIA US Equity', 'QQQ US Equity']]

df = df.pct_change().dropna()

start_date_total_set = "2011-01-03"
end_date_total_set = "2025-05-03"

# Jours de bourse aux États-Unis (NYSE)
nyse = mcal.get_calendar("NYSE")

# Obtenir tous les jours d'ouverture du marché entre les deux dates
trading_days = nyse.schedule(start_date=start_date_total_set, end_date=end_date_total_set)
trading_days_index = trading_days.index

# df utilisé dans le papier
df_total_set = df.loc[df.index.intersection(trading_days_index)]

###############################################################################
########################## Répartition des périodes ###########################
###############################################################################

# in sample = 70%
_70_percent = int(len(trading_days_index) * 0.7)
traning_test_set_end = trading_days_index[_70_percent].strftime('%Y-%m-%d')

_70_percent_in_sample = int(len(trading_days_index[:_70_percent]) * 0.7)
training_set_end = trading_days_index[_70_percent_in_sample-1].strftime('%Y-%m-%d')
test_set_start = trading_days_index[_70_percent_in_sample].strftime('%Y-%m-%d')

# out of sample = 30%
_30_percent = int(len(trading_days_index) * 0.7) + 1
out_sample_set_start = trading_days_index[_30_percent].strftime('%Y-%m-%d')

###############################################################################
########################## Création des set ###################################
###############################################################################

# in sample = 70%
# données pour l'échantillon d'entraînement
start_date_training_set = start_date_total_set
end_date_training_set = training_set_end

# données pour l'échantillon de test
start_date_test_set = test_set_start
end_date_test_set = traning_test_set_end

# out of sample = 30%
# données pour l'échantillon hors échantillon
start_date_out_sample_set = out_sample_set_start
end_date_out_sample_set = end_date_total_set

# Création des ensembles de données
df_in_sample_set = df_total_set.loc[start_date_training_set:end_date_test_set]
df_training_set = df_total_set.loc[start_date_training_set:end_date_training_set]
df_test_set = df_total_set.loc[start_date_test_set:end_date_test_set]
df_out_sample_set = df_total_set.loc[start_date_out_sample_set:end_date_out_sample_set]

