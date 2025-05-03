import pandas as pd
import numpy as np
from scipy.stats import jarque_bera, spearmanr
from statsmodels.tsa.stattools import adfuller
import pandas_market_calendars as mcal

df = pd.read_csv(r"data/px_last.csv", index_col=0, parse_dates=True)

start_date = "2011-01-03"
end_date = "2015-04-13"
max_date = "2025-05-03"

# Jours de bourse aux États-Unis (NYSE)
nyse = mcal.get_calendar("NYSE")

# Obtenir tous les jours d'ouverture du marché entre les deux dates
trading_days = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days_index = trading_days.index

df = df.loc[df.index.intersection(trading_days_index)]

############################################################################
####### TABLE 2 Descriptive statistics and correlation matrix ##############
############################################################################

# Calcul des rendements logarithmiques
returns = np.log(df / df.shift(1)).dropna()

# PANEL A: Statistiques descriptives
stats = {}

for col in returns.columns:
    ret = returns[col]
    jb_stat, jb_p = jarque_bera(ret)
    adf_stat, adf_p, *_ = adfuller(ret)

    stats[col] = {
        "Mean": ret.mean(),
        "Standard deviation": ret.std(),
        "Skewness": ret.skew(),
        "Kurtosis": ret.kurtosis() + 3,  # Ajouter 3 pour la kurtose totale
        "Jarque–Bera (p value)": jb_p,
        "ADF (p value)": adf_p
    }

desc_stats_df = pd.DataFrame(stats)
desc_stats_df = desc_stats_df[['SPY US Equity','DIA US Equity', 'QQQ US Equity']]

# PANEL B: Matrice de corrélation

# Corrélation de Pearson
pearson_corr = returns.corr()

# Corrélation de Spearman
spearman_corr = returns.corr(method='spearman')

# Affichage

print("Panel A: Descriptive statistics")
print(desc_stats_df.round(5))

print("\nPanel B: Correlation matrix (Pearson followed by Spearman in brackets)")
for row in returns.columns:
    line = []
    for col in returns.columns:
        pearson_val = pearson_corr.loc[row, col]
        spearman_val = spearman_corr.loc[row, col]
        line.append(f"{pearson_val:.3f} [{spearman_val:.3f}]")
    print(f"{row} {' '.join(line)}")
