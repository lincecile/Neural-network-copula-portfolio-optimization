import pandas as pd
import numpy as np
from scipy.stats import jarque_bera, spearmanr
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv(r"data/px_last.csv", index_col=0, parse_dates=True)



############################################################################
####### TABLE 2 Descriptive statistics and correlation matrix ##############
############################################################################

df = df.loc["2011-01-03":"2015-04-13"]

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

desc_stats_df = pd.DataFrame(stats).T

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
