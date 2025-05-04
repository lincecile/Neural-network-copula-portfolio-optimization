import pandas as pd
from scipy.stats import jarque_bera, spearmanr
from statsmodels.tsa.stattools import adfuller
from clean_df_paper import df_total_set_daily

############################################################################
####### TABLE 2 Descriptive statistics and correlation matrix ##############
############################################################################

# PANEL A: Statistiques descriptives
stats = {}

for col in df_total_set_daily.columns:
    ret = df_total_set_daily[col]
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

# PANEL B: Matrice de corrélation

# Corrélation de Pearson
pearson_corr = df_total_set_daily.corr()

# Corrélation de Spearman
spearman_corr = df_total_set_daily.corr(method='spearman')

# Affichage

print("Panel A: Descriptive statistics")
print(desc_stats_df.round(5))

print("\nPanel B: Correlation matrix (Pearson followed by Spearman in brackets)")
for row in df_total_set_daily.columns:
    line = []
    for col in df_total_set_daily.columns:
        pearson_val = pearson_corr.loc[row, col]
        spearman_val = spearman_corr.loc[row, col]
        line.append(f"{pearson_val:.3f} [{spearman_val:.3f}]")
    print(f"{row} {' '.join(line)}")
