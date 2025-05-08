import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_df_paper import df_training_set_daily

def autocorrelation_analysis(returns_series, etf_name):
    """
    Calcule et visualise l'autocorrélation et l'autocorrélation partielle
    """
    # Créer une figure pour ACF et PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Autocorrélation sur le premier subplot
    plot_acf(returns_series, lags=20, alpha=0.1, ax=ax1, title=f'Autocorrélation - {etf_name}')
    
    # Autocorrélation partielle sur le second subplot
    plot_pacf(returns_series, lags=20, alpha=0.1, ax=ax2, title=f'Autocorrélation Partielle - {etf_name}')
    
    plt.tight_layout()
    plt.show()

# Boucle principale
for etf in df_training_set_daily.columns:
    print(f"\nAnalyse de l'autocorrélation pour {etf}")
    
    # Graph des rendements dans une figure séparée
    plt.figure(figsize=(15, 5))
    plt.plot(df_training_set_daily[etf], label=f'Returns - {etf}')
    plt.title(f'Returns - {etf}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.show()
    
    # Analyse d'autocorrélation
    autocorrelation_analysis(df_training_set_daily[etf], etf)