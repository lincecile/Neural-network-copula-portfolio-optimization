import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clean_df_paper import df_training_set_daily

def autocorrelation_analysis(returns_series, etf_name):
    """
    Calcule et visualise l'autocorrélation et l'autocorrélation partielle
    """
    plt.figure(figsize=(15,5))
    
    # Autocorrélation
    plt.subplot(121)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(returns_series, lags=20, ax=plt.gca(), title=f'Autocorrélation - {etf_name}', alpha=0.1)
    plt.title(f'Autocorrélation - {etf_name}')
    
    # Autocorrélation partielle
    plt.subplot(122)
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(returns_series, lags=20, ax=plt.gca(), title=f'Autocorrélation Partielle - {etf_name}', alpha=0.1)
    plt.title(f'Autocorrélation Partielle - {etf_name}')
    
    plt.tight_layout()
    plt.show()
    

for etf in df_training_set_daily.columns:
    print(f"\nAnalyse de l'autocorrélation pour {etf}")
    autocorrelation_analysis(df_training_set_daily[etf], etf)