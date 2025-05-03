import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from clean_df_paper import df_total_set, df_in_sample_set, df_training_set, df_test_set, df_out_sample_set

# Crée un sous-plot pour chaque série de prix
fig, axes = plt.subplots(3, 2, figsize=(8, 8))

for i, col in enumerate(df_in_sample_set.columns):
    plot_acf(df_in_sample_set[col].dropna(), ax=axes[i, 0], lags=15)
    plot_pacf(df_in_sample_set[col].dropna(), ax=axes[i, 1], lags=15)
    axes[i, 0].set_title(f"ACF de {col}")
    axes[i, 1].set_title(f"PACF de {col}")

plt.tight_layout()
plt.show()


