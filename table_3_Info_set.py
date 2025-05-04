import pandas as pd
from clean_df_paper import df_total_set_prix, df_training_set_daily, df_test_set_daily, df_out_sample_set_daily

############################################################################
######################### TABLE 3 The total data-set #######################
############################################################################

dfs = {
    "Total data-set": df_total_set_prix,
    "Training data-set" : df_training_set_daily,
    "Test data-set": df_test_set_daily,
    "Out-of-sample data-set": df_out_sample_set_daily}

dico = []

for name, df in dfs.items():
    dico = dico + [{"Datasets": name,
        "Trading days": len(df),
        "Start date": df.index.min().strftime("%d/%m/%Y"),
        "End date": df.index.max().strftime("%d/%m/%Y")}]

summary_table = pd.DataFrame(dico)

# Affichage
print("Table 3. The total data-set")
print(summary_table)
