import os
# Set working directory to the root of your project (adjust path if needed)
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from Copula import SkewedTCopulaModel
from clean_df_paper import df_out_sample_set_weekly, df_test_set_weekly

# Load DCC matrices
with open("dcc_results_all_weekly.pkl", "rb") as f:
    dcc_data = pickle.load(f)

# Initialize copula model
copula = SkewedTCopulaModel.__new__(SkewedTCopulaModel)
copula.weekly_matrices = dcc_data["matrices"]
copula.weekly_dates = dcc_data["weekly_dates"]
copula.copula_params = {}  # <- ✅ fix: manual initialization

# Estimate copula parameters
copula.estimate_weekly_copula_parameters(df_test_set_weekly + df_out_sample_set_weekly, window_size=26)
print("Earliest copula param:", min(copula.copula_params.keys()))
print("Latest copula param:", max(copula.copula_params.keys()))

print(copula.copula_params)
# Export results
with open("copula_params_dcc.pkl", "wb") as f:
    pickle.dump({"copula_params": copula.copula_params}, f)

# with open("copula_params_dcc.pkl", "rb") as f:
#     copula_data = pickle.load(f)
#     aa = copula_data["copula_params"]

# print(aa)

print("\n✅ Copula parameters saved to: copula_params_dcc.pkl")
