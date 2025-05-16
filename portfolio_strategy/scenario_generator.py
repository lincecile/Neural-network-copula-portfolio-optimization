# portfolio_strategy/scenario_generator.py

import numpy as np

from clean_df_paper import df_out_sample_set_weekly
from Copula import SkewedTCopulaModel
def generate_return_scenarios(copula_model:SkewedTCopulaModel, date, n_samples=1000):
    """
    Generate return scenarios using the skewed-t copula model.
    
    Args:
        copula_model: Trained instance of SkewedTCopulaModel.
        date: Date for which to simulate.
        n_samples (int): Number of return scenarios.
        
    Returns:
        np.ndarray: Simulated returns (n_samples x n_assets)
    """
    try:
        simulated_df = copula_model.simulate_skewed_t_copula(date, n_samples=n_samples, weekly_returns=df_out_sample_set_weekly)
        return simulated_df.values
    except Exception as e:
        raise RuntimeError(f"Simulation failed for {date}: {e}")
