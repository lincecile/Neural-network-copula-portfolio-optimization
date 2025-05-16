# portfolio_strategy/scenario_generator.py

import numpy as np

def generate_return_scenarios(copula_model, date, n_samples=1000):
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
        simulated_df = copula_model.simulate_skewed_t_copula(date, n_samples=n_samples)
        return simulated_df.values
    except Exception as e:
        raise RuntimeError(f"Simulation failed for {date}: {e}")
