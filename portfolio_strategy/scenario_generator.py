# portfolio_strategy/scenario_generator.py

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
    df = copula_model.copula_params[date].get('df', copula_model.copula_params[date]['nu'])
    return df
