# portfolio_strategy/optimizer_cvar.py

import numpy as np

import cvxpy as cp

def solve_cvar_lp(scenarios, alpha=0.95, allow_short=False):
    """
    Solve CVaR portfolio optimization as LP (Rockafellar & Uryasev).
    
    Args:
        scenarios (np.ndarray): Shape (n_samples, n_assets). Each row is a return vector.
        alpha (float): Confidence level for CVaR.
        allow_short (bool): Whether to allow short-selling.
    
    Returns:
        np.ndarray: Optimal portfolio weights.
    """
    n_samples, n_assets = scenarios.shape
    w = cp.Variable(n_assets)
    z = cp.Variable(n_samples)
    eta = cp.Variable()

    # Loss = - return
    losses = -scenarios @ w

    constraints = [
        z >= 0,
        z >= losses - eta,
        cp.sum(w) == 1
    ]
    if not allow_short:
        constraints.append(w >= 0)

    # Objective: eta + 1/(1-alpha) * average of z
    objective = cp.Minimize(eta + (1 / ((1 - alpha) * n_samples)) * cp.sum(z))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return w.value
