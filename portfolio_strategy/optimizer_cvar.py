import sys
import subprocess
import numpy as np

# Install ecos if not available (skip direct import)
try:
    import cvxpy as cp
    cp.installed_solvers()  # will raise if cvxpy is not ready
except (ImportError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy", "ecos"])
    import cvxpy as cp  # re-import after install

def solve_cvar_lp(scenarios, alpha=0.95, allow_short=False):
    import cvxpy as cp

    n_samples, n_assets = scenarios.shape
    w = cp.Variable(n_assets)
    z = cp.Variable(n_samples)
    eta = cp.Variable()

    losses = -scenarios @ w

    constraints = [
        z >= 0,
        z >= losses - eta,
        cp.sum(w) == 1
    ]
    if not allow_short:
        constraints.append(w >= 0)

    objective = cp.Minimize(eta + (1 / ((1 - alpha) * n_samples)) * cp.sum(z))
    problem = cp.Problem(objective, constraints)

    # Try solvers in order of reliability
    for solver in [cp.ECOS, cp.OSQP, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=False)
            if w.value is not None:
                return w.value
        except cp.SolverError as e:
            print(f"⚠️ {solver} solver failed: {str(e)}")
            continue

    raise RuntimeError("All solvers failed.")

