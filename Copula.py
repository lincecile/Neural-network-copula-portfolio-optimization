import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import t, norm
from arch import arch_model
from datetime import timedelta

class SkewedTCopula:
    def __init__(self, n_dim):
        """
        Initialize a skewed-t copula for n dimensions
        
        Parameters:
        -----------
        n_dim : int
            Number of dimensions (number of tickers)
        """
        self.n_dim = n_dim
        self.df = None  # Degrees of freedom
        self.skew_params = None  # Skewness parameters
        self.corr_matrix = None  # Correlation matrix
        
    def _skewed_t_cdf(self, x, df, alpha):
        """
        Cumulative distribution function of the skewed-t distribution
        
        Parameters:
        -----------
        x : array
            Evaluation points
        df : float
            Degrees of freedom
        alpha : float
            Skewness parameter
        """
        # Conditional transformation
        m = np.sqrt(df) * np.pi * (alpha - 1/alpha) / (2 * np.sqrt(df-2))
        
        # Ajouter une protection contre les valeurs négatives
        term = (alpha**2 + 1/alpha**2 - 1 - m**2)
        if np.any(term < 0):
            term = np.maximum(term, 0)  # Assurer que le terme est positif
        s = np.sqrt(term)
        
        z = x + m
        pos_mask = z >= 0
        
        result = np.zeros_like(x, dtype=float)
        # For positive values
        result[pos_mask] = t.cdf(z[pos_mask]/alpha, df) * 2
        # For negative values
        result[~pos_mask] = t.cdf(z[~pos_mask]*alpha, df) * 2
        
        return result
    
    def _skewed_t_ppf(self, u, df, alpha):
        """
        Percent point function (inverse of CDF) of the skewed-t distribution
        """
        # Conditional transformation
        m = np.sqrt(df) * np.pi * (alpha - 1/alpha) / (2 * np.sqrt(df-2))
        
        # Ajouter une protection contre les valeurs négatives
        term = (alpha**2 + 1/alpha**2 - 1 - m**2)
        if np.any(term < 0):
            term = np.maximum(term, 0)  # Assurer que le terme est positif
        s = np.sqrt(term)
        
        result = np.zeros_like(u, dtype=float)
        # For u <= 0.5
        cond = u <= 0.5
        result[cond] = t.ppf(u[cond]/2, df) / alpha
        # For u > 0.5
        result[~cond] = t.ppf((u[~cond]-0.5)/2 + 0.5, df) * alpha
        
        return result - m
        
    def fit(self, U, R=None, method='MLE'):
        """
        Fit the skewed-t copula to the data
        
        Parameters:
        -----------
        U : array-like, shape (n_samples, n_dim)
            Uniform data (after transformation by marginal CDF)
        R : array-like, shape (n_dim, n_dim), optional
            Correlation matrix (if pre-estimated, e.g., from DCC model)
        method : str
            Estimation method ('MLE' or 'IFM')
        """
        if method not in ['MLE', 'IFM']:
            raise ValueError("Method must be 'MLE' or 'IFM'")
        
        # Transform to normal values
        Z = norm.ppf(U)
        
        # Estimate correlation matrix if not provided
        if R is None:
            P = np.corrcoef(Z.T)
        else:
            P = R
        
        # Initial parameters
        init_params = np.ones(self.n_dim + 1) * 5  # df = 5, alpha_i = 5
        
        # Negative log-likelihood function - CORRIGÉ POUR RETOURNER UN SCALAIRE
        def neg_loglik(params):
            df = params[0]
            alphas = params[1:]
            
            if df <= 2 or np.any(alphas <= 0):
                return 1e10
            
            # Transform to skewed-t quantiles
            T = np.zeros_like(U)
            for i in range(self.n_dim):
                T[:, i] = self._skewed_t_ppf(U[:, i], df, alphas[i])
            
            # Log-likelihood of the skewed-t copula
            try:
                # Compute multivariate t density
                log_lik = 0.0
                for i in range(len(U)):
                    # Utiliser la log-densité t multivariée correcte
                    pdf_value = t.pdf(T[i], df, loc=0, scale=1)
                    log_lik += np.log(np.maximum(1e-10, np.sum(pdf_value)))
                
                # Retourner un scalaire unique (la somme)
                return -log_lik
            except Exception as e:
                print(f"Error in neg_loglik: {e}")
                return 1e10
        
        # Optimization
        bounds = [(2.1, 30)] + [(0.1, 10)] * self.n_dim
        result = minimize(neg_loglik, init_params, bounds=bounds, method='L-BFGS-B')
        
        self.df = result.x[0]
        self.skew_params = result.x[1:]
        self.corr_matrix = P
        
        return self
    
    def sample(self, n_samples=1000, R=None):
        """
        Generate samples from the skewed-t copula
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        R : array-like, shape (n_dim, n_dim), optional
            Correlation matrix to use (overrides fitted correlation)
        
        Returns:
        --------
        U : array, shape (n_samples, n_dim)
            Samples distributed according to the copula
        """
        if self.df is None or self.skew_params is None:
            raise ValueError("Copula must be fitted before generating samples")
        
        # Use provided correlation matrix if given
        corr_matrix = R if R is not None else self.corr_matrix
        
        # Generate multivariate t samples
        Z = np.random.multivariate_normal(np.zeros(self.n_dim), corr_matrix, size=n_samples)
        S = np.sqrt(self.df / np.random.chisquare(self.df, size=(n_samples, 1)))
        T = Z * S
        
        # Apply skewness
        U = np.zeros_like(T)
        for i in range(self.n_dim):
            # Conditional transformation
            alpha = self.skew_params[i]
            m = np.sqrt(self.df) * np.pi * (alpha - 1/alpha) / (2 * np.sqrt(self.df-2))
            
            t_skewed = T[:, i] + m
            pos_mask = t_skewed >= 0
            
            U[pos_mask, i] = t.cdf(t_skewed[pos_mask]/alpha, self.df) * 2
            U[~pos_mask, i] = t.cdf(t_skewed[~pos_mask]*alpha, self.df) * 2
        
        return U