import pandas as pd
import numpy as np
from arch import arch_model
from scipy.optimize import minimize
import warnings
import pickle
warnings.filterwarnings("ignore")

class DCC_Covariance:
    def __init__(self, returns_data):
        self.returns = returns_data.copy()
        self.tickers = list(self.returns.columns)
        self.T, self.n = self.returns.shape
        
        # Paramètres ARMA spécifiés dans le papier
        self.arma_params = {
            'SPY': (8, 8),
            'DIA': (10, 10), 
            'QQQ': (7, 7)
        }
        
        self.standardized_residuals = None
        self.conditional_variances = None
        self.H_t = None
        self.alpha = None
        self.beta = None
        
    def get_arma_order(self, ticker):
        """Obtenir l'ordre ARMA pour un ETF"""
        for key in self.arma_params:
            if key in ticker.upper():
                return self.arma_params[key]
        return (1, 1)  # Par défaut
        
    def fit_arma_gjr_garch_models(self):
        residuals_dict = {}
        variances_dict = {}
        
        for ticker in self.tickers:
            returns = self.returns[ticker].values
            p, q = self.get_arma_order(ticker)
            
            try:
                model = arch_model(
                    returns,
                    mean='AR',
                    vol='GARCH',
                    lags=p,
                    p=1, o=1, q=1
                )
                
                result = model.fit(disp='off', show_warning=False)
                
                # Résidus standardisés
                residuals = result.resid
                volatility = result.conditional_volatility
                
                # Gestion des valeurs problématiques
                mask = ~(np.isnan(residuals) | np.isnan(volatility) | (volatility <= 0))
                if not mask.all():
                    residuals = pd.Series(residuals).fillna(method='ffill').fillna(method='bfill').values
                    volatility = pd.Series(volatility).fillna(method='ffill').fillna(method='bfill').values
                    volatility = np.maximum(volatility, 1e-8)
                
                residuals_dict[ticker] = residuals / volatility
                variances_dict[ticker] = volatility ** 2
                
            except:
                # Fallback simple
                residuals = returns - np.mean(returns)
                sigma = np.std(returns)
                residuals_dict[ticker] = residuals / sigma
                variances_dict[ticker] = np.full_like(returns, sigma**2)
        
        self.standardized_residuals = pd.DataFrame(residuals_dict, index=self.returns.index)
        self.conditional_variances = pd.DataFrame(variances_dict, index=self.returns.index)
        
        # Nettoyer une dernière fois
        self.standardized_residuals = self.standardized_residuals.fillna(0)
        self.conditional_variances = self.conditional_variances.fillna(self.returns.var())
    
    def estimate_dcc_parameters(self):
        """Estimer alpha et beta par maximum de vraisemblance"""
        epsilon = self.standardized_residuals.values
        T, n = epsilon.shape
        
        def dcc_log_likelihood(params):
            alpha, beta = params
            
            if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
                return 1e6
                
            Q_bar = np.corrcoef(epsilon.T)
            Q_t = np.zeros((T, n, n))
            Q_t[0] = Q_bar
            
            log_lik = 0
            
            for t in range(1, T):
                Q_t[t] = (1 - alpha - beta) * Q_bar + alpha * np.outer(epsilon[t-1], epsilon[t-1]) + beta * Q_t[t-1]
                
                Q_diag_inv = 1.0 / np.sqrt(np.diag(Q_t[t]))
                R_t = Q_t[t] * np.outer(Q_diag_inv, Q_diag_inv)
                np.fill_diagonal(R_t, 1.0)
                
                sign, logdet = np.linalg.slogdet(R_t)
                if sign <= 0:
                    return 1e6
                    
                log_lik += -0.5 * (logdet + epsilon[t] @ np.linalg.inv(R_t) @ epsilon[t] - epsilon[t] @ epsilon[t])
            
            return -log_lik
        
        bounds = [(0.001, 0.999), (0.001, 0.999)]
        constraints = [{'type': 'ineq', 'fun': lambda x: 0.999 - x[0] - x[1]}]
        
        result = minimize(
            dcc_log_likelihood,
            x0=[0.01, 0.95],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.alpha, self.beta = result.x
        else:
            self.alpha, self.beta = 0.01, 0.95
        
        return self.alpha, self.beta
    
    def fit_dcc_model(self):
        """Ajuster le modèle DCC"""
        self.estimate_dcc_parameters()
        
        epsilon = self.standardized_residuals.values
        T, n = epsilon.shape
        
        Q_bar = np.corrcoef(epsilon.T)
        Q_t = np.zeros((T, n, n))
        self.H_t = np.zeros((T, n, n))
        
        Q_t[0] = Q_bar.copy()
        
        for t in range(T):
            if t > 0:
                Q_t[t] = (1 - self.alpha - self.beta) * Q_bar + self.alpha * np.outer(epsilon[t-1], epsilon[t-1]) + self.beta * Q_t[t-1]
            
            Q_diag = np.sqrt(np.diag(Q_t[t]))
            R_t = Q_t[t] / np.outer(Q_diag, Q_diag)
            np.fill_diagonal(R_t, 1.0)
            
            D_t = np.diag(np.sqrt(self.conditional_variances.iloc[t].values))
            self.H_t[t] = D_t @ R_t @ D_t
    
    def get_covariance_matrix(self, date_or_index):
        """Obtenir la matrice de covariance"""
        if isinstance(date_or_index, int):
            idx = date_or_index
        else:
            idx = self.returns.index.get_loc(date_or_index)
        
        return pd.DataFrame(
            self.H_t[idx], 
            columns=self.tickers, 
            index=self.tickers
        )
    
    def export_to_pickle(self, filepath='dcc_results.pkl'):
        """Exporter tous les résultats dans un fichier pickle"""
        # Préparer les données à exporter
        export_data = {
            'tickers': self.tickers,
            'dates': self.returns.index.tolist(),
            'H_t': self.H_t,  # Matrices de covariance pour toutes les dates
            'alpha': self.alpha,
            'beta': self.beta,
            'conditional_variances': self.conditional_variances,
            'standardized_residuals': self.standardized_residuals,
            'arma_params': self.arma_params
        }
        
        # Sauvegarder dans un fichier pickle
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"Résultats DCC exportés dans: {filepath}")
        return filepath

def main():
    from clean_df_paper import df_in_sample_set_daily
    
    dcc = DCC_Covariance(df_in_sample_set_daily)
    dcc.fit_arma_gjr_garch_models()
    dcc.fit_dcc_model()
    
    # Exporter les résultats
    dcc.export_to_pickle('dcc_results_in_sample.pkl')
    
    print(f"Paramètres DCC estimés: alpha={dcc.alpha:.4f}, beta={dcc.beta:.4f}")
    print("\n=== 10 dernières matrices de variance-covariance ===")
    
    for i in range(10, 0, -1):
        idx = -i
        date = df_in_sample_set_daily.index[idx]
        print(f"\nDate: {date}")
        cov_matrix = dcc.get_covariance_matrix(idx)
        print(cov_matrix.round(6))
        print("-" * 60)

if __name__ == "__main__":
    main()
