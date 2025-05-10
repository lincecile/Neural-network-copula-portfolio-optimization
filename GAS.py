import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from arch.univariate import GARCH
from scipy.optimize import minimize
from clean_df_paper import df_training_set_weekly, df_test_set_weekly, df_out_sample_set_weekly

class GASModel:
    def __init__(self, returns_df):
        """
        Initialise le modèle GAS pour plusieurs séries de rendements
        
        Parameters:
        -----------
        returns_df : pandas DataFrame
            DataFrame contenant les rendements des actifs
        """
        self.returns = returns_df
        self.tickers = list(returns_df.columns)
        self.T, self.n = returns_df.shape
        
        if self.n < 2:
            raise ValueError("Le modèle GAS nécessite au moins deux actifs")
        
        self.univariate_models = {}
        self.std_residuals = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        self.conditional_variances = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        # Pour le modèle GAS, on travaille avec Q_t comme DCC mais avec une mise à jour GAS
        self.Q_t = np.zeros((self.T+1, self.n, self.n))
        self.R_t = np.zeros((self.T, self.n, self.n))
        self.H_t = np.zeros((self.T, self.n, self.n))
        
    def fit_univariate_garch(self, p=1, q=1):
        """
        Ajuste des modèles GARCH univariés pour chaque série
        """
        for ticker in self.tickers:
            print(f"Ajustement du modèle GARCH pour {ticker}")
            returns = self.returns[ticker].values
            
            # Utiliser rescale pour éviter les warnings
            model = arch_model(returns, vol='GARCH', p=p, q=q, mean='Zero', rescale=True)
            result = model.fit(disp='off')
            
            self.univariate_models[ticker] = result
            self.std_residuals[ticker] = result.resid / result.conditional_volatility
            self.conditional_variances[ticker] = result.conditional_volatility**2
            
        print("Modèles GARCH univariés ajustés avec succès")
    
    def _gas_likelihood(self, params):
        """
        Fonction de vraisemblance pour l'estimation GAS
        
        Implémentation simplifiée : le score est approximé par (ε_t ⊗ ε_t - Q_{t-1})
        
        Parameters:
        -----------
        params : array
            [omega, A, B] - paramètres globaux pour toutes les corrélations
        """
        omega, A, B = params
        
        # Vérifier la validité des paramètres
        if A < 0 or B < 0 or A + B >= 0.999 or omega <= 0:
            return 1e10
        
        epsilon_t = self.std_residuals.values
        n = self.n
        T = self.T
        
        # Initialisation
        Q_bar = np.corrcoef(epsilon_t.T)
        self.Q_t[0] = Q_bar.copy()
        
        log_likelihood = 0
        
        for t in range(1, T):
            eps_tm1 = epsilon_t[t-1]
            
            # Le score est approximé par (ε_t-1 ⊗ ε_t-1 - Q_t-1)
            score = np.outer(eps_tm1, eps_tm1) - self.Q_t[t-1]
            
            # Mise à jour GAS de Q_t
            self.Q_t[t] = (1 - B) * omega * Q_bar + A * score + B * self.Q_t[t-1]
            
            # S'assurer que Q_t reste une matrice définie positive
            eigvals = np.linalg.eigvals(self.Q_t[t])
            if np.any(eigvals <= 0):
                # Si Q_t n'est pas définie positive, on revient à Q_bar
                self.Q_t[t] = Q_bar.copy()
            
            # Calculer la matrice de corrélation conditionnelle
            Q_t_diag = np.diag(np.sqrt(np.diag(self.Q_t[t])))
            Q_t_diag_inv = np.linalg.inv(Q_t_diag)
            self.R_t[t] = Q_t_diag_inv @ self.Q_t[t] @ Q_t_diag_inv
            
            # S'assurer que R_t est une matrice de corrélation valide
            np.fill_diagonal(self.R_t[t], 1.0)
            
            # Calculer la matrice de covariance conditionnelle
            D_t = np.diag(np.sqrt([self.conditional_variances.iloc[t, i] for i in range(n)]))
            self.H_t[t] = D_t @ self.R_t[t] @ D_t
            
            # Calculer la log-vraisemblance
            eps_t = epsilon_t[t]
            if not np.all(np.isnan(eps_t)):
                try:
                    H_t_det = np.linalg.det(self.H_t[t])
                    if H_t_det <= 0:
                        return 1e10
                    
                    H_t_inv = np.linalg.inv(self.H_t[t])
                    log_likelihood -= 0.5 * (np.log(H_t_det) + eps_t.T @ H_t_inv @ eps_t)
                except:
                    return 1e10
        
        return -log_likelihood
    
    def fit_gas(self, omega_start=0.1, A_start=0.1, B_start=0.8):
        """
        Ajuste le modèle GAS
        """
        print("Ajustement du modèle GAS...")
        
        initial_params = [omega_start, A_start, B_start]
        bounds = [(0.001, 1.0), (0.001, 0.3), (0.5, 0.999)]
        
        result = minimize(
            self._gas_likelihood,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        self.gas_params = result.x
        self.gas_loglikelihood = -result.fun
        
        print(f"Modèle GAS ajusté avec succès : omega = {self.gas_params[0]:.4f}, A = {self.gas_params[1]:.4f}, B = {self.gas_params[2]:.4f}")
        print(f"Log-vraisemblance: {self.gas_loglikelihood:.2f}")
        
        # Calculer une dernière fois les matrices avec les paramètres optimaux
        _ = self._gas_likelihood(self.gas_params)
        
        return result
    
    def get_dynamic_correlations(self):
        """
        Renvoie les corrélations dynamiques entre les paires de tickers
        """
        pairs = []
        correlations = {}
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                pair = f"{self.tickers[i]}-{self.tickers[j]}"
                pairs.append(pair)
                correlations[pair] = [self.R_t[t, i, j] if t > 0 else self.R_t[1, i, j] for t in range(self.T)]
        
        return pd.DataFrame(correlations, index=self.returns.index)
    
    def plot_dynamic_correlations(self, figsize=(12, 8)):
        """
        Trace l'évolution des corrélations dynamiques
        """
        corr_df = self.get_dynamic_correlations()
        
        plt.figure(figsize=figsize)
        for column in corr_df.columns:
            plt.plot(corr_df.index, corr_df[column], label=column)
        
        plt.title("Corrélations Dynamiques GAS")
        plt.xlabel("Date")
        plt.ylabel("Corrélation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_conditional_volatilities(self, figsize=(12, 8)):
        """
        Trace l'évolution des volatilités conditionnelles
        """
        vol_df = np.sqrt(self.conditional_variances)
        
        plt.figure(figsize=figsize)
        for column in vol_df.columns:
            plt.plot(vol_df.index, vol_df[column], label=column)
        
        plt.title("Volatilités Conditionnelles")
        plt.xlabel("Date")
        plt.ylabel("Volatilité")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

# Exemple d'utilisation
if __name__ == "__main__":
    gas_model = GASModel(df_training_set_weekly)
    gas_model.fit_univariate_garch(p=1, q=1)
    gas_model.fit_gas()

    # Récupérer les corrélations dynamiques
    dynamic_corrs = gas_model.get_dynamic_correlations()
    print("\nCorrélations dynamiques:")
    print(dynamic_corrs.head(10))
    
    # # Tracer les corrélations dynamiques
    gas_model.plot_dynamic_correlations()
    plt.show()
    
    # # Tracer les volatilités conditionnelles
    gas_model.plot_conditional_volatilities()
    plt.show()